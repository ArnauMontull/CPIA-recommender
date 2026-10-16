"""Emparejado de peliculas de Letterboxd con el catalogo (movies.parquet): exacto, API de TMDB y sin_emparejar (G5).

1. Exacto: titulo normalizado (minusculas, sin acentos ni puntuacion, articulo final movido al principio:
   «Matrix, The» == «The Matrix»). Se prueba el titulo principal de MovieLens y los alternativos entre parentesis.
   Anio +-0 primero y, si falla, +-1. Con varios candidatos gana el de mas `n_ratings` (y, en empate, el menor movieId).
2. API: /search/movie?query=..&year=.. -> tmdbId -> movieId del catalogo (el de mas n_ratings si el tmdbId se repite).
3. Si tampoco esta en el catalogo: `sin_emparejar` (se ignora; G5).
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ARTICULOS = ("the", "a", "an", "la", "le", "les", "el", "il", "das", "der", "die")
_FINAL = re.compile(r"^(.*),\s*(" + "|".join(ARTICULOS) + r")$")


def normalizar(titulo: str) -> str:
    """Minusculas, sin acentos ni puntuacion; «, The / , A / , An» del final pasa al principio."""
    t = unicodedata.normalize("NFKD", str(titulo)).encode("ascii", "ignore").decode().lower().strip()
    t = t.replace("&", " and ")
    m = _FINAL.match(t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def titulos_catalogo(titulo: str) -> list[str]:
    """Claves normalizadas de un titulo de MovieLens: el principal y los alternativos entre parentesis."""
    titulo = str(titulo)
    principal = titulo.split(" (")[0]
    claves = [normalizar(principal)]
    for alt in re.findall(r"\(([^()]*)\)", titulo):
        alt = re.sub(r"^a\.k\.a\.\s*", "", alt.strip(), flags=re.I)
        if alt and not re.fullmatch(r"\d{4}", alt):
            claves.append(normalizar(alt))
    return [c for c in dict.fromkeys(claves) if c]


class Catalogo:
    """Indices del catalogo para emparejar: por titulo normalizado y por tmdbId."""

    def __init__(self, movies: pd.DataFrame, mapeo: pd.DataFrame):
        cols = ["movieId", "titulo", "anio", "tmdbId", "n_ratings"]
        m = movies[cols].merge(mapeo[["movieId", "item_idx"]], on="movieId", how="left")
        self.item_idx = dict(zip(m.movieId.astype(int), m.item_idx))
        self.por_titulo: dict[str, list[tuple[int | None, int, int]]] = {}
        for mid, tit, anio, nr in zip(m.movieId, m.titulo, m.anio, m.n_ratings):
            a = None if pd.isna(anio) else int(anio)
            for clave in titulos_catalogo(tit):
                self.por_titulo.setdefault(clave, []).append((a, int(mid), int(nr)))
        self.por_tmdb: dict[int, int] = {}
        mejor: dict[int, int] = {}
        for mid, tid, nr in zip(m.movieId, m.tmdbId, m.n_ratings):
            if pd.isna(tid):
                continue
            tid = int(tid)
            if tid not in mejor or (nr, -mid) > (mejor[tid], -self.por_tmdb[tid]):
                mejor[tid], self.por_tmdb[tid] = int(nr), int(mid)

    def exacto(self, nombre: str, anio: int | None) -> int | None:
        cand = self.por_titulo.get(normalizar(nombre), [])
        if not cand:
            return None
        if anio is None:                       # Letterboxd sin anio: vale cualquiera
            return max((nr, -mid, mid) for _, mid, nr in cand)[2]
        for delta in (0, 1):
            ok = [(nr, -mid, mid) for a, mid, nr in cand if a is not None and abs(a - anio) <= delta]
            if ok:
                return max(ok)[2]
        return None

    def por_tmdb_id(self, tmdb_id: int) -> int | None:
        return self.por_tmdb.get(int(tmdb_id))


# --- API de TMDB -----------------------------------------------------------------------------------------

Buscador = Callable[[str, "int | None"], list]     # (nombre, anio) -> [{"id":..,"title":..,"release_date":..}, ...]


def _clave(nombre: str, anio: int | None) -> str:
    return json.dumps([nombre, None if anio is None else int(anio)], ensure_ascii=False)


class BuscadorTMDB:
    """/search/movie con cache en disco (jsonl: una linea por consulta con respuesta 200), concurrencia baja y
    reintentos (los hace `descargar_tmdb.Cliente`). `cliente` debe tener `get_ruta(ruta) -> dict con 'estado'`."""

    def __init__(self, cliente, cache: Path):
        self.cliente, self.cache, self.lock = cliente, Path(cache), threading.Lock()
        self.memoria: dict[str, list] = {}
        if self.cache.exists():
            with self.cache.open(encoding="utf-8") as f:
                for linea in f:
                    if linea.strip():
                        d = json.loads(linea)
                        self.memoria[d["clave"]] = d["resultados"]
        self.llamadas = 0

    def __call__(self, nombre: str, anio: int | None) -> list:
        k = _clave(nombre, anio)
        with self.lock:
            if k in self.memoria:
                return self.memoria[k]
        q = {"query": nombre, "include_adult": "false", "language": "en-US"}
        if anio is not None:
            q["year"] = int(anio)
        resp = self.cliente.get_ruta("/search/movie?" + urllib.parse.urlencode(q))
        if resp.get("estado") != 200:
            return []                    # no se cachea: puede ser un fallo pasajero
        res = [{"id": r["id"], "title": r.get("title"), "release_date": r.get("release_date")}
               for r in resp.get("results", [])[:5]]
        with self.lock:
            self.llamadas += 1
            self.memoria[k] = res
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            with self.cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"clave": k, "resultados": res}, ensure_ascii=False) + "\n")
        return res


def elegir_resultado(resultados: list, anio: int | None) -> int | None:
    """tmdbId del primer resultado de la busqueda (TMDB ya los ordena); si hay anio, el primero con anio +-1."""
    if not resultados:
        return None
    if anio is not None:
        for r in resultados:
            rd = (r.get("release_date") or "")[:4]
            if rd.isdigit() and abs(int(rd) - anio) <= 1:
                return int(r["id"])
    return int(resultados[0]["id"])


def emparejar(peliculas: pd.DataFrame, catalogo: Catalogo, buscar: Buscador | None = None,
              hilos: int = 4) -> pd.DataFrame:
    """peliculas: Name, Year, ... -> mismas filas + movieId (Int32), item_idx (Int32), metodo."""
    nombres = peliculas["Name"].astype(str).tolist()
    anios = [None if pd.isna(a) else int(a) for a in peliculas["Year"]]
    movie_id: list[int | None] = [catalogo.exacto(n, a) for n, a in zip(nombres, anios)]
    metodo = ["exacto" if m is not None else "sin_emparejar" for m in movie_id]

    if buscar is not None:
        pend = [i for i, m in enumerate(movie_id) if m is None]
        with ThreadPoolExecutor(max(1, hilos)) as ex:
            respuestas = list(ex.map(lambda i: buscar(nombres[i], anios[i]), pend))
        for i, res in zip(pend, respuestas):
            tid = elegir_resultado(res, anios[i])
            mid = None if tid is None else catalogo.por_tmdb_id(tid)
            if mid is not None:
                movie_id[i], metodo[i] = mid, "api"

    out = peliculas.copy().reset_index(drop=True)
    out["movieId"] = pd.array(movie_id, dtype="Int32")
    out["item_idx"] = pd.array([None if m is None else catalogo.item_idx[m] for m in movie_id], dtype="Int32")
    out["metodo"] = metodo
    return out
