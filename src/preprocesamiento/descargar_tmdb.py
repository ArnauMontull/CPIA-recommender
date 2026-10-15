"""Etapa 0 (opcional): descarga de la API de TMDB los metadatos de las películas que faltan en Kaggle (2017).

Para cada tmdbId de MovieLens sin sinopsis en Kaggle pide /movie/{id} con keywords y créditos, y guarda una
línea JSON por película en data/external/tmdb_api.jsonl. Se puede interrumpir: al relanzarlo salta las ya hechas.
validar.py usa este fichero si existe.

Credencial (nunca en git): variable de entorno TMDB_TOKEN, o el primer fichero que exista de
  <LIFE_OS>/secrets/token tmdb.json, <LIFE_OS>/secrets/tmdb.json, ./secrets/tmdb.json
con {"token": "..."} (o el texto tal cual). Vale el token de lectura (eyJ…) o la clave v3 (32 hex).

Uso: python -m src.preprocesamiento.descargar_tmdb [--limite N] [--hilos 8]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .comun import CRUDOS_ML, CRUDOS_TMDB, DATA, PROCESADOS, RAIZ, marcar

TAREA = "CPIA TMDB API"
SALIDA = DATA / "external" / "tmdb_api.jsonl"
URL = "https://api.themoviedb.org/3/movie/{id}?language=en-US&append_to_response=keywords,credits"
SINOPSIS_VACIAS = {"", "no overview found.", "no overview", "no movie overview available."}


def leer_credencial() -> str:
    if os.environ.get("TMDB_TOKEN"):
        return os.environ["TMDB_TOKEN"].strip()
    for p in [RAIZ.parents[2] / "secrets" / "token tmdb.json", RAIZ.parents[2] / "secrets" / "tmdb.json",
              RAIZ / "secrets" / "tmdb.json"]:
        if p.exists():
            texto = p.read_text(encoding="utf-8-sig").strip()
            try:
                datos = json.loads(texto)
            except json.JSONDecodeError:
                return texto
            if isinstance(datos, str):
                return datos.strip()
            return next(v for v in datos.values() if isinstance(v, str)).strip()
    raise SystemExit("No hay credencial de TMDB: define TMDB_TOKEN o crea secrets/tmdb.json")


def pendientes() -> list[int]:
    """tmdbIds sin sinopsis. Si ya existe movies.parquet (validar.py), solo las películas que pasaron el filtro."""
    movies = PROCESADOS / "movies.parquet"
    if movies.exists():
        m = pd.read_parquet(movies, columns=["tmdbId", "tiene_sinopsis"])
        ids = {int(t) for t in m.tmdbId[~m.tiene_sinopsis].dropna()}
    else:
        links = pd.read_csv(CRUDOS_ML / "links.csv", dtype={"tmdbId": "Int64"})
        meta = pd.read_csv(CRUDOS_TMDB / "movies_metadata.csv", usecols=["id", "overview"], low_memory=False)
        meta["id"] = pd.to_numeric(meta.id, errors="coerce").astype("Int64")
        ok = meta.overview.fillna("").str.strip().str.lower().map(lambda s: s not in SINOPSIS_VACIAS)
        con_sinopsis = set(meta.id[ok].dropna().astype(int))
        ids = {int(t) for t in links.tmdbId.dropna()} - con_sinopsis
    hechos = set()
    if SALIDA.exists():
        with SALIDA.open(encoding="utf-8") as f:
            hechos = {json.loads(l)["tmdbId"] for l in f if l.strip()}
    return sorted(ids - hechos)


class Cliente:
    def __init__(self, credencial: str):
        self.es_token = credencial.startswith("eyJ")
        self.cred = credencial
        self.pausa_hasta = 0.0
        self.lock = threading.Lock()

    def get(self, tmdb_id: int) -> dict:
        url = URL.format(id=tmdb_id)
        cab = {"accept": "application/json"}
        if self.es_token:
            cab["Authorization"] = f"Bearer {self.cred}"
        else:
            url += f"&api_key={self.cred}"
        for intento in range(6):
            espera = self.pausa_hasta - time.time()
            if espera > 0:
                time.sleep(espera)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=cab), timeout=20) as resp:
                    return {"estado": 200, **json.load(resp)}
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return {"estado": 404}
                if e.code == 401:
                    raise SystemExit("TMDB responde 401: la credencial no es válida")
                if e.code == 429 or e.code >= 500:
                    with self.lock:
                        self.pausa_hasta = time.time() + float(e.headers.get("Retry-After", 2 ** intento))
                    continue
                return {"estado": e.code}
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** intento)
        return {"estado": -1}


def resumir(tmdb_id: int, d: dict) -> dict:
    """Solo lo que usaremos; no se guarda la respuesta entera."""
    if d["estado"] != 200:
        return {"tmdbId": tmdb_id, "estado": d["estado"]}
    cred = d.get("credits", {})
    return {
        "tmdbId": tmdb_id,
        "estado": 200,
        "titulo": d.get("title"),
        "sinopsis": (d.get("overview") or "").strip(),
        "generos": [g["name"] for g in d.get("genres", [])],
        "estreno": d.get("release_date"),
        "duracion": d.get("runtime"),
        "idioma": d.get("original_language"),
        "keywords": [k["name"] for k in d.get("keywords", {}).get("keywords", [])],
        "director": [c["name"] for c in cred.get("crew", []) if c.get("job") == "Director"],
        "reparto": [c["name"] for c in sorted(cred.get("cast", []), key=lambda c: c.get("order", 99))[:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0, help="descargar solo N (para probar)")
    ap.add_argument("--hilos", type=int, default=8)
    a = ap.parse_args()

    cliente = Cliente(leer_credencial())
    ids = pendientes()
    if a.limite:
        ids = ids[: a.limite]
    print(f"{len(ids)} películas por descargar ({'token' if cliente.es_token else 'clave v3'})")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)

    estados: dict[int, int] = {}
    t0 = time.time()
    with SALIDA.open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.hilos) as ex:
        futuros = {ex.submit(cliente.get, i): i for i in ids}
        for n, fut in enumerate(as_completed(futuros), 1):
            fila = resumir(futuros[fut], fut.result())
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            estados[fila["estado"]] = estados.get(fila["estado"], 0) + 1
            if n % 100 == 0 or n == len(ids):
                f.flush()
                marcar(TAREA, n, len(ids), f"{n / (time.time() - t0):.0f} pelis/s")
    print("estados HTTP:", estados, f"en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
