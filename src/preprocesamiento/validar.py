"""Etapa 1 (B4): validación de integridad de MovieLens 32M y TMDB.

Corrige lo que se puede corregir sin tomar decisiones de modelado y marca (flag) el resto. Además fusiona los
movieId duplicados (mismo tmdbId) y elimina las películas con menos de --min-ratings ratings (B1).
Si existe data/external/tmdb_api.jsonl (descargar_tmdb.py), rellena con él las sinopsis que faltan en Kaggle.
Salidas:
  data/processed/ratings.parquet   ratings limpios (+ flag antes_estreno)
  data/processed/movies.parquet    catálogo con año, géneros en lista, tmdbId, n_ratings y flags
  data/processed/fusiones.parquet  movieId absorbido -> movieId que se queda
  reports/validacion.md            qué se ha comprobado, cuántas filas y qué se ha hecho

Uso: python -m src.preprocesamiento.validar [--min-ratings 20]   (desde la raíz del repo)
"""
from __future__ import annotations

import argparse
import ast
import time

import numpy as np
import pandas as pd

from .comun import CRUDOS_ML, CRUDOS_TMDB, DATA, INFORMES, PROCESADOS, leer_ratings, marcar

TAREA = "CPIA validar"
API = DATA / "external" / "tmdb_api.jsonl"
API_CORR = DATA / "external" / "tmdb_correcciones.jsonl"
# Géneros de TMDB (películas y series) -> vocabulario de MovieLens; los que no tienen equivalente se ignoran
TMDB_A_ML = {
    "Action": ["Action"], "Adventure": ["Adventure"], "Action & Adventure": ["Action", "Adventure"],
    "Animation": ["Animation"], "Comedy": ["Comedy"], "Crime": ["Crime"], "Documentary": ["Documentary"],
    "Drama": ["Drama"], "Family": ["Children"], "Kids": ["Children"], "Fantasy": ["Fantasy"],
    "Science Fiction": ["Sci-Fi"], "Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"], "Horror": ["Horror"],
    "Mystery": ["Mystery"], "Romance": ["Romance"], "Thriller": ["Thriller"], "War": ["War"],
    "War & Politics": ["War"], "Western": ["Western"], "Music": ["Musical"],
}


def a_generos_ml(generos_tmdb: list[str]) -> list[str]:
    return sorted({g for t in generos_tmdb for g in TMDB_A_ML.get(t, [])})
NOTAS_VALIDAS = np.arange(0.5, 5.01, 0.5, dtype="float32")
# MovieLens 32M: ratings entre 1995 y octubre de 2023 (README del dataset)
TS_MIN = int(pd.Timestamp("1995-01-01").timestamp())
TS_MAX = int(pd.Timestamp("2023-11-01").timestamp())
SINOPSIS_VACIAS = {"", "no overview found.", "no overview", "no movie overview available."}

filas: list[tuple[str, str, int, str]] = []  # (tabla, comprobación, afectadas, acción)


def apunta(tabla: str, comprobacion: str, n: int, accion: str) -> None:
    filas.append((tabla, comprobacion, int(n), accion))
    print(f"[{tabla}] {comprobacion}: {n} -> {accion}")


def validar_movies() -> pd.DataFrame:
    m = pd.read_csv(CRUDOS_ML / "movies.csv", dtype={"movieId": "int32"})
    apunta("movies", "filas leídas", len(m), "—")
    apunta("movies", "movieId duplicado", m.movieId.duplicated().sum(), "se conserva la primera")
    m = m.drop_duplicates("movieId")

    m["titulo"] = m.title.str.strip()
    anio = m.titulo.str.extract(r"\((\d{4})(?:[-–]\d{0,4})?\)\s*$")[0]
    m["anio"] = pd.to_numeric(anio, errors="coerce").astype("Int16")
    apunta("movies", "sin año en el título", m.anio.isna().sum(), "flag sin_anio (anio = NA)")
    m["titulo"] = m.titulo.str.replace(r"\s*\(\d{4}(?:[-–]\d{0,4})?\)\s*$", "", regex=True)

    m["sin_generos"] = m.genres.eq("(no genres listed)")
    m["generos"] = m.genres.where(~m.sin_generos, "").str.split("|").map(lambda g: [x for x in g if x])
    apunta("movies", "«(no genres listed)»", m.sin_generos.sum(), "flag sin_generos, lista vacía (imputación: decisión B3)")

    dup = m.duplicated(["titulo", "anio"], keep=False) & m.anio.notna()
    m["titulo_duplicado"] = dup
    apunta("movies", "mismo título y año con distinto movieId", dup.sum(), "flag titulo_duplicado")

    links = pd.read_csv(CRUDOS_ML / "links.csv", dtype={"movieId": "int32", "tmdbId": "Int64"})
    m = m.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    apunta("links", "película sin tmdbId", m.tmdbId.isna().sum(), "flag (no podrá usar metadatos TMDB)")
    rep = m.tmdbId.notna() & m.duplicated("tmdbId", keep=False)
    m["tmdb_compartido"] = rep
    apunta("links", "tmdbId compartido por varios movieId", rep.sum(), "flag tmdb_compartido")
    return m


def validar_tmdb(m: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(CRUDOS_TMDB / "movies_metadata.csv", low_memory=False,
                       usecols=["id", "genres", "overview"])
    apunta("tmdb", "filas leídas", len(meta), "—")
    meta["id"] = pd.to_numeric(meta.id, errors="coerce").astype("Int64")
    apunta("tmdb", "id no numérico (filas desplazadas del CSV)", meta.id.isna().sum(), "se eliminan")
    meta = meta.dropna(subset=["id"])
    apunta("tmdb", "id duplicado", meta.id.duplicated().sum(), "se conserva la primera")
    meta = meta.drop_duplicates("id")

    def nombres(s):
        try:
            return [d["name"] for d in ast.literal_eval(s)]
        except (ValueError, SyntaxError, TypeError):
            return []

    meta["generos_tmdb"] = meta.genres.map(nombres)
    sinopsis = meta.overview.fillna("").str.strip()
    vacia = sinopsis.str.lower().isin(SINOPSIS_VACIAS)
    apunta("tmdb", "sinopsis vacía o de relleno («No overview found.»…)", vacia.sum(), "se deja vacía")
    meta["sinopsis"] = sinopsis.where(~vacia, "")

    m = m.merge(meta[["id", "generos_tmdb", "sinopsis"]], left_on="tmdbId", right_on="id", how="left").drop(columns="id")
    m["en_tmdb"] = m.sinopsis.notna()
    m["sinopsis"] = m.sinopsis.fillna("")
    m["generos_tmdb"] = m.generos_tmdb.map(lambda g: g if isinstance(g, list) else [])
    m["fuente_sinopsis"] = np.where(m.sinopsis.str.len() > 0, "kaggle", "")

    # Lo descargado de la API (descargar_tmdb.py) rellena los huecos de Kaggle
    if API.exists():
        api = pd.read_json(API, lines=True)
        api = api[api.estado == 200].drop_duplicates("tmdbId", keep="last").set_index("tmdbId")
        hueco = (m.sinopsis.str.len() == 0) & m.tmdbId.isin(api.index)
        desde_api = api.loc[m.tmdbId[hueco].astype(int)]
        m.loc[hueco, "sinopsis"] = desde_api.sinopsis.fillna("").to_numpy()
        m.loc[hueco, "generos_tmdb"] = pd.Series(list(desde_api.generos), index=m.index[hueco])
        m.loc[hueco & (m.sinopsis.str.len() > 0), "fuente_sinopsis"] = "api"
        m.loc[hueco, "en_tmdb"] = True
        apunta("movies×tmdb", "sinopsis recuperadas de la API de TMDB", (m.fuente_sinopsis == "api").sum(), "rellenadas")

    # Correcciones manuales (correcciones_tmdb.csv): tmdbId roto, casi siempre series o episodios
    m["es_serie"] = False
    m["generos_imputados"] = False
    if API_CORR.exists():
        corr = pd.read_json(API_CORR, lines=True)
        corr = corr[corr.estado == 200].set_index("movieId")
        idx = m.index[m.movieId.isin(corr.index)]
        c = corr.loc[m.loc[idx, "movieId"]]
        m.loc[idx, "sinopsis"] = c.sinopsis.fillna("").to_numpy()
        m.loc[idx, "generos_tmdb"] = pd.Series(list(c.generos), index=idx)
        m.loc[idx, "fuente_sinopsis"] = "correccion"
        m.loc[idx, "en_tmdb"] = True
        m.loc[idx, "es_serie"] = c.tipo.isin(["tv", "episodio"]).to_numpy()
        # Si MovieLens no les da géneros, se traducen los de TMDB a su vocabulario
        vacios = idx[m.loc[idx, "sin_generos"].to_numpy()]
        m.loc[vacios, "generos"] = pd.Series([a_generos_ml(g) for g in m.loc[vacios, "generos_tmdb"]], index=vacios)
        m.loc[vacios, "generos_imputados"] = m.loc[vacios, "generos"].map(len) > 0
        m.loc[vacios, "sin_generos"] = ~m.loc[vacios, "generos_imputados"]
        apunta("movies×tmdb", "corregidas a mano (series, episodios, vídeos)", len(idx),
               f"sinopsis de TMDB; géneros traducidos al vocabulario de MovieLens en {m.generos_imputados.sum()}; flag es_serie")
    m["tiene_sinopsis"] = m.sinopsis.str.len() > 0
    apunta("movies×tmdb", "película de ML sin fila en TMDB Kaggle (2017)", (~m.en_tmdb).sum(), "flag en_tmdb = False")
    apunta("movies×tmdb", "película de ML sin sinopsis", (~m.tiene_sinopsis).sum(), "flag tiene_sinopsis = False (B2: se mantiene)")
    return m


def validar_ratings(m: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    r = leer_ratings()
    marcar(TAREA, 3, 7, "ratings leídos")
    apunta("ratings", "filas leídas", len(r), f"lectura en {time.time() - t0:.1f} s")

    nulos = r.isna().any(axis=1)
    apunta("ratings", "valores nulos", nulos.sum(), "se eliminan")
    r = r[~nulos]

    invalida = ~np.isin(r.rating.to_numpy(), NOTAS_VALIDAS)
    apunta("ratings", "nota fuera de {0.5, 1.0, …, 5.0}", invalida.sum(), "se eliminan")
    r = r[~invalida]

    huerfana = ~r.movieId.isin(m.movieId)
    apunta("ratings", "movieId inexistente en movies.csv", huerfana.sum(), "se eliminan")
    r = r[~huerfana]

    ts = r.timestamp.to_numpy()
    fuera = (ts < TS_MIN) | (ts > TS_MAX)
    apunta("ratings", "timestamp fuera de 1995-01-01 … 2023-10-31", fuera.sum(), "se eliminan")
    r = r[~fuera]
    marcar(TAREA, 4, 7, "ratings: nulos, notas y fechas")

    # Duplicado (usuario, película): clave única en int64 y nos quedamos con la valoración más reciente
    clave = r.userId.to_numpy().astype("int64") * 1_000_000 + r.movieId.to_numpy()
    r = r.assign(_k=clave).sort_values(["_k", "timestamp"])
    dup = r._k.duplicated(keep="last")
    apunta("ratings", "par (usuario, película) repetido", dup.sum(), "se conserva el más reciente")
    r = r[~dup].drop(columns="_k")
    marcar(TAREA, 5, 7, "ratings: duplicados")

    anio_peli = r.movieId.map(m.set_index("movieId").anio).astype("float32")
    anio_rating = pd.to_datetime(r.timestamp, unit="s").dt.year
    antes = (anio_rating < anio_peli - 1).to_numpy()  # margen de 1 año por festivales y preestrenos
    apunta("ratings", "valorada más de 1 año antes del estreno", antes.sum(), "flag antes_estreno (no se borra)")
    r["antes_estreno"] = antes

    return r


def fusionar_duplicados(r: pd.DataFrame, m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Varios movieId con el mismo tmdbId son la misma película: se quedan en el que tiene más ratings."""
    n = r.movieId.value_counts()
    dup = m[m.tmdb_compartido].assign(n=lambda d: d.movieId.map(n).fillna(0)).sort_values(["tmdbId", "n"], ascending=[True, False])
    canon = dup.groupby("tmdbId").movieId.transform("first")
    fus = dup.assign(destino=canon)[lambda d: d.movieId != d.destino][["movieId", "destino", "titulo", "anio", "tmdbId", "n"]]
    apunta("movies", "duplicados por tmdbId fusionados", len(fus), f"{len(fus)} movieId absorbidos en otro (ver data/processed/fusiones.parquet)")

    mapa = dict(zip(fus.movieId, fus.destino))
    r = r.assign(movieId=r.movieId.map(lambda x: mapa.get(x, x)).astype("int32"))
    clave = r.userId.to_numpy().astype("int64") * 1_000_000 + r.movieId.to_numpy()
    r = r.assign(_k=clave).sort_values(["_k", "timestamp"])
    dup_r = r._k.duplicated(keep="last")
    apunta("ratings", "usuario que valoró las dos copias de un duplicado", dup_r.sum(), "se conserva la nota más reciente")
    r = r[~dup_r].drop(columns="_k")
    m = m[~m.movieId.isin(fus.movieId)].copy()
    m["tmdb_compartido"] = m.tmdbId.notna() & m.duplicated("tmdbId", keep=False)
    return r, m, fus


def filtrar_peliculas(r: pd.DataFrame, m: pd.DataFrame, minimo: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decisión B1 (revisada el 15/10/2026): fuera las películas con menos de `minimo` ratings, y sus ratings."""
    n = r.movieId.value_counts()
    m["n_ratings"] = m.movieId.map(n).fillna(0).astype("int32")
    fuera = m.n_ratings < minimo
    apunta("movies", f"películas con < {minimo} ratings", fuera.sum(),
           f"se eliminan ({m.n_ratings[fuera].sum():,}".replace(",", ".") + f" ratings; {m.n_ratings[fuera].sum() / len(r):.2%})")
    m = m[~fuera].reset_index(drop=True)
    r = r[r.movieId.isin(m.movieId)]
    por_usuario = r.groupby("userId").size()
    apunta("ratings", "usuarios", len(por_usuario), "—")
    apunta("ratings", "usuarios que quedan con < 20 ratings tras el filtro", (por_usuario < 20).sum(), "se mantienen")
    return r.sort_values(["userId", "timestamp"]).reset_index(drop=True), m


def escribir_informe(r: pd.DataFrame, m: pd.DataFrame, fus: pd.DataFrame, minimo: int) -> None:
    INFORMES.mkdir(exist_ok=True)
    mm = m.set_index("movieId")
    cobertura = mm.loc[r.movieId.to_numpy(), "tiene_sinopsis"].mean()
    lineas = [
        "# Validación de integridad (B4)",
        "",
        f"Generado por `python -m src.preprocesamiento.validar --min-ratings {minimo}`. Corrige lo que no implica"
        " decisiones de modelado y marca el resto con columnas booleanas en `data/processed/*.parquet`.",
        "",
        "| Tabla | Comprobación | Filas afectadas | Acción |",
        "|---|---|---:|---|",
        *[f"| {t} | {c} | {n:,} | {a} |".replace(",", ".") for t, c, n, a in filas],
        "",
        "## Resumen (después de fusionar y filtrar)",
        f"- Ratings: {len(r):,}; usuarios: {r.userId.nunique():,}; películas: {len(m):,}.".replace(",", "."),
        f"- Películas con sinopsis: {m.tiene_sinopsis.sum():,} ({m.tiene_sinopsis.mean():.1%});"
        f" ratings cuya película tiene sinopsis: **{cobertura:.1%}**.".replace(",", "."),
        f"- Películas sin géneros ni sinopsis: {(m.sin_generos & ~m.tiene_sinopsis).sum()}.",
        "",
        "## Duplicados fusionados (mismo tmdbId)",
        "| movieId absorbido | → movieId | Título | Año | tmdbId | Ratings que aporta |",
        "|---:|---:|---|---:|---:|---:|",
        *[f"| {f.movieId} | {f.destino} | {f.titulo} | {f.anio} | {f.tmdbId} | {int(f.n)} |" for f in fus.itertuples()],
    ]
    (INFORMES / "validacion.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratings", type=int, default=20, help="mínimo de ratings por película (B1)")
    a = ap.parse_args()

    PROCESADOS.mkdir(parents=True, exist_ok=True)
    marcar(TAREA, 0, 7, "movies y links")
    m = validar_movies()
    marcar(TAREA, 1, 7, "TMDB")
    m = validar_tmdb(m)
    marcar(TAREA, 2, 7, "leyendo ratings (32M)")
    r = validar_ratings(m)
    marcar(TAREA, 6, 7, "fusión, filtro y guardado")
    r, m, fus = fusionar_duplicados(r, m)
    r, m = filtrar_peliculas(r, m, a.min_ratings)
    r.to_parquet(PROCESADOS / "ratings.parquet", index=False)
    m.drop(columns=["title", "genres"]).to_parquet(PROCESADOS / "movies.parquet", index=False)
    fus.to_parquet(PROCESADOS / "fusiones.parquet", index=False)
    escribir_informe(r, m, fus, a.min_ratings)
    marcar(TAREA, 7, 7, "hecho")


if __name__ == "__main__":
    main()
