"""Uso: python -m src.letterboxd --export <zip|carpeta> --nombre <alias> [--sin-api] [--mu X] [--hilos 4]

Salida en data/external/letterboxd/<alias>/ (fuera de git): emparejado.parquet, hist_{A,B}_{indices,pesos}.npy,
valorados_indices.npy y meta.json. La API de TMDB solo se usa si no se pasa --sin-api (credencial: descargar_tmdb.leer_credencial).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ..preprocesamiento.comun import DATA, PROCESADOS, RAIZ
from .construir import ejecutar
from .emparejar import BuscadorTMDB, Catalogo
from .perfil import media_global_train


def crear_buscador(cache: Path) -> BuscadorTMDB:
    """Importa la credencial aqui para que los tests (que no llegan a esta funcion) nunca la lean."""
    from ..preprocesamiento.descargar_tmdb import Cliente, leer_credencial
    return BuscadorTMDB(Cliente(leer_credencial()), cache)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.letterboxd")
    ap.add_argument("--export", required=True, help="zip del export o carpeta con ratings.csv y diary.csv")
    ap.add_argument("--nombre", required=True, help="alias; la salida va a data/<letterboxd.carpeta>/<alias>/")
    ap.add_argument("--sin-api", action="store_true", help="emparejar solo por titulo exacto")
    ap.add_argument("--mu", type=float, default=None, help="media global de train (por defecto, la de dev/senal.parquet)")
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    movies = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId", "titulo", "anio", "tmdbId", "n_ratings"])
    mapeo = pd.read_parquet(PROCESADOS / "dev" / "mapeo_peliculas.parquet")
    mu = a.mu if a.mu is not None else media_global_train(PROCESADOS / "dev")
    carpeta = DATA / cfg["letterboxd"]["carpeta"] / a.nombre
    buscar = None if a.sin_api else crear_buscador(DATA / cfg["letterboxd"]["carpeta"] / "cache_tmdb.jsonl")
    meta = ejecutar(a.export, carpeta, Catalogo(movies, mapeo), len(mapeo), mu, cfg, buscar, a.hilos)
    if buscar is not None:
        meta["llamadas_api_nuevas"] = buscar.llamadas
    print(json.dumps({k: v for k, v in meta.items() if k != "parametros"}, indent=2))
    print("salida:", carpeta)


if __name__ == "__main__":
    main()
