"""Etapa 1 (bloque D): split por usuario y por tiempo, con los índices item_idx / user_idx.

Pasos: quitar los ratings anteriores al estreno (C11); numerar películas y usuarios (H3); de cada usuario, el último 10 %
(por timestamp, desempate por movieId) a test y el 10 % anterior a validación (D1, D2, D4, D7); marcar las primeras 24 h
de cada usuario (C9); y quitar de val/test las películas sin ningún rating en train (D8). Todo vectorizado (sort + cumcount).

Uso: python -m src.preprocesamiento.split --muestra {dev,completo} [--forzar]
Salida en data/processed/<muestra>/: split.parquet, mapeo_peliculas.parquet, mapeo_usuarios.parquet, meta_split.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from .comun import PROCESADOS, RAIZ, marcar

TAREA = "CPIA split"
TRAIN, VAL, TEST = 0, 1, 2


def asignar_indices(movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ratings con user_idx e item_idx, mapeo_peliculas, mapeo_usuarios); item_idx sobre todo `movies` por movieId."""
    mapeo_p = pd.DataFrame({"movieId": np.sort(movies.movieId.to_numpy()).astype("int32")})
    mapeo_p["item_idx"] = np.arange(len(mapeo_p), dtype="int32")
    mapeo_u = pd.DataFrame({"userId": np.unique(ratings.userId.to_numpy()).astype("int32")})
    mapeo_u["user_idx"] = np.arange(len(mapeo_u), dtype="int32")
    r = ratings.merge(mapeo_p, on="movieId", how="left", validate="many_to_one")
    if r.item_idx.isna().any():
        raise ValueError("hay ratings de películas que no están en movies")
    r = r.merge(mapeo_u, on="userId", how="left", validate="many_to_one")
    r["item_idx"] = r.item_idx.astype("int32")
    return r, mapeo_p, mapeo_u


def particionar(r: pd.DataFrame, frac_val: float, frac_test: float, min_por_particion: int) -> pd.DataFrame:
    """Ordena por (user_idx, timestamp, item_idx) y añade `particion` (0 train, 1 val, 2 test) tomada del final de cada usuario."""
    r = r.sort_values(["user_idx", "timestamp", "item_idx"], kind="stable").reset_index(drop=True)
    g = r.groupby("user_idx", sort=False)
    n = g.user_idx.transform("size").to_numpy()
    desde_final = n - 1 - g.cumcount().to_numpy()
    n_test = np.maximum(min_por_particion, np.rint(frac_test * n))
    n_val = np.maximum(min_por_particion, np.rint(frac_val * n))
    part = np.where(desde_final < n_test, TEST, np.where(desde_final < n_test + n_val, VAL, TRAIN))
    part = np.where(n < 2 * min_por_particion + 1, TRAIN, part)  # n < 3: el usuario entero a train
    r["particion"] = part.astype("int8")
    return r


def marcar_primer_dia(r: pd.DataFrame, segundos: int) -> pd.DataFrame:
    """Añade `primer_dia` = timestamp < primer timestamp del usuario + segundos (C9)."""
    primero = r.groupby("user_idx").timestamp.transform("min")
    r = r.copy()
    r["primer_dia"] = (r.timestamp < primero + segundos).to_numpy()
    return r


def quitar_frias(r: pd.DataFrame) -> pd.DataFrame:
    """Quita de val/test las filas de películas sin ningún rating en train (D8)."""
    en_train = np.unique(r.item_idx.to_numpy()[r.particion.to_numpy() == TRAIN])
    fria = (r.particion.to_numpy() != TRAIN) & ~np.isin(r.item_idx.to_numpy(), en_train)
    return r[~fria].reset_index(drop=True)


def usuarios_corte_empate(r: pd.DataFrame) -> int:
    """Usuarios con un corte de partición entre dos ratings del mismo segundo (r ordenado y con `particion`)."""
    u, p, t = (r[c].to_numpy() for c in ("user_idx", "particion", "timestamp"))
    corte = (u[1:] == u[:-1]) & (p[1:] != p[:-1]) & (t[1:] == t[:-1])
    return int(np.unique(u[1:][corte]).size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    s = cfg["split"]
    salida = PROCESADOS / a.muestra
    salida.mkdir(parents=True, exist_ok=True)
    if (salida / "split.parquet").exists() and not a.forzar:
        print(f"{salida / 'split.parquet'} ya existe; usa --forzar para rehacerlo")
        return

    marcar(TAREA, 0, 5, "leyendo parquet")
    r = pd.read_parquet(PROCESADOS / cfg["muestras"][a.muestra])
    movies = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId"])
    n_entrada = len(r)

    marcar(TAREA, 1, 5, "índices y C11")
    r, mapeo_p, mapeo_u = asignar_indices(movies, r)  # antes de C11: el mapeo cubre a todos los usuarios de la muestra
    if s["quitar_antes_estreno"]:
        r = r[~r.antes_estreno].reset_index(drop=True)
    n_c11 = n_entrada - len(r)

    marcar(TAREA, 2, 5, "partición por usuario")
    r = particionar(r, s["frac_val"], s["frac_test"], s["min_por_particion"])
    n_empate = usuarios_corte_empate(r)
    r = marcar_primer_dia(r, cfg["senal"]["segundos_primer_dia"])
    antes = np.bincount(r.particion.to_numpy(), minlength=3)

    marcar(TAREA, 3, 5, "D8: quitar películas frías")
    if s["quitar_frias_eval"]:
        r = quitar_frias(r)
    despues = np.bincount(r.particion.to_numpy(), minlength=3)

    marcar(TAREA, 4, 5, "guardando")
    r = r[["user_idx", "item_idx", "rating", "timestamp", "particion", "primer_dia"]].astype(
        {"user_idx": "int32", "item_idx": "int32", "rating": "float32", "timestamp": "int64", "particion": "int8"})
    r.to_parquet(salida / "split.parquet", index=False)
    mapeo_p.to_parquet(salida / "mapeo_peliculas.parquet", index=False)
    mapeo_u.to_parquet(salida / "mapeo_usuarios.parquet", index=False)
    meta = {
        "muestra": a.muestra,
        "filas_entrada": int(n_entrada),
        "quitadas_c11_antes_estreno": int(n_c11),
        "quitadas_d8_val": int(antes[VAL] - despues[VAL]),
        "quitadas_d8_test": int(antes[TEST] - despues[TEST]),
        "filas_train": int(despues[TRAIN]),
        "filas_val": int(despues[VAL]),
        "filas_test": int(despues[TEST]),
        "n_usuarios": int(len(mapeo_u)),
        "n_peliculas": int(len(mapeo_p)),
        "usuarios_solo_train": int((r.groupby("user_idx").particion.max() == TRAIN).sum()),
        "usuarios_corte_en_empate_de_segundo": n_empate,
        "filas_primer_dia": int(r.primer_dia.sum()),
    }
    (salida / "meta_split.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    marcar(TAREA, 5, 5, "hecho")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
