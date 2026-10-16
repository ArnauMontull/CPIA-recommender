"""Experimento de arranque en frío de usuarios (G6): split por usuarios, con las mismas salidas que la pipeline normal.

Usuarios fríos: el `frac_usuarios` de los usuarios, elegidos con un hash estable de `userId` + semilla (D10). Sus ratings no
entran nunca en train: ordenados por (timestamp, movieId), el primer `frac_historia` es la historia dada (`particion` 3) y el
resto el test (`particion` 2). Un frío con menos de MIN_RATINGS_FRIO ratings queda fuera del experimento.
Usuarios normales: train y val como `split.particionar`, pero su test pasa a train (train = todo menos su val), así val sigue
sirviendo para el early stopping y el test solo lo forman los fríos.
media_u y P(i) salen solo de train (D3). La media_u de un frío sale de su historia dada, encogida como en G4 si tiene menos de
`min_notas_sin_encoger` notas: (Σr + λ·μ)/(n + λ), con μ = media de los ratings de train. D8: fuera de val/test las películas
sin ratings en train (la historia dada no se toca: es entrada, no objetivo).
Historiales (F): vistas `train`, `train_val` (particiones 0, 1 y 3: para un frío = su historia dada) y `dada` (solo partición 3).
Máscara C14 del test de los fríos = su historia dada (`valorados_dada_*`; `valorados_train_val_*` coincide en los fríos).

Uso: python -m src.preprocesamiento.frio --muestra {dev,completo} [--forzar]
Salida en data/processed/<muestra>_frio/: los archivos de split, senal e historial, hist_dada_*, valorados_dada_* y meta_frio.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from . import historial as H
from . import senal as S
from . import split as SP
from .comun import PROCESADOS, RAIZ, marcar

TAREA = "CPIA frio"
MIN_RATINGS_FRIO = 5
TRAIN, VAL, TEST, DADA = 0, 1, 2, 3
VISTAS = {"train": (0,), "train_val": (0, 1, 3), "dada": (3,)}   # vista -> particiones incluidas
SALIDAS = (["split.parquet", "senal.parquet", "usuarios.parquet", "logq.npy", "valorados_train_indptr.npy",
            "valorados_train_indices.npy", "valorados_train_val_indptr.npy", "valorados_train_val_indices.npy",
            "valorados_dada_indptr.npy", "valorados_dada_indices.npy", "ejemplos_train.parquet",
            "mapeo_peliculas.parquet", "mapeo_usuarios.parquet", "meta_frio.json"]
           + [f"hist_{v}_{b}_{x}.npy" for v in VISTAS for b in H.BOLSAS for x in ("indptr", "indices", "pesos")])


def hash_estable(ids: np.ndarray, semilla: int) -> np.ndarray:
    """splitmix64 de (id + semilla·phi): uint64 determinista entre ejecuciones y plataformas (no usa hash() de Python)."""
    phi = 0x9E3779B97F4A7C15
    c = lambda v: np.uint64(v)  # noqa: E731
    x = ids.astype("uint64") + c((int(semilla) * phi) % 2**64)
    x = x + c(phi)
    z = (x ^ (x >> c(30))) * c(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> c(27))) * c(0x94D049BB133111EB)
    return z ^ (z >> c(31))


def elegir_frios(user_ids: np.ndarray, frac: float, semilla: int) -> np.ndarray:
    """Bool [len(user_ids)]: los round(frac·U) usuarios con menor hash(userId, semilla). No depende del orden de entrada."""
    h = hash_estable(np.asarray(user_ids), semilla)
    n = int(np.rint(frac * len(h)))
    orden = np.lexsort((np.asarray(user_ids), h))     # empate de hash (casi imposible): por userId
    out = np.zeros(len(h), dtype=bool)
    out[orden[:n]] = True
    return out


def particionar_frio(r: pd.DataFrame, frios: np.ndarray, frac_val: float, frac_test: float, min_por_particion: int,
                     frac_historia: float, min_ratings: int = MIN_RATINGS_FRIO) -> tuple[pd.DataFrame, dict]:
    """r con user_idx, item_idx, timestamp; frios = bool indexado por user_idx. Devuelve (r con `particion` 0/1/2/3, meta).
    Normales: particionar normal y su test pasa a train. Fríos: dada (3) / test (2); los de < min_ratings desaparecen."""
    es_frio = frios[r.user_idx.to_numpy()]
    normales = SP.particionar(r[~es_frio], frac_val, frac_test, min_por_particion)
    normales["particion"] = np.where(normales.particion == TEST, TRAIN, normales.particion).astype("int8")

    f = r[es_frio].sort_values(["user_idx", "timestamp", "item_idx"], kind="stable").reset_index(drop=True)
    n = f.groupby("user_idx", sort=False).user_idx.transform("size").to_numpy()
    k = f.groupby("user_idx", sort=False).cumcount().to_numpy()
    n_dada = np.clip(np.rint(frac_historia * n), 1, np.maximum(n - 1, 1))
    f["particion"] = np.where(k < n_dada, DADA, TEST).astype("int8")
    pocos = n < min_ratings
    meta = {"frios_con_ratings": int(f.user_idx.nunique()),
            "frios_fuera_menos_de_min": int(f.user_idx[pocos].nunique()),
            "ratings_frios_fuera": int(pocos.sum())}
    f = f[~pocos]
    out = pd.concat([normales, f], ignore_index=True).sort_values(
        ["user_idx", "timestamp", "item_idx"], kind="stable").reset_index(drop=True)
    return out, meta


def quitar_frias_eval(r: pd.DataFrame) -> pd.DataFrame:
    """D8: quita de val y test las películas sin ningún rating en train. La historia dada (3) no se toca."""
    p, it = r.particion.to_numpy(), r.item_idx.to_numpy()
    en_train = np.unique(it[p == TRAIN])
    fria = ((p == VAL) | (p == TEST)) & ~np.isin(it, en_train)
    return r[~fria].reset_index(drop=True)


def medias_usuario(df: pd.DataFrame, n_usuarios: int, min_sin_encoger: int, lam: float) -> tuple[np.ndarray, float]:
    """media_u [U]: normales = media cruda de train; fríos = media de su historia dada, encogida (G4) si n < min_sin_encoger.
    Devuelve también μ (media de los ratings de train)."""
    u, rt, p = df.user_idx.to_numpy(), df.rating.to_numpy().astype("float64"), df.particion.to_numpy()
    media = S.medias_train(u, rt, p, n_usuarios)
    mu = float(rt[p == TRAIN].mean())
    d = p == DADA
    n = np.bincount(u[d], minlength=n_usuarios)
    suma = np.bincount(u[d], weights=rt[d], minlength=n_usuarios)
    con = n > 0
    cruda = np.where(con, suma / np.maximum(n, 1), np.nan)
    encogida = np.where(n < min_sin_encoger, (suma + lam * mu) / (n + lam), cruda)
    return np.where(con, encogida, media), mu


def construir_frio(r: pd.DataFrame, frios: np.ndarray, cfg: dict, n_usuarios: int, n_items: int) -> dict:
    """Todo el cálculo en memoria. r: user_idx, item_idx, rating, timestamp (ya sin C11); frios: bool por user_idx."""
    s, c_s, c_h = cfg["split"], cfg["senal"], cfg["historial"]
    r, meta = particionar_frio(r, frios, s["frac_val"], s["frac_test"], s["min_por_particion"],
                               cfg["frio"]["frac_historia"],
                               cfg["frio"].get("min_ratings", MIN_RATINGS_FRIO))
    r = SP.marcar_primer_dia(r, c_s["segundos_primer_dia"])
    antes = np.bincount(r.particion.to_numpy(), minlength=4)
    r = quitar_frias_eval(r)
    despues = np.bincount(r.particion.to_numpy(), minlength=4)
    meta.update({"quitadas_d8_val": int(antes[VAL] - despues[VAL]), "quitadas_d8_test": int(antes[TEST] - despues[TEST]),
                 "filas_train": int(despues[TRAIN]), "filas_val": int(despues[VAL]),
                 "filas_test": int(despues[TEST]), "filas_dada": int(despues[DADA])})
    r = r[["user_idx", "item_idx", "rating", "timestamp", "particion", "primer_dia"]].astype(
        {"user_idx": "int32", "item_idx": "int32", "rating": "float32", "timestamp": "int64", "particion": "int8"})

    u, i, p = r.user_idx.to_numpy(), r.item_idx.to_numpy(), r.particion.to_numpy()
    media, mu = medias_usuario(r, n_usuarios, cfg["letterboxd"]["min_notas_sin_encoger"],
                               cfg["letterboxd"]["lambda_encogimiento"])
    senal = S.etiquetar(r, media, c_s)
    senal["peso"] = S.pesos(senal, media, c_s)
    logq = S.logq(i, senal.positivo.to_numpy(), p, n_items, c_s["suavizado_logq"])
    ip, ix = S.mascara_valorados(u, i, p, n_usuarios, n_items)

    usuarios = S.usuarios_elegibles(senal, media, n_usuarios)
    es_frio = frios.copy()
    pos, rel = senal.positivo.to_numpy(), senal.relevante_4.to_numpy()
    cuenta = lambda m: np.bincount(u[m], minlength=n_usuarios)  # noqa: E731
    n_dada, n_pos_dada = cuenta(p == DADA), cuenta(pos & (p == DADA))
    usuarios["frio"] = es_frio & (n_dada > 0)
    usuarios["n_dada"] = n_dada.astype("int32")
    usuarios["n_pos_dada"] = n_pos_dada.astype("int32")
    fr = usuarios.frio.to_numpy()
    # C8 para un frío: >= 1 positivo en su historia dada (no tiene train) y >= 1 en el test
    usuarios.loc[fr, "elegible_test"] = (n_pos_dada >= 1)[fr] & (cuenta(pos & (p == TEST)) >= 1)[fr]
    usuarios.loc[fr, "elegible_test_4"] = (n_pos_dada >= 1)[fr] & (cuenta(rel & (p == TEST)) >= 1)[fr]

    vistas, mascaras = {}, {}
    w, es_b = H.pesos_historial(senal, media, c_s, c_h)
    for vista, parts in VISTAS.items():
        dentro = np.isin(p, parts)
        for bolsa in H.BOLSAS:
            m = dentro & (es_b if bolsa == "B" else ~es_b)
            vistas[(vista, bolsa)] = H.csr(u[m], i[m], w[m], n_usuarios, n_items)
        mascaras[vista] = S.mascara_valorados(u, i, np.where(dentro, 0, 9), n_usuarios, n_items)
    meta.update({"usuarios_frios_en_experimento": int(usuarios.frio.sum()),
                 "usuarios_frios_elegibles_test": int(usuarios.elegible_test[fr].sum()),
                 "usuarios_frios_elegibles_test_4": int(usuarios.elegible_test_4[fr].sum()),
                 "usuarios_normales_elegibles_val": int(usuarios.elegible_val[~fr].sum()),
                 "mu_global_train": mu})
    return {"split": r, "senal": senal, "usuarios": usuarios, "logq": logq, "indptr": ip, "indices": ix,
            "vistas": vistas, "mascaras": mascaras, "ejemplos": H.ejemplos_train(senal), "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    salida = PROCESADOS / f"{a.muestra}_frio"
    salida.mkdir(parents=True, exist_ok=True)
    if all((salida / f).exists() for f in SALIDAS) and not a.forzar:
        print(f"{salida}: ya existe; usa --forzar para rehacerlo")
        return
    t0 = time.time()

    marcar(TAREA, 0, 5, "leyendo parquet")
    r = pd.read_parquet(PROCESADOS / cfg["muestras"][a.muestra])
    movies = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId"])
    n_entrada = len(r)

    marcar(TAREA, 1, 5, "índices, C11 y elección de fríos")
    r, mapeo_p, mapeo_u = SP.asignar_indices(movies, r)
    if cfg["split"]["quitar_antes_estreno"]:
        r = r[~r.antes_estreno].reset_index(drop=True)
    n_usuarios, n_items = len(mapeo_u), len(mapeo_p)
    frios = elegir_frios(mapeo_u.userId.to_numpy(), cfg["frio"]["frac_usuarios"], cfg["semilla"])

    marcar(TAREA, 2, 5, "partición, señal y vistas")
    res = construir_frio(r, frios, cfg, n_usuarios, n_items)

    marcar(TAREA, 3, 5, "guardando")
    res["split"].to_parquet(salida / "split.parquet", index=False)
    res["senal"].to_parquet(salida / "senal.parquet", index=False)
    res["usuarios"].to_parquet(salida / "usuarios.parquet", index=False)
    res["ejemplos"].to_parquet(salida / "ejemplos_train.parquet", index=False)
    mapeo_p.to_parquet(salida / "mapeo_peliculas.parquet", index=False)
    mapeo_u.to_parquet(salida / "mapeo_usuarios.parquet", index=False)
    np.save(salida / "logq.npy", res["logq"])
    np.save(salida / "valorados_train_indptr.npy", res["indptr"])
    np.save(salida / "valorados_train_indices.npy", res["indices"])
    for vista, (ip, ix) in res["mascaras"].items():
        if vista != "train":
            np.save(salida / f"valorados_{vista}_indptr.npy", ip)
            np.save(salida / f"valorados_{vista}_indices.npy", ix)
    for (v, b), (indptr, indices, pesos) in res["vistas"].items():
        np.save(salida / f"hist_{v}_{b}_indptr.npy", indptr)
        np.save(salida / f"hist_{v}_{b}_indices.npy", indices)
        np.save(salida / f"hist_{v}_{b}_pesos.npy", pesos)

    marcar(TAREA, 4, 5, "meta")
    meta = {"muestra": a.muestra, "filas_entrada": int(n_entrada), "n_usuarios": n_usuarios, "n_peliculas": n_items,
            "n_frios_elegidos": int(frios.sum()), "min_ratings_frio": cfg["frio"].get("min_ratings", MIN_RATINGS_FRIO), **res["meta"],
            "parametros": {"frio": cfg["frio"], "semilla": cfg["semilla"]},
            "segundos": round(time.time() - t0, 2)}
    (salida / "meta_frio.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    marcar(TAREA, 5, 5, "hecho")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
