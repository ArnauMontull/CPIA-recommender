"""Evaluación sin etiquetas con la watchlist de Letterboxd: ¿qué fracción del top-k del modelo está en lo que el usuario ya
marcó como «quiero ver»? El modelo nunca ve la watchlist. Se compara con popularidad (lo más visto en train) y con el azar.

La watchlist elegible son las películas emparejadas con el catálogo y que el usuario NO ha visto (las vistas se excluyen de
las recomendaciones y no pueden acertar). Es una métrica débil: la watchlist son intereses declarados, no gustos medidos, y
tiende a llenarse de clásicos populares, de ahí la comparación con popularidad.

Uso: python -m src.evaluar_watchlist --alias arnau_api --export <zip> --checkpoint modelos/.../mejor.pt [--sin-api]
Escribe data/external/letterboxd/<alias>/watchlist_<etiqueta>.md (fuera de git: son datos personales).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .datos import cargar_features
from .demo_letterboxd import recomendar_usuario
from .letterboxd.__main__ import crear_buscador
from .letterboxd.emparejar import Catalogo, emparejar
from .letterboxd.export import _leer_csv
from .modelo import DosTorres
from .preprocesamiento.comun import DATA, PROCESADOS, RAIZ

KS = (10, 20, 50, 200)


def metricas_watchlist(scores: np.ndarray, excluidos: np.ndarray, watchlist: np.ndarray, ks=KS) -> dict:
    """Con `scores` [N] (más alto = mejor) y sin los `excluidos`: precisión@k, recall@k y lift sobre el azar.
    `watchlist` = índices elegibles (ya sin lo excluido). Azar: la fracción de candidatos que está en la watchlist."""
    s = np.asarray(scores, dtype="float64").copy()
    s[np.asarray(excluidos, dtype=np.int64)] = -np.inf
    n_cand = int(np.isfinite(s).sum())
    orden = np.lexsort((np.arange(len(s)), -s))[:max(ks)]
    en_w = np.isin(orden, np.asarray(watchlist, dtype=np.int64))
    azar = len(watchlist) / max(n_cand, 1)
    out = {"n_watchlist": int(len(watchlist)), "candidatos": n_cand, "azar": azar}
    for k in ks:
        hits = int(en_w[:k].sum())
        out[f"hits@{k}"] = hits
        out[f"precision@{k}"] = hits / k
        out[f"recall@{k}"] = hits / max(len(watchlist), 1)
        out[f"lift@{k}"] = (hits / k) / azar if azar > 0 else float("nan")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alias", required=True)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--muestra", default="completo")
    ap.add_argument("--sin-api", action="store_true")
    ap.add_argument("--etiqueta", default="s42")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    base = DATA / cfg["letterboxd"]["carpeta"]
    carpeta = base / a.alias
    movies = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId", "titulo", "anio", "tmdbId", "n_ratings"])
    mapeo = pd.read_parquet(PROCESADOS / "dev" / "mapeo_peliculas.parquet")
    catalogo = Catalogo(movies, mapeo)
    buscar = None if a.sin_api else crear_buscador(base / "cache_tmdb.jsonl")

    w = _leer_csv(a.export, "watchlist.csv")
    if w is None or not len(w):
        raise SystemExit(f"{a.export}: no trae watchlist.csv")
    w = w[["Name", "Year"]].copy()
    w["Year"] = pd.to_numeric(w["Year"], errors="coerce").astype("Int16")
    emp = emparejar(w, catalogo, buscar)
    emparejada = emp[emp.metodo != "sin_emparejar"].item_idx.to_numpy().astype("int64")

    ld = lambda n: np.load(carpeta / n)
    excl = ld("valorados_indices.npy")
    elegible = np.setdiff1d(np.unique(emparejada), excl)

    feats = cargar_features(PROCESADOS / a.muestra)
    modelo = DosTorres.desde_config(feats, cfg).eval()
    modelo.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["modelo"])
    with torch.no_grad():
        V = modelo.vectores_items()
    N = V.shape[0]
    A = (ld("hist_A_indices.npy"), ld("hist_A_pesos.npy"))
    B = (ld("hist_B_indices.npy"), ld("hist_B_pesos.npy"))
    top, sc = recomendar_usuario(modelo, V, A, B, excl, N)       # el ranking completo, sin lo visto
    score_modelo = np.full(N, -np.inf)
    score_modelo[top] = sc
    pop = np.load(PROCESADOS / a.muestra / "logq.npy").astype("float64")   # log P(i): monótono con la popularidad en train

    res = {"Two-Tower": metricas_watchlist(score_modelo, excl, elegible),
           "Popularidad": metricas_watchlist(pop, excl, elegible)}
    r0 = res["Two-Tower"]
    lineas = [f"# Watchlist de `{a.alias}` frente al top-k ({a.etiqueta})", "",
              f"Watchlist: {len(w)} películas; emparejadas con el catálogo: {len(np.unique(emparejada))}; elegibles (sin las ya vistas): "
              f"{r0['n_watchlist']} de {r0['candidatos']} candidatos. **Azar: {100 * r0['azar']:.2f} % de aciertos.**", "",
              "| k | Two-Tower: aciertos | precisión | recall | lift | Popularidad: aciertos | precisión | recall | lift |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in KS:
        m, p = res["Two-Tower"], res["Popularidad"]
        lineas.append(f"| {k} | {m[f'hits@{k}']} | {m[f'precision@{k}']:.3f} | {m[f'recall@{k}']:.3f} | {m[f'lift@{k}']:.1f}× "
                      f"| {p[f'hits@{k}']} | {p[f'precision@{k}']:.3f} | {p[f'recall@{k}']:.3f} | {p[f'lift@{k}']:.1f}× |")
    salida = carpeta / f"watchlist_{a.etiqueta}.md"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    if buscar is not None:
        print(f"(llamadas nuevas a la API: {buscar.llamadas})")


if __name__ == "__main__":
    main()
