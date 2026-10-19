"""Demo cualitativa (plan_entrenamiento.md, tarea 4): top-N de un perfil de Letterboxd ya emparejado con el catálogo.

Entrada: data/external/letterboxd/<alias>/ (salida de `python -m src.letterboxd`) y un checkpoint. Escribe
data/external/letterboxd/<alias>/recomendaciones_<etiqueta>.md, fuera de git (son datos personales).

Uso: python -m src.demo_letterboxd --alias arnau --checkpoint modelos/completo_tau0.05_s42/mejor.pt [--k 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .datos import cargar_features
from .modelo import Bolsa, DosTorres
from .preprocesamiento.comun import DATA, PROCESADOS, RAIZ


def recomendar_usuario(modelo: DosTorres, V: torch.Tensor, A: tuple, B: tuple, excluidos: np.ndarray, k: int = 20):
    """(items [k], scores [k]) para un usuario nuevo con bolsas A y B = (indices, pesos). Sin lo ya valorado (C14)."""
    bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
    with torch.no_grad():
        u = modelo.vectores_usuarios(V, torch.tensor([0]), bolsa(*A), bolsa(*B))[0]
        s = (V @ u).numpy().copy()
    s[np.asarray(excluidos, dtype=np.int64)] = -np.inf
    top = np.lexsort((np.arange(len(s)), -s))[:k]
    return top, s[top]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alias", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--muestra", default="completo", help="carpeta de data/processed con las features (no cambia entre dev y completo)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--etiqueta", default="")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    carpeta = DATA / cfg["letterboxd"]["carpeta"] / a.alias
    feats = cargar_features(PROCESADOS / a.muestra)
    modelo = DosTorres.desde_config(feats, cfg).eval()
    modelo.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["modelo"])
    with torch.no_grad():
        V = modelo.vectores_items()

    ld = lambda n: np.load(carpeta / n)
    A = (ld("hist_A_indices.npy"), ld("hist_A_pesos.npy"))
    B = (ld("hist_B_indices.npy"), ld("hist_B_pesos.npy"))
    top, score = recomendar_usuario(modelo, V, A, B, ld("valorados_indices.npy"), a.k)

    movies = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId", "titulo", "anio"])
    mapeo = pd.read_parquet(PROCESADOS / "dev" / "mapeo_peliculas.parquet").set_index("item_idx")
    ficha = movies.set_index("movieId")
    lineas = [f"# Recomendaciones para `{a.alias}`", "",
              f"Checkpoint `{a.checkpoint.name}`. Bolsa A: {len(A[0])} películas, bolsa B (notas bajas): {len(B[0])}, valoradas en el catálogo: {len(ld('valorados_indices.npy'))}.", "",
              "| # | Película | Año | Score |", "|---:|---|---:|---:|"]
    for i, (it, sc) in enumerate(zip(top, score), 1):
        f = ficha.loc[int(mapeo.loc[int(it), "movieId"])]
        anio = "" if pd.isna(f.anio) else int(f.anio)
        lineas.append(f"| {i} | {f.titulo} | {anio} | {sc:.3f} |")
    salida = carpeta / f"recomendaciones{a.etiqueta}.md"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    print("escrito:", salida)


if __name__ == "__main__":
    main()
