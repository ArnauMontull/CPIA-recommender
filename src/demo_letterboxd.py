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


def mmr(V: torch.Tensor, candidatos: np.ndarray, scores: np.ndarray, k: int = 20, lam: float = 0.7) -> np.ndarray:
    """Reordenación MMR (Carbonell y Goldstein): en cada paso, el candidato con mayor lam * score - (1 - lam) * (máxima
    similitud coseno con lo ya elegido). V tiene filas de norma 1, así que la similitud es el producto escalar.
    La relevancia se normaliza a [0, 1] entre los candidatos: los scores del modelo ocupan un rango estrecho (~0,1) y
    las similitudes uno ancho, así que sin normalizar `lam` no equilibraría nada (la diversidad aplastaría al score).
    lam = 1: el ranking original; lam menor: más diversidad. Devuelve k índices de película."""
    C = V[torch.as_tensor(candidatos, dtype=torch.int64)].numpy()
    rel = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-12)
    sim = C @ C.T
    elegidos = [0]                                           # el primero es siempre el mejor
    while len(elegidos) < min(k, len(candidatos)):
        resto = np.setdiff1d(np.arange(len(candidatos)), elegidos)
        valor = lam * rel[resto] - (1 - lam) * sim[np.ix_(resto, elegidos)].max(axis=1)
        elegidos.append(int(resto[np.argmax(valor)]))
    return np.asarray(candidatos)[elegidos]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alias", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--muestra", default="completo", help="carpeta de data/processed con las features (no cambia entre dev y completo)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--mmr", type=float, default=None, help="lambda del MMR (p. ej. 0.7); por defecto, el top-k sin reordenar")
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
    top, score = recomendar_usuario(modelo, V, A, B, ld("valorados_indices.npy"), 200 if a.mmr is not None else a.k)
    if a.mmr is not None:                                    # MMR sobre los 200 mejores, como en el motor en C (H10)
        elegidos = mmr(V, top, score, a.k, a.mmr)
        score = score[[int(np.where(top == i)[0][0]) for i in elegidos]]
        top = elegidos

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
