"""Recomendación para dos personas (o un grupo): películas que no han visto ninguno y que encajen con los dos.

Cada perfil de Letterboxd (salida de `python -m src.letterboxd`) da un vector de usuario u_i; el score de una película para esa
persona es u_i · v. Para juntar dos personas hay tres reglas, todas sobre el mismo espacio de vectores:
  - interseccion: lo que está a la vez en el top-k de los dos (sencilla, pero con k pequeño suele salir casi vacía);
  - minimo: min(score_1, score_2), «nadie sufre» (least misery);
  - media: (score_1 + score_2) / 2.
Siempre se excluye la unión de lo visto por los dos. Escribe data/external/letterboxd/pareja_<a>_<b>.md (fuera de git).

Uso: python -m src.demo_pareja --alias arnau_api uriel_api --checkpoint modelos/.../mejor.pt [--k 20]
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
from .modelo import DosTorres
from .preprocesamiento.comun import DATA, PROCESADOS, RAIZ


def scores_pareja(modelo, V, perfiles: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """([P, N] scores de cada persona sobre todo el catálogo, unión de lo visto, N). Lo visto por cualquiera queda en -inf."""
    N = V.shape[0]
    excl = np.unique(np.concatenate([p["valorados"] for p in perfiles]))
    filas = []
    for p in perfiles:
        top, sc = recomendar_usuario(modelo, V, p["A"], p["B"], p["valorados"], N)
        s = np.full(N, -np.inf)
        s[top] = sc
        s[excl] = -np.inf
        filas.append(s)
    return np.vstack(filas), excl, N


def interseccion(S: np.ndarray, ks=(20, 50, 200)) -> dict[int, np.ndarray]:
    """Películas que están a la vez en el top-k de todas las personas, para cada k."""
    tops = [np.lexsort((np.arange(S.shape[1]), -s)) for s in S]
    return {k: np.array(sorted(set.intersection(*[set(t[:k].tolist()) for t in tops]))) for k in ks}


def combinar(S: np.ndarray, regla: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k de la regla `minimo` o `media` y sus scores combinados."""
    c = S.min(axis=0) if regla == "minimo" else S.mean(axis=0)
    top = np.lexsort((np.arange(len(c)), -c))[:k]
    return top, c[top]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alias", nargs=2, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--muestra", default="completo")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    base = DATA / cfg["letterboxd"]["carpeta"]
    feats = cargar_features(PROCESADOS / a.muestra)
    modelo = DosTorres.desde_config(feats, cfg).eval()
    modelo.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["modelo"])
    with torch.no_grad():
        V = modelo.vectores_items()
    perfiles = []
    for alias in a.alias:
        ld = lambda n, al=alias: np.load(base / al / n)
        perfiles.append({"A": (ld("hist_A_indices.npy"), ld("hist_A_pesos.npy")), "B": (ld("hist_B_indices.npy"), ld("hist_B_pesos.npy")),
                         "valorados": ld("valorados_indices.npy")})
    S, excl, N = scores_pareja(modelo, V, perfiles)

    movies = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId", "titulo", "anio"]).set_index("movieId")
    mapeo = pd.read_parquet(PROCESADOS / "dev" / "mapeo_peliculas.parquet").set_index("item_idx")
    nombre = lambda i: (lambda f: f"{f.titulo} ({'' if pd.isna(f.anio) else int(f.anio)})")(movies.loc[int(mapeo.loc[int(i), 'movieId'])])

    lineas = [f"# Pareja `{a.alias[0]}` + `{a.alias[1]}`", "",
              f"Excluido lo visto por cualquiera de los dos: {len(excl)} películas de {N}.", "",
              "## Intersección de los top-k individuales", "", "| k | Coinciden | % de k |", "|---:|---:|---:|"]
    inter = interseccion(S)
    for k, v in inter.items():
        lineas.append(f"| {k} | {len(v)} | {100 * len(v) / k:.0f} % |")
    for regla in ("minimo", "media"):
        top, c = combinar(S, regla, a.k)
        lineas += ["", f"## Top-{a.k} combinando con la regla `{regla}`", "",
                   "| # | Película | Score persona 1 | Score persona 2 |", "|---:|---|---:|---:|"]
        for r, i in enumerate(top, 1):
            lineas.append(f"| {r} | {nombre(i)} | {S[0, i]:.3f} | {S[1, i]:.3f} |")
    salida = base / f"pareja_{a.alias[0]}_{a.alias[1]}.md"
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
