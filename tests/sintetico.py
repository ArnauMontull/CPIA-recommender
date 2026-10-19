"""Mini-dataset sintético con el mismo formato que data/processed/<muestra>/ (sin datos reales)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocesamiento import historial, senal
from src.preprocesamiento.comun import RAIZ

CFG = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


def crear(raiz: Path, n_usuarios: int = 40, n_items: int = 30, por_usuario: int = 20, semilla: int = 0) -> Path:
    """Escribe raiz/processed/sint/ y raiz/processed/emb_texto.npy; devuelve la carpeta de la muestra.
    Cada usuario valora `por_usuario` películas: 70 % train, 15 % val, 15 % test (por orden temporal)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for u in range(n_usuarios):
        items = rng.choice(n_items, por_usuario, replace=False)
        for k, i in enumerate(items):
            p = 0 if k < int(0.7 * por_usuario) else (1 if k < int(0.85 * por_usuario) else 2)
            filas.append((u, i, float(rng.choice([1, 2, 3, 3.5, 4, 4.5, 5])), 1_000_000 + k * 200_000, p, k < 3))
    df = pd.DataFrame(filas, columns=["user_idx", "item_idx", "rating", "timestamp", "particion", "primer_dia"])
    df = df.astype({"user_idx": "int32", "item_idx": "int32", "particion": "int8"})
    r = senal.construir(df, CFG["senal"], n_usuarios, n_items)
    s, media = r["senal"], r["usuarios"].media_u.to_numpy().astype("float64")

    c = raiz / "processed" / "sint"
    c.mkdir(parents=True)
    s.to_parquet(c / "senal.parquet", index=False)
    r["usuarios"].to_parquet(c / "usuarios.parquet", index=False)
    np.save(c / "logq.npy", r["logq"])
    np.save(c / "valorados_train_indptr.npy", r["indptr"])
    np.save(c / "valorados_train_indices.npy", r["indices"])
    ip, ix = senal.mascara_valorados(s.user_idx.to_numpy(), s.item_idx.to_numpy(),
                                     np.where(s.particion.to_numpy() <= 1, 0, 2), n_usuarios, n_items)
    np.save(c / "valorados_train_val_indptr.npy", ip)
    np.save(c / "valorados_train_val_indices.npy", ix)
    for (v, b), (indptr, indices, pesos) in historial.construir_vistas(s, media, CFG["senal"], CFG["historial"], n_usuarios, n_items).items():
        np.save(c / f"hist_{v}_{b}_indptr.npy", indptr)
        np.save(c / f"hist_{v}_{b}_indices.npy", indices)
        np.save(c / f"hist_{v}_{b}_pesos.npy", pesos)
    historial.ejemplos_train(s).to_parquet(c / "ejemplos_train.parquet", index=False)
    (c / "meta_historial.json").write_text(json.dumps({"n_items": n_items}), encoding="utf-8")

    feats = pd.DataFrame({
        "item_idx": np.arange(n_items, dtype="int32"), "movieId": np.arange(n_items, dtype="int32"),
        "generos": [rng.choice(16, rng.integers(1, 4), replace=False).tolist() for _ in range(n_items)],
        "directores": [[int(rng.integers(1, 6))] for _ in range(n_items)],
        "actores": [rng.integers(1, 9, rng.integers(0, 6)).tolist() for _ in range(n_items)],
        "anio_z": rng.normal(size=n_items).astype("float32"), "duracion_z": rng.normal(size=n_items).astype("float32"),
        **{f: np.zeros(n_items, dtype=bool) for f in ["sin_generos", "sin_director", "sin_reparto", "sin_anio", "sin_duracion", "sin_texto"]}})
    feats.to_parquet(c / "peliculas_features.parquet", index=False)
    np.save(c.parent / "emb_texto.npy", rng.normal(size=(n_items, 384)).astype("float16"))
    return c
