"""media_u encogida (G4) y bolsas A/B del usuario nuevo (F4/F5 con c9 = 1), reutilizando senal.py e historial.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocesamiento.historial import csr, pesos_historial
from ..preprocesamiento.senal import etiquetar


def media_global_train(carpeta) -> float:
    """mu = media de los ratings de train (particion 0) en <carpeta>/senal.parquet."""
    s = pd.read_parquet(carpeta / "senal.parquet", columns=["rating", "particion"])
    return float(s.rating[s.particion == 0].astype("float64").mean())


def media_encogida(ratings: np.ndarray, mu: float, cfg: dict) -> tuple[float, float]:
    """(media cruda, media_u) con G4: si n < min_notas_sin_encoger, (suma + lambda*mu) / (n + lambda); si no, la cruda."""
    r = np.asarray(ratings, dtype="float64")
    n = len(r)
    if n == 0:
        return float("nan"), float(mu)
    cruda = float(r.mean())
    if n >= cfg["min_notas_sin_encoger"]:
        return cruda, cruda
    lam = cfg["lambda_encogimiento"]
    return cruda, float((r.sum() + lam * mu) / (n + lam))


def bolsas(item_idx: np.ndarray, rating: np.ndarray, media_u: float, n_items: int, cfg_senal: dict, cfg_hist: dict) -> dict:
    """Bolsas A (positivos + neutras) y B (notas bajas) como CSR de un solo usuario, pesos de F4 con c9 = 1.
    Devuelve {"A": (indices int32, pesos float32), "B": (...)}, items ordenados."""
    n = len(item_idx)
    df = pd.DataFrame({"user_idx": np.zeros(n, dtype="int32"), "rating": np.asarray(rating, dtype="float32"),
                       "primer_dia": np.zeros(n, dtype=bool)})
    media = np.array([media_u], dtype="float64")
    df = etiquetar(df, media, cfg_senal)
    w, es_b = pesos_historial(df, media, cfg_senal, cfg_hist)
    out = {}
    for nombre, m in (("A", ~es_b), ("B", es_b)):
        _, ix, ps = csr(np.zeros(int(m.sum()), dtype="int32"), np.asarray(item_idx)[m], w[m], 1, n_items)
        out[nombre] = (ix, ps)
    return out
