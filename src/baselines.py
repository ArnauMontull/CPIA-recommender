"""Baselines (T5) con las mismas métricas, usuarios y máscara que el modelo. De momento: popularidad."""
from __future__ import annotations

import numpy as np

from .datos import Datos
from .evaluacion.metricas import evaluar_por_lotes


def evaluar_popularidad(datos: Datos, particion: int, ks, usuarios_max: int | None = None, semilla: int = 0,
                        tam_lote: int = 1024) -> dict[str, float]:
    """Recomienda a todos lo más visto (positivos de train), sin lo que el usuario ya valoró (C14).
    Mismo conjunto de usuarios que `entrenar.evaluar_particion` para la misma `semilla` y `usuarios_max`."""
    vista = "train" if particion == 1 else "train_val"
    rel, elig = datos.relevantes(particion)
    usuarios = np.unique(np.concatenate(list(elig.values())))
    if usuarios_max and len(usuarios) > usuarios_max:
        usuarios = np.sort(np.random.default_rng(semilla).choice(usuarios, usuarios_max, replace=False))
        elig = {k: np.intersect1d(v, usuarios) for k, v in elig.items()}
    pop = datos.pop_train.astype("float32")
    score = lambda filas: np.tile(pop, (len(filas), 1))
    return evaluar_por_lotes(score, usuarios, rel, elig, ks, datos.pop_train, datos.n_items, *datos.valorados[vista],
                             tam_lote=tam_lote)
