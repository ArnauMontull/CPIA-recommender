"""Baselines (T5) con las mismas métricas, usuarios y máscara que el modelo: popularidad y EASE (Steck 2019)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .datos import Datos
from .evaluacion.metricas import evaluar_por_lotes


def _usuarios(datos: Datos, particion: int, usuarios_max: int | None, semilla: int):
    """Mismos usuarios que `entrenar.evaluar_particion` para la misma `semilla` y `usuarios_max`."""
    rel, elig = datos.relevantes(particion)
    usuarios = np.unique(np.concatenate(list(elig.values())))
    if usuarios_max and len(usuarios) > usuarios_max:
        usuarios = np.sort(np.random.default_rng(semilla).choice(usuarios, usuarios_max, replace=False))
        elig = {k: np.intersect1d(v, usuarios) for k, v in elig.items()}
    return usuarios, rel, elig


def _evaluar(datos: Datos, particion: int, score, ks, usuarios_max, semilla, tam_lote) -> dict[str, float]:
    vista = "train" if particion == 1 else "train_val"
    usuarios, rel, elig = _usuarios(datos, particion, usuarios_max, semilla)
    return evaluar_por_lotes(lambda filas: score(filas, vista), usuarios, rel, elig, ks, datos.pop_train, datos.n_items,
                             *datos.valorados[vista], tam_lote=tam_lote)


def evaluar_popularidad(datos: Datos, particion: int, ks, usuarios_max: int | None = None, semilla: int = 0,
                        tam_lote: int = 1024) -> dict[str, float]:
    """Recomienda a todos lo más visto (positivos de train), sin lo que el usuario ya valoró (C14)."""
    pop = datos.pop_train.astype("float32")
    return _evaluar(datos, particion, lambda filas, vista: np.tile(pop, (len(filas), 1)), ks, usuarios_max, semilla, tam_lote)


class EASE:
    """B = -P / diag(P) con P = (X'X + lam I)^-1 y diag(B) = 0, sobre las `max_items` películas con más positivos
    de train (la inversión es densa: con las 23 335 hacen falta ~9 GB). Fuera de ellas, score -inf.
    X = historial A de la vista `train` en binario (positivos + neutras)."""

    def __init__(self, datos: Datos, lam: float = 500.0, max_items: int = 10000):
        self.datos, self.lam = datos, lam
        n = datos.n_items
        self.items = np.sort(np.argsort(-datos.pop_train, kind="stable")[:min(max_items, n)])
        self.X_cols = self._binaria("train")
        X = self.X_cols
        G = (X.T @ X).toarray().astype("float64")
        G[np.diag_indices_from(G)] += lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype("float32")

    def _binaria(self, vista: str, filas: np.ndarray | None = None) -> sp.csr_matrix:
        ip, ix, _ = self.datos.hist[(vista, "A")]
        X = sp.csr_matrix((np.ones(len(ix), "float32"), ix, ip), shape=(self.datos.n_usuarios, self.datos.n_items))
        if filas is not None:
            X = X[filas]
        return X[:, self.items]

    def scores(self, filas: np.ndarray, vista: str) -> np.ndarray:
        out = np.full((len(filas), self.datos.n_items), -np.inf, dtype="float32")
        out[:, self.items] = self._binaria(vista, filas) @ self.B
        return out


def evaluar_ease(datos: Datos, particion: int, ks, usuarios_max: int | None = None, semilla: int = 0,
                 tam_lote: int = 1024, modelo: EASE | None = None, **kw) -> dict[str, float]:
    """EASE ajustado con train; en test el historial es train + val (F8), como en el modelo."""
    modelo = modelo or EASE(datos, **kw)
    return _evaluar(datos, particion, modelo.scores, ks, usuarios_max, semilla, tam_lote)
