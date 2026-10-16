"""Métricas de evaluación del recomendador (D6, C7, C8, C14). numpy puro y vectorizado.

Los relevantes y la máscara de train van en CSR `(indptr, indices)` indexado por `user_idx` global
(formato de la Etapa 2). Las matrices de scores se evalúan por lotes: nunca se materializa U x N.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_BASE = np.int64(1) << 31  # clave = fila * _BASE + item (cabe en int64)


def csr_desde_pares(user_idx, item_idx, n_usuarios: int) -> tuple[np.ndarray, np.ndarray]:
    """(user, item) -> (indptr int64 [n_usuarios+1], indices int32), ordenado y sin duplicados."""
    u = np.asarray(user_idx, dtype=np.int64)
    i = np.asarray(item_idx, dtype=np.int64)
    claves = np.unique(u * _BASE + i)
    u, i = claves // _BASE, claves % _BASE
    indptr = np.zeros(n_usuarios + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n_usuarios), out=indptr[1:])
    return indptr, i.astype(np.int32)


def _filas_por_defecto(n: int, filas) -> np.ndarray:
    return np.arange(n, dtype=np.int64) if filas is None else np.asarray(filas, dtype=np.int64)


def _extraer(indptr, indices, filas) -> tuple[np.ndarray, np.ndarray]:
    """Para las filas globales `filas`: (posición en el lote, columna) de cada elemento del CSR."""
    indptr = np.asarray(indptr, dtype=np.int64)
    ini = indptr[filas]
    n = indptr[filas + 1] - ini
    pos = np.repeat(np.arange(len(filas)), n)
    desplaz = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return pos, np.asarray(indices)[np.repeat(ini, n) + desplaz]


def top_k(scores, k: int, excluir_indptr, excluir_indices, filas=None) -> np.ndarray:
    """Índices top-k por fila (orden descendente de score) tras poner -inf en lo excluido (C14).

    `filas[j]` es el user_idx global de la fila j de `scores` (por defecto 0..B-1).
    Empates: dentro del trozo gana el menor índice; en el corte del k-ésimo es arbitrario.
    """
    s = np.array(scores, dtype=np.float64 if np.asarray(scores).dtype == np.float64 else np.float32)
    filas = _filas_por_defecto(len(s), filas)
    pos, cols = _extraer(excluir_indptr, excluir_indices, filas)
    s[pos, cols] = -np.inf
    k = min(k, s.shape[1])
    if k < s.shape[1]:
        trozo = np.sort(np.argpartition(-s, k - 1, axis=1)[:, :k], axis=1)
    else:
        trozo = np.tile(np.arange(k), (len(s), 1))
    orden = np.argsort(-np.take_along_axis(s, trozo, axis=1), axis=1, kind="stable")
    return np.take_along_axis(trozo, orden, axis=1)


def _aciertos(topk, rel_indptr, rel_indices, filas) -> tuple[np.ndarray, np.ndarray]:
    """(matriz bool de aciertos [B, k], |R_u| por fila)."""
    pos, cols = _extraer(rel_indptr, rel_indices, filas)
    claves = np.sort(pos.astype(np.int64) * _BASE + cols)
    q = np.arange(len(topk), dtype=np.int64)[:, None] * _BASE + topk
    if len(claves) == 0:
        hit = np.zeros(q.shape, dtype=bool)
    else:
        hit = claves[np.minimum(np.searchsorted(claves, q), len(claves) - 1)] == q
    rel_indptr = np.asarray(rel_indptr, dtype=np.int64)
    return hit, rel_indptr[filas + 1] - rel_indptr[filas]


def recall_at_k(topk, rel_indptr, rel_indices, k: int, filas=None) -> np.ndarray:
    """|top_k ∩ R_u| / |R_u| por usuario (NaN si |R_u| = 0)."""
    filas = _filas_por_defecto(len(topk), filas)
    hit, n_rel = _aciertos(np.asarray(topk)[:, :k], rel_indptr, rel_indices, filas)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_rel > 0, hit.sum(1) / n_rel, np.nan)


def techo_recall(rel_indptr, rel_indices, k: int, filas) -> float:
    """Recall máximo alcanzable: media de min(k, |R_u|) / |R_u| sobre `filas` con |R_u| > 0."""
    filas = np.asarray(filas, dtype=np.int64)
    rel_indptr = np.asarray(rel_indptr, dtype=np.int64)
    n_rel = (rel_indptr[filas + 1] - rel_indptr[filas]).astype(np.float64)
    n_rel = n_rel[n_rel > 0]
    return float(np.mean(np.minimum(k, n_rel) / n_rel)) if len(n_rel) else float("nan")


def ndcg_at_k(topk, rel_indptr, rel_indices, k: int, filas=None) -> np.ndarray:
    """NDCG@k por usuario: ganancia binaria, descuento 1/log2(pos+2), IDCG con min(k, |R_u|) (NaN si |R_u| = 0)."""
    filas = _filas_por_defecto(len(topk), filas)
    topk = np.asarray(topk)[:, :k]
    hit, n_rel = _aciertos(topk, rel_indptr, rel_indices, filas)
    desc = 1.0 / np.log2(np.arange(topk.shape[1]) + 2)
    dcg = (hit * desc).sum(1)
    idcg_tabla = np.cumsum(1.0 / np.log2(np.arange(max(k, 1)) + 2))
    idcg = idcg_tabla[np.clip(np.minimum(k, n_rel), 1, None) - 1]
    return np.where(n_rel > 0, dcg / idcg, np.nan)


def cobertura(topk, N: int) -> float:
    """Fracción del catálogo que aparece en algún top-k."""
    return float(len(np.unique(topk))) / N


def _info_item(pop_train) -> np.ndarray:
    """-log2 p(i), con p(i) = (pop+1) / sum(pop+1) (suavizado +1)."""
    pop = np.asarray(pop_train, dtype=np.float64) + 1.0
    return -np.log2(pop / pop.sum())


def novedad(topk, pop_train) -> float:
    """Media de -log2 p(i) sobre los items recomendados (p = popularidad de train suavizada)."""
    return float(_info_item(pop_train)[np.asarray(topk)].mean())


class _Acumulador:
    """Suma métricas lote a lote; `resultado()` devuelve el dict plano."""

    def __init__(self, relevantes, elegibles, ks, pop_train, N):
        self.rel, self.ks, self.N = relevantes, sorted(ks), N
        self.elig = {d: np.unique(np.asarray(e, dtype=np.int64)) for d, e in elegibles.items()}
        self.nov = _info_item(pop_train)
        self.suma: dict[str, float] = {}
        self.cuenta: dict[str, int] = {}
        self.vistos = {k: np.zeros(N, dtype=bool) for k in self.ks}

    def _sumar(self, clave, valores):
        valores = valores[~np.isnan(valores)]  # usuario elegible sin relevantes: no cuenta
        self.suma[clave] = self.suma.get(clave, 0.0) + float(valores.sum())
        self.cuenta[clave] = self.cuenta.get(clave, 0) + len(valores)

    def anadir(self, topk, filas) -> None:
        filas = np.asarray(filas, dtype=np.int64)
        for k in self.ks:
            tk = topk[:, :k]
            self.vistos[k][tk.ravel()] = True
            self._sumar(f"novedad@{k}", self.nov[tk].mean(1))
            for d, (ip, ix) in self.rel.items():
                mask = np.isin(filas, self.elig[d])
                if not mask.any():
                    continue
                f = filas[mask]
                self._sumar(f"recall@{k}/{d}", recall_at_k(tk[mask], ip, ix, k, f))
                self._sumar(f"ndcg@{k}/{d}", ndcg_at_k(tk[mask], ip, ix, k, f))

    def resultado(self) -> dict[str, float]:
        out = {c: (self.suma[c] / self.cuenta[c] if self.cuenta[c] else float("nan")) for c in self.suma}
        for k in self.ks:
            out[f"cobertura@{k}"] = float(self.vistos[k].sum()) / self.N
            for d, (ip, ix) in self.rel.items():
                out[f"techo_recall@{k}/{d}"] = techo_recall(ip, ix, k, self.elig[d])
                out.setdefault(f"recall@{k}/{d}", float("nan"))
                out.setdefault(f"ndcg@{k}/{d}", float("nan"))
        return out


def evaluar(topk, relevantes_por_definicion: dict, usuarios_elegibles: dict, ks, pop_train, N: int,
            filas=None) -> dict[str, float]:
    """Dict plano {"recall@20/train", "ndcg@20/r4", "techo_recall@20/train", "cobertura@20", "novedad@20"}.

    `topk` [B, >=max(ks)] con `filas` = user_idx de cada fila. Recall y NDCG se promedian solo sobre
    los usuarios elegibles de cada definición (C8). Cobertura y novedad, sobre todas las filas.
    """
    topk = np.asarray(topk)
    acum = _Acumulador(relevantes_por_definicion, usuarios_elegibles, ks, pop_train, N)
    acum.anadir(topk, _filas_por_defecto(len(topk), filas))
    return acum.resultado()


def evaluar_por_lotes(score_fn: Callable[[np.ndarray], np.ndarray], usuarios, relevantes_por_definicion: dict,
                      usuarios_elegibles: dict, ks, pop_train, N: int, excluir_indptr, excluir_indices,
                      tam_lote: int = 1024) -> dict[str, float]:
    """Como `evaluar`, pero pidiendo los scores por lotes: `score_fn(filas) -> [len(filas), N]`.

    `usuarios` son user_idx globales. Mismo resultado que evaluar sobre todos de una vez.
    """
    usuarios = np.asarray(usuarios, dtype=np.int64)
    acum = _Acumulador(relevantes_por_definicion, usuarios_elegibles, ks, pop_train, N)
    kmax = max(ks)
    for ini in range(0, len(usuarios), tam_lote):
        filas = usuarios[ini:ini + tam_lote]
        topk = top_k(score_fn(filas), kmax, excluir_indptr, excluir_indices, filas)
        acum.anadir(topk, filas)
    return acum.resultado()
