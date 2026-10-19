"""Pérdida contrastiva: softmax con los negativos del lote, corrección logQ (C13), máscara de valorados (C14) y peso (C4 x C9)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Valorados:
    """Máscara C14: ¿ha valorado el usuario u la película i en train? Búsqueda binaria sobre claves u * N + i
    (el CSR de `valorados_*` ya está ordenado por usuario y por película, así que las claves salen ordenadas)."""

    def __init__(self, indptr, indices, n_items: int, device="cpu"):
        indptr = torch.as_tensor(indptr, dtype=torch.int64)
        usuarios = torch.repeat_interleave(torch.arange(len(indptr) - 1), indptr[1:] - indptr[:-1])
        self.claves = (usuarios * n_items + torch.as_tensor(indices, dtype=torch.int64)).to(device)
        self.n_items = n_items

    def contiene(self, usuarios, items) -> torch.Tensor:
        """usuarios [B], items [M] -> bool [B, M]."""
        q = usuarios.long()[:, None] * self.n_items + items.long()[None, :]
        if len(self.claves) == 0:
            return torch.zeros_like(q, dtype=torch.bool)
        k = self.claves[torch.searchsorted(self.claves, q).clamp_max(len(self.claves) - 1)]
        return k == q


def mascara_lote(usuarios, items, valorados: Valorados) -> torch.Tensor:
    """[B, B] True donde la columna j es una película que el usuario i ha valorado en train, salvo su propio objetivo
    (la diagonal). Cubre también otro ejemplo del lote con la misma película que el objetivo."""
    m = valorados.contiene(usuarios, items)
    m.fill_diagonal_(False)
    return m


def perdida_softmax_logq(U, Vb, logq_b, tau: float, peso, enmascarar=None) -> torch.Tensor:
    """U [B,d] y Vb [B,d] (el objetivo de cada fila) con norma 1; logq_b [B] = log P(objetivo); peso [B].
    logits_ij = u_i · v_j / tau - log P(j); el positivo de la fila i es la columna i. Media ponderada por `peso`."""
    logits = U @ Vb.T / tau - logq_b[None, :]
    if enmascarar is not None:
        logits = logits.masked_fill(enmascarar, float("-inf"))
    ce = F.cross_entropy(logits, torch.arange(len(U), device=U.device), reduction="none")
    return (peso * ce).sum() / peso.sum().clamp_min(1e-12)
