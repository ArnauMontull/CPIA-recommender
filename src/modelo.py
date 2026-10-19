"""Two-Tower (bloques E y F). Item: contenido + embedding de movieId con dropout -> MLP -> d -> L2.
Usuario: medias ponderadas del historial con los v_i de la torre de ítem (F6) -> MLP de 2 capas -> d -> L2.

La torre de ítem se evalúa sobre TODO el catálogo una vez por paso (23 335 películas): V [N, d]. El historial de
cada usuario y los negativos del lote salen de ese mismo V, así que los gradientes llegan a la torre de ítem por
las dos vías. Los CSR de los historiales van como tensores (indptr, indices, pesos) en el dispositivo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
torch.sparse.check_sparse_tensor_invariants.disable()
import torch.nn.functional as F

UMBRAL_VACIA = 1e-4     # suma de pesos <= esto: bolsa vacía (el peso mínimo real es 0,25 * 0,75)


class Bolsa:
    """CSR de un historial (una bolsa de una vista) como tensores en el dispositivo."""

    def __init__(self, indptr, indices, pesos, device="cpu"):
        self.indptr = torch.as_tensor(indptr, dtype=torch.int64, device=device)
        self.indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
        self.pesos = torch.as_tensor(pesos, dtype=torch.float32, device=device)


def media_bolsa(V, bolsa: Bolsa, filas, excluir_item=None, excluir_peso=None):
    """Versión torch de `historial.media_bolsa`: (sum w_j v_j - w_t v_t) / (sum w_j - w_t).
    Devuelve (media [B,d], vacia [B] bool, n [B] nº de elementos sin el objetivo). Diferenciable respecto a V.
    La suma ponderada es un producto disperso [B,N] @ [N,d] (no materializa B x |historial| x d)."""
    filas = filas.long()
    B, N = len(filas), V.shape[0]
    ini = bolsa.indptr[filas]
    largos = bolsa.indptr[filas + 1] - ini
    total = int(largos.sum())
    seg = torch.repeat_interleave(torch.arange(B, device=V.device), largos)
    pos = torch.repeat_interleave(ini, largos) + torch.arange(total, device=V.device) \
        - torch.repeat_interleave(torch.cumsum(largos, 0) - largos, largos)
    cols, w = bolsa.indices[pos], bolsa.pesos[pos]
    W = torch.sparse_coo_tensor(torch.stack([seg, cols]), w, (B, N))
    S = torch.sparse.mm(W, V)
    sw = torch.zeros(B, device=V.device, dtype=w.dtype).index_add_(0, seg, w)
    n = largos.clone()
    if excluir_item is not None:
        ew = excluir_peso.to(V.dtype)
        S = S - ew[:, None] * V[excluir_item.long()]
        sw = sw - ew
        n = n - 1
    vacia = sw <= UMBRAL_VACIA
    media = torch.where(vacia[:, None], torch.zeros_like(S), S / sw.clamp_min(UMBRAL_VACIA)[:, None])
    return media, vacia, n


class TorreItem(nn.Module):
    def __init__(self, feats: dict, d: int, oculta: int, dim_texto: int, dim_credito: int, dim_id: int, dropout_id: float):
        super().__init__()
        n_items = len(feats["texto"])
        for k in ("texto", "generos", "directores", "actores", "numerico", "en_train"):
            self.register_buffer(k, torch.as_tensor(feats[k]), persistent=False)
        self.dropout_id = dropout_id
        self.texto_denso = nn.Linear(feats["texto"].shape[1], dim_texto)     # E8: el MiniLM va congelado
        self.emb_dir = nn.EmbeddingBag(int(feats["directores"].max()) + 1, dim_credito, mode="mean", padding_idx=0)
        self.emb_act = nn.EmbeddingBag(int(feats["actores"].max()) + 1, dim_credito, mode="mean", padding_idx=0)
        self.emb_id = nn.Embedding(n_items, dim_id)
        for e in (self.emb_dir, self.emb_act, self.emb_id):
            nn.init.normal_(e.weight, std=0.1)
        with torch.no_grad():
            self.emb_dir.weight[0].zero_()
            self.emb_act.weight[0].zero_()
        entrada = dim_texto + feats["generos"].shape[1] + 2 * dim_credito + feats["numerico"].shape[1] + dim_id
        self.mlp = nn.Sequential(nn.Linear(entrada, oculta), nn.ReLU(), nn.Linear(oculta, d))   # sin ReLU final: va L2

    def forward(self) -> torch.Tensor:
        """V [N, d] con filas de norma 1."""
        ident = self.emb_id.weight * self.en_train[:, None]       # sin rating en train: el ID no se ha entrenado
        if self.training and self.dropout_id > 0:                 # E6/T3: película entera sin ID, con reescalado
            mantener = (torch.rand(len(ident), 1, device=ident.device) >= self.dropout_id).to(ident.dtype)
            ident = ident * mantener / (1.0 - self.dropout_id)
        x = torch.cat([F.relu(self.texto_denso(self.texto.float())), self.generos,
                       self.emb_dir(self.directores), self.emb_act(self.actores), self.numerico, ident], dim=1)
        return F.normalize(self.mlp(x), dim=1)


class TorreUsuario(nn.Module):
    """[media A, media B, vacía A, vacía B, log1p(n)] -> MLP de 2 capas -> d -> L2 (F7)."""

    def __init__(self, d: int, oculta: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d + 3, oculta), nn.ReLU(), nn.Linear(oculta, d))

    def forward(self, V, filas, A: Bolsa, B: Bolsa, excluir_item=None, excluir_peso=None) -> torch.Tensor:
        """`excluir_*` = el positivo objetivo, que sale de la bolsa A (F3); solo en entrenamiento."""
        mA, vA, nA = media_bolsa(V, A, filas, excluir_item, excluir_peso)
        mB, vB, nB = media_bolsa(V, B, filas)
        x = torch.cat([mA, mB, vA[:, None].float(), vB[:, None].float(), torch.log1p((nA + nB).float())[:, None]], dim=1)
        return F.normalize(self.mlp(x), dim=1)


class DosTorres(nn.Module):
    def __init__(self, feats: dict, d: int = 128, oculta: int = 256, dim_texto: int = 128, dim_credito: int = 32,
                 dim_id: int = 32, dropout_id: float = 0.5):
        super().__init__()
        self.items = TorreItem(feats, d, oculta, dim_texto, dim_credito, dim_id, dropout_id)
        self.usuario = TorreUsuario(d, oculta)

    @classmethod
    def desde_config(cls, feats: dict, cfg: dict) -> "DosTorres":
        m = cfg["modelo"]
        return cls(feats, d=cfg["historial"]["dim"], oculta=m["oculta"], dim_texto=m["dim_texto"],
                   dim_credito=m["dim_credito"], dim_id=m["dim_id"], dropout_id=m["dropout_id"])

    def vectores_items(self) -> torch.Tensor:
        return self.items()

    def vectores_usuarios(self, V, filas, A: Bolsa, B: Bolsa, excluir_item=None, excluir_peso=None) -> torch.Tensor:
        return self.usuario(V, filas, A, B, excluir_item, excluir_peso)
