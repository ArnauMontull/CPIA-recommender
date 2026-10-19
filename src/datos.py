"""Carga de data/processed/<muestra>/ para el modelo: features de películas, historiales (CSR), ejemplos y relevantes.

Todo en numpy; `entrenar.py` lo pasa al dispositivo. Los índices son los contiguos `user_idx` / `item_idx` de H3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluacion.metricas import csr_desde_pares

VISTAS = ("train", "train_val")     # F8: val se evalúa con train; test, con train + val
FLAGS = ["sin_generos", "sin_director", "sin_reparto", "sin_anio", "sin_duracion", "sin_texto"]


def _rellenar(listas, ancho: int | None = None) -> np.ndarray:
    """Lista de listas de ids -> int64 [N, L] con 0 (PAD) a la derecha."""
    ancho = max(1, max((len(x) for x in listas), default=1)) if ancho is None else ancho
    out = np.zeros((len(listas), ancho), dtype=np.int64)
    for i, x in enumerate(listas):
        x = list(x)[:ancho]
        out[i, :len(x)] = x
    return out


def cargar_features(carpeta: Path, emb_texto: Path | None = None) -> dict[str, np.ndarray]:
    """peliculas_features.parquet + emb_texto.npy -> arrays alineados por item_idx. `en_train`: la película
    tiene algún rating en train (las demás no han entrenado su embedding de movieId)."""
    carpeta = Path(carpeta)
    f = pd.read_parquet(carpeta / "peliculas_features.parquet").sort_values("item_idx").reset_index(drop=True)
    n = len(f)
    assert (f.item_idx.to_numpy() == np.arange(n)).all(), "item_idx no es contiguo"
    emb = np.load(emb_texto or carpeta.parent / "emb_texto.npy", mmap_mode="r")
    assert emb.shape[0] == n, f"emb_texto tiene {emb.shape[0]} filas y el catálogo {n}"
    generos = np.zeros((n, 16), dtype=np.float32)
    for i, g in enumerate(f.generos):
        generos[i, list(g)] = 1.0
    ip, ix = np.load(carpeta / "valorados_train_indptr.npy"), np.load(carpeta / "valorados_train_indices.npy")
    en_train = np.zeros(n, dtype=bool)
    en_train[np.unique(ix)] = True
    return {
        "texto": np.array(emb, dtype=np.float16),
        "generos": generos,
        "directores": _rellenar(f.directores),
        "actores": _rellenar(f.actores),
        "numerico": np.column_stack([f.anio_z, f.duracion_z] + [f[c].astype("float32") for c in FLAGS]).astype("float32"),
        "en_train": en_train,
    }


@dataclass
class Datos:
    carpeta: Path
    n_usuarios: int
    n_items: int
    feats: dict
    logq: np.ndarray
    ej_user: np.ndarray
    ej_item: np.ndarray
    ej_peso: np.ndarray
    hist: dict                  # (vista, "A"|"B") -> (indptr, indices, pesos)
    valorados: dict             # vista -> (indptr, indices): máscara C14
    pop_train: np.ndarray       # positivos de train por película (para la novedad)
    usuarios: pd.DataFrame
    _rel: dict = field(default_factory=dict)

    def relevantes(self, particion: int) -> tuple[dict, dict]:
        """(relevantes_por_definicion, usuarios_elegibles) de val (1) o test (2) para metricas.evaluar_*.
        Definiciones: `pos` (C2: positivo) y `r4` (C7: rating >= 4)."""
        if particion not in self._rel:
            s = pd.read_parquet(self.carpeta / "senal.parquet",
                                columns=["user_idx", "item_idx", "particion", "positivo", "relevante_4"])
            s = s[s.particion == particion]
            sufijo = {1: "val", 2: "test"}[particion]
            u = self.usuarios
            rel = {"pos": csr_desde_pares(s.user_idx[s.positivo], s.item_idx[s.positivo], self.n_usuarios),
                   "r4": csr_desde_pares(s.user_idx[s.relevante_4], s.item_idx[s.relevante_4], self.n_usuarios)}
            elig = {"pos": u.user_idx[u[f"elegible_{sufijo}"]].to_numpy(),
                    "r4": u.user_idx[u[f"elegible_{sufijo}_4"]].to_numpy()}
            self._rel[particion] = (rel, elig)
        return self._rel[particion]


def cargar(carpeta: str | Path, emb_texto: Path | None = None) -> Datos:
    c = Path(carpeta)
    usuarios = pd.read_parquet(c / "usuarios.parquet")
    ej = pd.read_parquet(c / "ejemplos_train.parquet")
    meta = json.loads((c / "meta_historial.json").read_text(encoding="utf-8"))
    hist = {(v, b): tuple(np.load(c / f"hist_{v}_{b}_{x}.npy") for x in ("indptr", "indices", "pesos"))
            for v in VISTAS for b in "AB"}
    valorados = {v: (np.load(c / f"valorados_{v}_indptr.npy"), np.load(c / f"valorados_{v}_indices.npy")) for v in VISTAS}
    n_items = int(meta["n_items"])
    return Datos(
        carpeta=c, n_usuarios=len(usuarios), n_items=n_items,
        feats=cargar_features(c, emb_texto), logq=np.load(c / "logq.npy"),
        ej_user=ej.user_idx.to_numpy(), ej_item=ej.item_idx.to_numpy(), ej_peso=ej.peso.to_numpy(),
        hist=hist, valorados=valorados,
        pop_train=np.bincount(ej.item_idx.to_numpy(), minlength=n_items), usuarios=usuarios)
