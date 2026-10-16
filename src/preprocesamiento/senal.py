"""Etapa 2 (bloque C): señal objetivo, qué significa «le gusta».

Entrada: data/processed/<muestra>/split.parquet (+ mapeo_peliculas.parquet) y config.yaml.
Todo lo que se aprende de los datos (media_u, P(i), máscara de valorados) sale SOLO de train
(particion == 0): D3. Con la media_u de train se etiquetan las tres particiones.
  positivo    = rating >= min(tope_umbral, media_u)            (C2)
  nota_baja   = rating <= media_u - margen_nota_baja, y no positivo (C3)
  peso        = (1 + max(0, rating - media_u)) * peso_primer_dia si primer_dia; solo positivos de train (C4, C9)
  relevante_4 = rating >= umbral_fijo_eval                     (C7)
  logq        = log((pos_i + s) / (total_pos + s*N)) con positivos de train (C13)
  máscara     = items valorados en train por usuario, en CSR (C14)
Salida en data/processed/<muestra>/: senal.parquet, usuarios.parquet, logq.npy,
valorados_train_indptr.npy, valorados_train_indices.npy, meta_senal.json.

Uso: python -m src.preprocesamiento.senal --muestra {dev,completo} [--forzar]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from .comun import PROCESADOS, RAIZ, marcar

TAREA = "CPIA senal"
SALIDAS = ["senal.parquet", "usuarios.parquet", "logq.npy", "valorados_train_indptr.npy",
           "valorados_train_indices.npy", "meta_senal.json"]


def medias_train(user_idx: np.ndarray, rating: np.ndarray, particion: np.ndarray, n_usuarios: int) -> np.ndarray:
    """media_u cruda (C6) float64 [U] con SOLO las filas de train; NaN si el usuario no tiene train."""
    t = particion == 0
    suma = np.bincount(user_idx[t], weights=rating[t].astype("float64"), minlength=n_usuarios)
    n = np.bincount(user_idx[t], minlength=n_usuarios)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, suma / n, np.nan)


def etiquetar(df: pd.DataFrame, media: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Añade positivo, nota_baja y relevante_4 (bool) a una copia de df, con la media_u de train."""
    m = media[df.user_idx.to_numpy()]
    r = df.rating.to_numpy().astype("float64")
    out = df.copy()
    out["positivo"] = r >= np.minimum(cfg["tope_umbral"], m)
    # Si un usuario con media_u = 5,0 pone un 4,0 en val/test, cumple las dos: gana positivo (C2, «un 4 es un 4»).
    out["nota_baja"] = (r <= m - cfg["margen_nota_baja"]) & ~out["positivo"].to_numpy()
    out["relevante_4"] = r >= cfg["umbral_fijo_eval"]
    return out


def pesos(df: pd.DataFrame, media: np.ndarray, cfg: dict) -> np.ndarray:
    """peso float32: (1 + max(0, r - media_u)) * (peso_primer_dia si primer_dia) en positivos de train; 0 en el resto."""
    m = media[df.user_idx.to_numpy()]
    r = df.rating.to_numpy().astype("float64")
    w = 1.0 + np.maximum(0.0, r - m)
    w = w * np.where(df.primer_dia.to_numpy(), cfg["peso_primer_dia"], 1.0)
    valido = df.positivo.to_numpy() & (df.particion.to_numpy() == 0)
    return np.where(valido, w, 0.0).astype("float32")


def logq(item_idx: np.ndarray, positivo: np.ndarray, particion: np.ndarray, n_items: int, s: float) -> np.ndarray:
    """log P(i) float32 [N] con los positivos de train y suavizado s (C13); exp(logq) suma 1."""
    t = positivo & (particion == 0)
    pos = np.bincount(item_idx[t], minlength=n_items).astype("float64")
    return np.log((pos + s) / (pos.sum() + s * n_items)).astype("float32")


def mascara_valorados(user_idx: np.ndarray, item_idx: np.ndarray, particion: np.ndarray,
                      n_usuarios: int, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """C14: items valorados en train por usuario en CSR: indptr int64 [U+1], indices int32 (ordenados por item)."""
    t = particion == 0
    u, i = user_idx[t].astype("int64"), item_idx[t].astype("int64")
    orden = np.argsort(u * n_items + i, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n_usuarios))]).astype("int64")
    return indptr, i[orden].astype("int32")


def usuarios_elegibles(df: pd.DataFrame, media: np.ndarray, n_usuarios: int) -> pd.DataFrame:
    """Una fila por user_idx (C8). df ya tiene positivo y relevante_4.
    elegible_val/test: >= 1 positivo en esa partición y >= 1 positivo en train.
    elegible_val_4/test_4: lo mismo con relevante_4 en lugar de positivo en val/test."""
    u, p = df.user_idx.to_numpy(), df.particion.to_numpy()
    pos, rel = df.positivo.to_numpy(), df.relevante_4.to_numpy()

    def cuenta(mask):
        return np.bincount(u[mask], minlength=n_usuarios)

    n_pos_train = cuenta(pos & (p == 0))
    out = pd.DataFrame({
        "user_idx": np.arange(n_usuarios, dtype="int32"),
        "media_u": media.astype("float32"),
        "n_train": cuenta(p == 0).astype("int32"),
        "n_pos_train": n_pos_train.astype("int32"),
    })
    base = n_pos_train >= 1
    for nombre, k in (("val", 1), ("test", 2)):
        out[f"elegible_{nombre}"] = base & (cuenta(pos & (p == k)) >= 1)
    for nombre, k in (("val", 1), ("test", 2)):
        out[f"elegible_{nombre}_4"] = base & (cuenta(rel & (p == k)) >= 1)
    return out


def construir(df: pd.DataFrame, cfg: dict, n_usuarios: int, n_items: int) -> dict:
    """Todo el cálculo en memoria (sin E/S): devuelve senal, usuarios, logq, indptr, indices."""
    u, i, p = df.user_idx.to_numpy(), df.item_idx.to_numpy(), df.particion.to_numpy()
    media = medias_train(u, df.rating.to_numpy(), p, n_usuarios)
    senal = etiquetar(df, media, cfg)
    senal["peso"] = pesos(senal, media, cfg)
    indptr, indices = mascara_valorados(u, i, p, n_usuarios, n_items)
    return {
        "senal": senal,
        "usuarios": usuarios_elegibles(senal, media, n_usuarios),
        "logq": logq(i, senal.positivo.to_numpy(), p, n_items, cfg["suavizado_logq"]),
        "indptr": indptr,
        "indices": indices,
    }


def resumen(senal: pd.DataFrame, usuarios: pd.DataFrame, cfg: dict) -> dict:
    meta = {"filas": int(len(senal)), "usuarios": int(len(usuarios)), "parametros": cfg}
    for k, nombre in enumerate(["train", "val", "test"]):
        s = senal[senal.particion == k]
        for col in ("positivo", "nota_baja", "relevante_4"):
            meta[f"pct_{col}_{nombre}"] = round(100 * float(s[col].mean()), 3) if len(s) else None
    for c in ["elegible_val", "elegible_test", "elegible_val_4", "elegible_test_4"]:
        meta[f"usuarios_{c}"] = int(usuarios[c].sum())
    meta["usuarios_sin_positivos_train"] = int((usuarios.n_pos_train == 0).sum())
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()

    carpeta = PROCESADOS / a.muestra
    if all((carpeta / f).exists() for f in SALIDAS) and not a.forzar:
        print(f"{a.muestra}: la señal ya existe (usa --forzar para rehacerla)")
        return
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))["senal"]

    marcar(TAREA, 0, 4, "leyendo split")
    df = pd.read_parquet(carpeta / "split.parquet")
    n_items = len(pd.read_parquet(carpeta / "mapeo_peliculas.parquet", columns=["item_idx"]))
    f_usr = carpeta / "mapeo_usuarios.parquet"
    n_usuarios = len(pd.read_parquet(f_usr, columns=["user_idx"])) if f_usr.exists() else int(df.user_idx.max()) + 1

    marcar(TAREA, 1, 4, "calculando señal")
    r = construir(df, cfg, n_usuarios, n_items)

    marcar(TAREA, 2, 4, "guardando")
    r["senal"].to_parquet(carpeta / "senal.parquet", index=False)
    r["usuarios"].to_parquet(carpeta / "usuarios.parquet", index=False)
    np.save(carpeta / "logq.npy", r["logq"])
    np.save(carpeta / "valorados_train_indptr.npy", r["indptr"])
    np.save(carpeta / "valorados_train_indices.npy", r["indices"])

    marcar(TAREA, 3, 4, "meta")
    meta = resumen(r["senal"], r["usuarios"], cfg)
    meta["n_items"] = n_items
    (carpeta / "meta_senal.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    marcar(TAREA, 4, 4, "hecho")
    print(json.dumps({k: v for k, v in meta.items() if k != "parametros"}, indent=2))


if __name__ == "__main__":
    main()
