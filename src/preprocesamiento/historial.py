"""Etapa de historiales (bloque F: F2-F5, F8): los ratings del usuario como bolsas ponderadas en CSR.

Entrada: data/processed/<muestra>/senal.parquet, usuarios.parquet, mapeo_peliculas.parquet y config.yaml.
Bolsa A = positivos + neutras (F4 media 1); bolsa B = notas bajas (F4 media 2). Pesos (F4/F5), c9 = peso_primer_dia si primer_dia:
  positivo -> (1 + max(0, r - media_u)) * c9      (media_u de train; = `peso` de senal en train)
  neutra   -> peso_neutra * c9
  nota_baja-> 1 * c9
Vistas (F8): `train` (particion 0; entrena y evalua val) y `train_val` (particiones 0 y 1; evalua test). Test nunca entra.
Salida en data/processed/<muestra>/: hist_<vista>_<A|B>_{indptr,indices,pesos}.npy (CSR por user_idx, items ordenados),
valorados_train_val_{indptr,indices}.npy (mascara C14 de test), ejemplos_train.parquet (un positivo de train por fila: F3),
meta_historial.json. El historial de un ejemplo = bolsa A de train sin el objetivo + bolsa B; no se materializa:
media_bolsa(..., excluir_item, excluir_peso) resta el objetivo de la suma: (S_u - w_t v_t) / (W_u - w_t).

Uso: python -m src.preprocesamiento.historial --muestra {dev,completo} [--forzar]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from .comun import PROCESADOS, RAIZ, marcar
from .senal import mascara_valorados

TAREA = "CPIA historial"
VISTAS = {"train": 0, "train_val": 1}          # vista -> particion maxima incluida
BOLSAS = ("A", "B")
SALIDAS = (["ejemplos_train.parquet", "meta_historial.json", "valorados_train_val_indptr.npy",
            "valorados_train_val_indices.npy"]
           + [f"hist_{v}_{b}_{x}.npy" for v in VISTAS for b in BOLSAS for x in ("indptr", "indices", "pesos")])


def pesos_historial(df: pd.DataFrame, media: np.ndarray, cfg_senal: dict, cfg_hist: dict) -> tuple[np.ndarray, np.ndarray]:
    """(peso float32, es_B bool) por rating. df: user_idx, rating, primer_dia, positivo, nota_baja.
    es_B = nota baja (bolsa B); el resto es bolsa A (positivo o neutra)."""
    m = media[df.user_idx.to_numpy()]
    r = df.rating.to_numpy().astype("float64")
    pos, baja = df.positivo.to_numpy(), df.nota_baja.to_numpy()
    c9 = np.where(df.primer_dia.to_numpy(), cfg_senal["peso_primer_dia"], 1.0)
    w = np.where(pos, 1.0 + np.maximum(0.0, r - m), np.where(baja, 1.0, cfg_hist["peso_neutra"])) * c9
    return w.astype("float32"), baja.astype(bool)


def csr(user_idx: np.ndarray, item_idx: np.ndarray, pesos: np.ndarray, n_usuarios: int, n_items: int):
    """CSR por usuario con los items ordenados: indptr int64 [U+1], indices int32, pesos float32."""
    u, i = user_idx.astype("int64"), item_idx.astype("int64")
    orden = np.argsort(u * n_items + i, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n_usuarios))]).astype("int64")
    return indptr, i[orden].astype("int32"), np.asarray(pesos)[orden].astype("float32")


def construir_vistas(df: pd.DataFrame, media: np.ndarray, cfg_senal: dict, cfg_hist: dict,
                     n_usuarios: int, n_items: int) -> dict:
    """{(vista, bolsa): (indptr, indices, pesos)}."""
    w, es_b = pesos_historial(df, media, cfg_senal, cfg_hist)
    u, i, p = df.user_idx.to_numpy(), df.item_idx.to_numpy(), df.particion.to_numpy()
    out = {}
    for vista, pmax in VISTAS.items():
        dentro = p <= pmax
        for bolsa in BOLSAS:
            m = dentro & (es_b if bolsa == "B" else ~es_b)
            out[(vista, bolsa)] = csr(u[m], i[m], w[m], n_usuarios, n_items)
    return out


def ejemplos_train(df: pd.DataFrame) -> pd.DataFrame:
    """C5 a / F3: una fila por positivo de train con su peso (el de senal)."""
    m = (df.particion.to_numpy() == 0) & df.positivo.to_numpy()
    return pd.DataFrame({"user_idx": df.user_idx.to_numpy()[m].astype("int32"),
                         "item_idx": df.item_idx.to_numpy()[m].astype("int32"),
                         "peso": df.peso.to_numpy()[m].astype("float32")}).reset_index(drop=True)


def media_bolsa(V: np.ndarray, indptr: np.ndarray, indices: np.ndarray, pesos: np.ndarray, filas: np.ndarray,
                excluir_item: np.ndarray | None = None, excluir_peso: np.ndarray | None = None):
    """Referencia numpy (la copiara el modelo en torch). V [N,d]; filas = user_idx [B].
    media = (sum w_j v_j - w_t v_t) / (sum w_j - w_t) con el objetivo t = (excluir_item, excluir_peso) opcional.
    Devuelve (medias [B,d] float32, vacia [B] bool); suma de pesos 0 -> vector 0 y vacia=True."""
    filas = np.asarray(filas, dtype="int64")
    ini, fin = indptr[filas], indptr[filas + 1]
    largos = fin - ini
    B = len(filas)
    seg = np.repeat(np.arange(B), largos)
    pos = np.arange(largos.sum()) - np.repeat(np.cumsum(largos) - largos, largos) + np.repeat(ini, largos)
    w = pesos[pos].astype("float64")
    S = np.zeros((B, V.shape[1]), dtype="float64")
    np.add.at(S, seg, w[:, None] * V[indices[pos]].astype("float64"))
    W = np.bincount(seg, weights=w, minlength=B)
    if excluir_item is not None:
        ew = np.asarray(excluir_peso, dtype="float64")
        S -= ew[:, None] * V[np.asarray(excluir_item, dtype="int64")].astype("float64")
        W = W - ew
    vacia = W <= 1e-9
    medias = np.where(vacia[:, None], 0.0, S / np.where(vacia, 1.0, W)[:, None])
    return medias.astype("float32"), vacia


def _percentiles(n: np.ndarray) -> dict:
    return {"p50": float(np.percentile(n, 50)), "p90": float(np.percentile(n, 90)),
            "p99": float(np.percentile(n, 99)), "max": int(n.max()), "media": round(float(n.mean()), 3)}


def resumen(vistas: dict, ej: pd.DataFrame, n_usuarios: int) -> dict:
    meta = {"usuarios": n_usuarios, "ejemplos_train": int(len(ej)), "tamano_bolsa_por_usuario": {}}
    for (vista, bolsa), (indptr, _, _) in vistas.items():
        meta["tamano_bolsa_por_usuario"][f"{vista}_{bolsa}"] = _percentiles(np.diff(indptr))
    nA, nB = np.diff(vistas[("train", "A")][0]), np.diff(vistas[("train", "B")][0])
    meta["usuarios_bolsa_B_vacia_train"] = int((nB == 0).sum())
    meta["usuarios_bolsa_B_vacia_train_val"] = int((np.diff(vistas[("train_val", "B")][0]) == 0).sum())
    # un ejemplo queda sin historial A si la bolsa A de su usuario solo contiene el objetivo
    meta["ejemplos_con_A_vacia_sin_objetivo"] = int((nA[ej.user_idx.to_numpy()] <= 1).sum())
    meta["memoria_bytes"] = {f"{v}_{b}": int(sum(a.nbytes for a in arr)) for (v, b), arr in vistas.items()}
    meta["memoria_bytes_total"] = int(sum(meta["memoria_bytes"].values()))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()

    carpeta = PROCESADOS / a.muestra
    if all((carpeta / f).exists() for f in SALIDAS) and not a.forzar:
        print(f"{a.muestra}: los historiales ya existen (usa --forzar para rehacerlos)")
        return
    t0 = time.time()
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))

    marcar(TAREA, 0, 4, "leyendo senal")
    df = pd.read_parquet(carpeta / "senal.parquet")
    media = pd.read_parquet(carpeta / "usuarios.parquet", columns=["user_idx", "media_u"]).media_u.to_numpy().astype("float64")
    n_usuarios = len(media)
    n_items = len(pd.read_parquet(carpeta / "mapeo_peliculas.parquet", columns=["item_idx"]))

    marcar(TAREA, 1, 4, "construyendo CSR")
    vistas = construir_vistas(df, media, cfg["senal"], cfg["historial"], n_usuarios, n_items)
    ej = ejemplos_train(df)
    ip, ix = mascara_valorados(df.user_idx.to_numpy(), df.item_idx.to_numpy(),
                               np.where(df.particion.to_numpy() <= 1, 0, 2), n_usuarios, n_items)

    marcar(TAREA, 2, 4, "guardando")
    for (v, b), (indptr, indices, pesos) in vistas.items():
        np.save(carpeta / f"hist_{v}_{b}_indptr.npy", indptr)
        np.save(carpeta / f"hist_{v}_{b}_indices.npy", indices)
        np.save(carpeta / f"hist_{v}_{b}_pesos.npy", pesos)
    np.save(carpeta / "valorados_train_val_indptr.npy", ip)
    np.save(carpeta / "valorados_train_val_indices.npy", ix)
    ej.to_parquet(carpeta / "ejemplos_train.parquet", index=False)

    marcar(TAREA, 3, 4, "meta")
    meta = resumen(vistas, ej, n_usuarios)
    meta["n_items"] = n_items
    meta["segundos"] = round(time.time() - t0, 2)
    meta["parametros"] = {"senal": cfg["senal"], "historial": cfg["historial"]}
    (carpeta / "meta_historial.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    marcar(TAREA, 4, 4, "hecho")
    print(json.dumps({k: v for k, v in meta.items() if k != "parametros"}, indent=2))


if __name__ == "__main__":
    main()
