"""Exporta el modelo entrenado al formato de la guía H10 (binarios little-endian row-major + meta.txt/meta.json) para el
motor en C de Uriel, con un usuario de ejemplo y las salidas de referencia calculadas en numpy a partir de los propios
archivos exportados (así Uriel valida contra lo que ve el C, no contra PyTorch).

Uso: python -m src.exportar_c --muestra dev --checkpoint modelos/dev_tau0.05_s42/mejor.pt --salida c/datos_modelo [--usuario 17]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .datos import Datos, cargar
from .modelo import UMBRAL_VACIA, Bolsa, DosTorres
from .preprocesamiento.comun import PROCESADOS, RAIZ

VERSION = 1
LAMBDA_MMR = 0.7


def _fila(csr, u: int):
    indptr, indices = csr[0], csr[1]
    return indices[indptr[u]:indptr[u + 1]], (csr[2][indptr[u]:indptr[u + 1]] if len(csr) > 2 else None)


def referencia_numpy(V, W1, b1, W2, b2, A, B, excl) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u, scores y top-200 con el algoritmo de la guía (§Validación), solo con los arrays exportados."""
    d = V.shape[1]

    def media(idx, w):
        if w.sum() <= UMBRAL_VACIA:
            return np.zeros(d), 1.0
        return (w[:, None].astype("f8") * V[idx].astype("f8")).sum(0) / w.sum(), 0.0

    a, fa = media(*A)
    b, fb = media(*B)
    x = np.concatenate([a, b, [fa, fb, np.log1p(len(A[0]) + len(B[0]))]]).astype("f4")
    z = W2 @ np.maximum(0, W1 @ x + b1) + b2
    u = (z / max(np.linalg.norm(z), 1e-12)).astype("f4")
    s = V @ u
    s[excl] = -np.inf
    top = np.lexsort((np.arange(len(s)), -s))[:200]          # score desc., empate -> índice menor
    return u, s.astype("f4"), top.astype("i4")


def exportar(modelo: DosTorres, datos: Datos, salida: Path, usuario: int | None = None, vista: str = "train") -> dict:
    """Escribe los archivos del contrato en `salida`. Devuelve el meta. Comprueba que numpy y PyTorch dan el mismo u."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    modelo.eval()
    with torch.no_grad():
        V_t = modelo.vectores_items()
    V = V_t.numpy().astype("<f4")
    lin1, lin2 = modelo.usuario.mlp[0], modelo.usuario.mlp[2]
    W1, b1 = lin1.weight.detach().numpy().astype("<f4"), lin1.bias.detach().numpy().astype("<f4")
    W2, b2 = lin2.weight.detach().numpy().astype("<f4"), lin2.bias.detach().numpy().astype("<f4")

    if usuario is None:                                       # el primer elegible de val con bolsa B no vacía
        _, elig = datos.relevantes(1)
        cand = [u for u in elig["pos"] if datos.hist[(vista, "B")][0][u + 1] > datos.hist[(vista, "B")][0][u]]
        usuario = int(cand[0] if cand else elig["pos"][0])
    A = _fila(datos.hist[(vista, "A")], usuario)
    B = _fila(datos.hist[(vista, "B")], usuario)
    excl = _fila(datos.valorados[vista], usuario)[0].astype("<i4")
    A = (A[0].astype("<i4"), A[1].astype("<f4"))
    B = (B[0].astype("<i4"), B[1].astype("<f4"))

    u_ref, s_ref, top_ref = referencia_numpy(V, W1, b1, W2, b2, A, B, excl)
    with torch.no_grad():                                     # el torch y el numpy deben coincidir
        u_t = modelo.vectores_usuarios(V_t, torch.tensor([usuario]), Bolsa(*datos.hist[(vista, "A")]),
                                       Bolsa(*datos.hist[(vista, "B")]))[0].numpy()
    dif = float(np.abs(u_t - u_ref).max())
    assert dif < 1e-4, f"numpy y PyTorch difieren en u ({dif:.2e}): el contrato de exportación está mal"

    ids = datos.carpeta / "mapeo_peliculas.parquet"
    if ids.exists():
        import pandas as pd
        movie_ids = pd.read_parquet(ids).sort_values("item_idx").movieId.to_numpy().astype("<i4")
    else:
        movie_ids = np.arange(len(V), dtype="<i4")
    archivos = dict(V=V, W1=W1, b1=b1, W2=W2, b2=b2, movie_ids=movie_ids, usuario_A_idx=A[0], usuario_A_w=A[1],
                    usuario_B_idx=B[0], usuario_B_w=B[1], usuario_excl=excl, ref_u=u_ref, ref_scores=s_ref, ref_top=top_ref)
    for nombre, arr in archivos.items():
        arr.tofile(salida / f"{nombre}.{'i32' if arr.dtype.kind == 'i' else 'f32'}")

    meta = {"version": VERSION, "N": int(V.shape[0]), "d": int(V.shape[1]), "d0": int(W1.shape[1]), "h1": int(W1.shape[0]),
            "relu_final": 0, "orden_entrada": "media_A,media_B,vacia_A,vacia_B,log1p_n", "n_para_log": "A+B",
            "umbral_vacia": UMBRAL_VACIA, "normalizado_V": 1, "lambda_mmr": LAMBDA_MMR,
            "usuario_ejemplo": usuario, "n_A": len(A[0]), "n_B": len(B[0]), "n_excl": len(excl),
            "dif_max_torch_numpy": dif}
    (salida / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (salida / "meta.txt").write_text("# contrato de exportación v%d\n" % VERSION
                                     + "".join(f"{k} = {v}\n" for k, v in meta.items()), encoding="utf-8")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--salida", type=Path, default=RAIZ / "c" / "datos_modelo")
    ap.add_argument("--usuario", type=int)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    datos = cargar(PROCESADOS / a.muestra)
    modelo = DosTorres.desde_config(datos.feats, cfg)
    modelo.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["modelo"])
    meta = exportar(modelo, datos, a.salida, a.usuario)
    print(f"Exportado a {a.salida}: N={meta['N']} d={meta['d']} usuario {meta['usuario_ejemplo']} "
          f"(dif torch/numpy {meta['dif_max_torch_numpy']:.1e})")


if __name__ == "__main__":
    main()
