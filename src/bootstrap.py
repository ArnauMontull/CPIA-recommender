"""Intervalos de confianza por bootstrap de usuarios para la diferencia entre métodos (misma lista de usuarios para todos).

Para cada método se calcula el Recall@k y el NDCG@k de cada usuario en test (historial = train + val, F8, máscara C14); después se
remuestrean usuarios con reemplazo y se obtiene el intervalo del 95 % de la diferencia Two-Tower − baseline. Si el intervalo no
contiene el 0, la diferencia no se explica por qué usuarios entraron en la muestra (no mide la variabilidad entre semillas, que se
ve en `resumen_semillas`).

Uso: python -m src.bootstrap --muestra completo --checkpoint .../mejor.pt --device cuda --usuarios-max 20000 --etiqueta s42
Escribe reports/bootstrap_<etiqueta>.md y .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baselines import EASE, _usuarios
from .datos import Datos, cargar
from .entrenar import Contexto, vectores_usuarios_eval
from .evaluacion.metricas import ndcg_at_k, recall_at_k, top_k
from .modelo import DosTorres
from .preprocesamiento.comun import INFORMES, PROCESADOS, RAIZ

KS = (20, 200)


def metricas_por_usuario(score_fn, usuarios: np.ndarray, rel: tuple, excl: tuple, ks=KS, lote: int = 1024) -> dict[str, np.ndarray]:
    """{"recall@k": [U], "ndcg@k": [U]} (NaN si el usuario no tiene relevantes)."""
    out = {f"{m}@{k}": [] for m in ("recall", "ndcg") for k in ks}
    for ini in range(0, len(usuarios), lote):
        f = usuarios[ini:ini + lote]
        tk = top_k(score_fn(f), max(ks), excl[0], excl[1], f)
        for k in ks:
            out[f"recall@{k}"].append(recall_at_k(tk, rel[0], rel[1], k, f))
            out[f"ndcg@{k}"].append(ndcg_at_k(tk, rel[0], rel[1], k, f))
    return {k: np.concatenate(v) for k, v in out.items()}


def ic_diferencia(a: np.ndarray, b: np.ndarray, n: int = 2000, semilla: int = 0) -> tuple[float, float, float]:
    """(diferencia media a − b, límite inferior, límite superior) del IC del 95 % por bootstrap de usuarios (pares)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1) if len(d) * n <= 4e8 else np.array(
        [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def ejecutar(datos: Datos, modelo: DosTorres, ctx: Contexto, ease: EASE | None, usuarios_max: int | None, semilla: int = 0, n: int = 2000) -> dict:
    usuarios, rel, elig = _usuarios(datos, 2, usuarios_max, semilla)
    usuarios = np.intersect1d(usuarios, elig["pos"])
    modelo.eval()
    with torch.no_grad():
        V = modelo.vectores_items()
        tt = lambda f: (vectores_usuarios_eval(modelo, ctx, V, f, "train_val") @ V.T).cpu().numpy()
        pop = datos.pop_train.astype("float32")
        metodos = {"Two-Tower": tt, "Popularidad": lambda f: np.tile(pop, (len(f), 1))}
        if ease is not None:
            metodos["EASE"] = lambda f: ease.scores(f, "train_val")
        por = {nombre: metricas_por_usuario(fn, usuarios, rel["pos"], datos.valorados["train_val"]) for nombre, fn in metodos.items()}
    res = {"n_usuarios": int(len(usuarios)), "medias": {m: {k: float(np.nanmean(v)) for k, v in d.items()} for m, d in por.items()}, "diferencias": {}}
    for base in [m for m in por if m != "Two-Tower"]:
        for k in por["Two-Tower"]:
            res["diferencias"][f"Two-Tower − {base} | {k}"] = dict(zip(("dif", "ic_inf", "ic_sup"), ic_diferencia(por["Two-Tower"][k], por[base][k], n, semilla)))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--muestra", default="completo")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--usuarios-max", type=int, default=20000)
    ap.add_argument("--max-items-ease", type=int, default=10000)
    ap.add_argument("--sin-ease", action="store_true")
    ap.add_argument("--etiqueta", required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    datos = cargar(PROCESADOS / a.muestra)
    modelo = DosTorres.desde_config(datos.feats, cfg).to(a.device)
    modelo.load_state_dict(torch.load(a.checkpoint, map_location=a.device, weights_only=True)["modelo"])
    ctx = Contexto(datos, a.device)
    ease = None if a.sin_ease else EASE(datos, 500.0, a.max_items_ease)
    r = ejecutar(datos, modelo, ctx, ease, a.usuarios_max or None, cfg["semilla"])
    (INFORMES / f"bootstrap_{a.etiqueta}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    md = [f"# Bootstrap de usuarios `{a.etiqueta}` (test, {r['n_usuarios']} usuarios, 2000 remuestreos)", "",
          "| Comparación | Métrica | Diferencia | IC 95 % | ¿Excluye el 0? |", "|---|---|---:|---|:---:|"]
    for clave, v in r["diferencias"].items():
        comp, met = clave.split(" | ")
        md.append(f"| {comp} | {met} | {v['dif']:+.4f} | [{v['ic_inf']:+.4f}, {v['ic_sup']:+.4f}] | {'sí' if v['ic_inf'] > 0 or v['ic_sup'] < 0 else 'no'} |")
    (INFORMES / f"bootstrap_{a.etiqueta}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
