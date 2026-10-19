"""Comprobaciones de un checkpoint sobre datos reales: que el modelo y la evaluación hacen lo que dicen.

Cada comprobación devuelve (ok, detalle). Se pueden ejecutar con cualquier muestra (`dev`, `completo`...); los ítems comparten
índices entre muestras, así que un checkpoint entrenado en `completo` se comprueba sobre los usuarios de `dev` (un
subconjunto, con el mismo split por usuario) con mucha menos RAM, pero con `--feats completo`: los vocabularios de directores y
actores se ajustan por muestra y no coinciden.

Uso: python -m src.comprobaciones --muestra dev --feats completo --checkpoint modelos/completo_tau0.05_s42/mejor.pt --etiqueta s42
Escribe reports/comprobaciones_<etiqueta>.json y .md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .datos import Datos, cargar, cargar_features
from .entrenar import Contexto, evaluar_particion, vectores_usuarios_eval
from .evaluacion.metricas import evaluar_por_lotes, recall_at_k, top_k
from .exportar_c import referencia_numpy
from .modelo import DosTorres
from .preprocesamiento.comun import INFORMES, PROCESADOS, RAIZ


def _k(datos: Datos) -> int:
    """200 en el catálogo real; menos si el catálogo es pequeño (un top-k que abarca todo no comprueba nada)."""
    return min(200, max(5, datos.n_items // 5))


def _usuarios_val(datos: Datos, n: int, semilla: int = 0) -> np.ndarray:
    _, elig = datos.relevantes(1)
    u = np.unique(elig["pos"])
    return np.sort(np.random.default_rng(semilla).choice(u, min(n, len(u)), replace=False))


def c_finitud_y_normas(modelo, ctx, datos):
    V = modelo.eval().vectores_items().detach()
    U = vectores_usuarios_eval(modelo, ctx, V, _usuarios_val(datos, 3000), "train").detach()
    ok = bool(torch.isfinite(V).all() and torch.isfinite(U).all())
    nv, nu = (V.norm(dim=1) - 1).abs().max().item(), (U.norm(dim=1) - 1).abs().max().item()
    return ok and nv < 1e-4 and nu < 1e-4, f"V {tuple(V.shape)}, U {tuple(U.shape)}; |‖·‖-1| máx: V {nv:.1e}, U {nu:.1e}"


def c_determinismo(modelo, ctx, datos):
    a = evaluar_particion(modelo, ctx, 1, [10, 200], 2000, 7)
    b = evaluar_particion(modelo, ctx, 1, [10, 200], 2000, 7)
    return a == b, f"dos evaluaciones idénticas (recall@200 = {a['recall@200/pos']:.4f})"


def c_exclusion_c14(modelo, ctx, datos):
    """Ningún top-200 contiene algo ya valorado en train (val) ni en train + val (test)."""
    malos, total = 0, 0
    for part, vista in ((1, "train"), (2, "train_val")):
        _, elig = datos.relevantes(part)
        us = np.sort(np.random.default_rng(1).choice(elig["pos"], min(1500, len(elig["pos"])), replace=False))
        ip, ix = datos.valorados[vista]
        modelo.eval()
        with torch.no_grad():
            V = modelo.vectores_items()
            for ini in range(0, len(us), 500):
                f = us[ini:ini + 500]
                s = (vectores_usuarios_eval(modelo, ctx, V, f, vista) @ V.T).cpu().numpy()
                tk = top_k(s, _k(datos), ip, ix, f)
                for fila, u in zip(tk, f):
                    malos += int(np.isin(fila, ix[ip[u]:ip[u + 1]]).sum())
                    total += len(fila)
    return malos == 0, f"{malos} recomendaciones ya valoradas de {total} (top-{_k(datos)}, val y test)"


def c_metricas_azar_y_oraculo(modelo, ctx, datos):
    """Con scores aleatorios el recall@k ≈ k/N de lo no valorado; con el oráculo, el techo."""
    k = _k(datos)
    us = _usuarios_val(datos, 3000)
    rel, elig = datos.relevantes(1)
    ip, ix = datos.valorados["train"]
    rng = np.random.default_rng(3)
    N = datos.n_items
    azar = evaluar_por_lotes(lambda f: rng.random((len(f), N), dtype=np.float32), us, rel, elig, [k], datos.pop_train, N, ip, ix)
    vistas = (ip[us + 1] - ip[us]).astype(float)          # candidatos de cada usuario = N - lo ya valorado
    esperado = float(np.mean(k / (N - vistas)))
    cerca = abs(azar[f"recall@{k}/pos"] - esperado) < 0.4 * esperado + 0.003
    rip, rix = rel["pos"]

    def oraculo(f):                                      # puntúa 1 a lo relevante de val
        s = np.zeros((len(f), N), dtype=np.float32)
        for j, u in enumerate(f):
            s[j, rix[rip[u]:rip[u + 1]]] = 1.0
        return s
    orac = evaluar_por_lotes(oraculo, us, rel, elig, [k], datos.pop_train, N, ip, ix)
    techo = orac[f"techo_recall@{k}/pos"]
    return cerca and abs(orac[f"recall@{k}/pos"] - techo) < 1e-3, (
        f"k={k}: azar {azar[f'recall@{k}/pos']:.4f} (≈ {esperado:.4f}); oráculo {orac[f'recall@{k}/pos']:.4f} = techo {techo:.4f}")


def c_numpy_igual_torch(modelo, ctx, datos):
    """u de PyTorch = u de la referencia numpy de la exportación a C, para 150 usuarios."""
    modelo.eval()
    with torch.no_grad():
        V = modelo.vectores_items()
    Vn = V.numpy().astype("<f4")
    l1, l2 = modelo.usuario.mlp[0], modelo.usuario.mlp[2]
    W1, b1, W2, b2 = (x.detach().numpy().astype("<f4") for x in (l1.weight, l1.bias, l2.weight, l2.bias))
    peor = 0.0
    for u in _usuarios_val(datos, 150, 5):
        A = (datos.hist[("train", "A")][1][datos.hist[("train", "A")][0][u]:datos.hist[("train", "A")][0][u + 1]],
             datos.hist[("train", "A")][2][datos.hist[("train", "A")][0][u]:datos.hist[("train", "A")][0][u + 1]])
        B = (datos.hist[("train", "B")][1][datos.hist[("train", "B")][0][u]:datos.hist[("train", "B")][0][u + 1]],
             datos.hist[("train", "B")][2][datos.hist[("train", "B")][0][u]:datos.hist[("train", "B")][0][u + 1]])
        ref, _, _ = referencia_numpy(Vn, W1, b1, W2, b2, A, B, np.array([], dtype=np.int64))
        ut = vectores_usuarios_eval(modelo, ctx, V, np.array([u]), "train")[0].detach().numpy()
        peor = max(peor, float(np.abs(ut - ref).max()))
    return peor < 1e-4, f"150 usuarios, diferencia máxima {peor:.1e}"


def c_id_de_peliculas_frias(modelo, ctx, datos):
    """Una película sin ratings en train no depende de su embedding de movieId (solo del contenido)."""
    modelo.eval()
    fria = ~torch.as_tensor(datos.feats["en_train"])
    if not bool(fria.any()):
        return True, "no hay películas frías en esta muestra"
    with torch.no_grad():
        antes = modelo.vectores_items().clone()
        w = modelo.items.emb_id.weight.clone()
        modelo.items.emb_id.weight += 5.0
        despues = modelo.vectores_items()
        modelo.items.emb_id.weight.copy_(w)
    dif = (antes[fria] - despues[fria]).abs().max().item()
    return dif < 1e-6, f"{int(fria.sum())} películas frías, cambio máximo al perturbar el ID: {dif:.1e}"


def c_invariancia_orden_historial(modelo, ctx, datos):
    """El vector de usuario no depende del orden de la bolsa (es una media ponderada)."""
    from .modelo import Bolsa
    modelo.eval()
    with torch.no_grad():
        V = modelo.vectores_items()
        u = int(_usuarios_val(datos, 1, 11)[0])
        ip, ix, pw = datos.hist[("train", "A")]
        i, p = ix[ip[u]:ip[u + 1]], pw[ip[u]:ip[u + 1]]
        perm = np.random.default_rng(0).permutation(len(i))
        B = Bolsa(*datos.hist[("train", "B")], device="cpu")
        a = modelo.vectores_usuarios(V, torch.tensor([0]), Bolsa(np.array([0, len(i)]), i, p), Bolsa(np.array([0, 0]), np.array([], dtype="int32"), np.array([], dtype="float32")))
        b = modelo.vectores_usuarios(V, torch.tensor([0]), Bolsa(np.array([0, len(i)]), i[perm], p[perm]), Bolsa(np.array([0, 0]), np.array([], dtype="int32"), np.array([], dtype="float32")))
    dif = (a - b).abs().max().item()
    return dif < 1e-5, f"bolsa de {len(i)} películas permutada: diferencia {dif:.1e}"


def bench_latencia(modelo, ctx, datos) -> dict:
    """Milisegundos: un usuario (torre + scores + top-200) y por usuario en lotes de 512. CPU, un solo proceso de Python."""
    modelo.eval()
    us = _usuarios_val(datos, 512, 13)
    ip, ix = datos.valorados["train"]
    with torch.no_grad():
        V = modelo.vectores_items()
        t0 = time.perf_counter()
        for u in us[:50]:
            s = (vectores_usuarios_eval(modelo, ctx, V, np.array([u]), "train") @ V.T).numpy()
            top_k(s, 200, ip, ix, np.array([u]))
        uno = (time.perf_counter() - t0) / 50 * 1000
        t0 = time.perf_counter()
        s = (vectores_usuarios_eval(modelo, ctx, V, us, "train") @ V.T).numpy()
        top_k(s, 200, ip, ix, us)
        lote = (time.perf_counter() - t0) / len(us) * 1000
        t0 = time.perf_counter()
        modelo.vectores_items()
        items = (time.perf_counter() - t0) * 1000
    return {"ms_por_usuario_solo": round(uno, 2), "ms_por_usuario_en_lote_512": round(lote, 3), "ms_torre_de_items_completa": round(items, 1)}


COMPROBACIONES = [c_finitud_y_normas, c_determinismo, c_exclusion_c14, c_metricas_azar_y_oraculo, c_numpy_igual_torch,
                  c_id_de_peliculas_frias, c_invariancia_orden_historial]


def ejecutar(datos: Datos, modelo: DosTorres, device: str = "cpu") -> dict:
    ctx = Contexto(datos, device)
    out = {"comprobaciones": {}}
    for f in COMPROBACIONES:
        try:
            ok, detalle = f(modelo, ctx, datos)
        except Exception as e:                          # una comprobación rota no tapa a las demás
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        out["comprobaciones"][f.__name__[2:]] = {"ok": bool(ok), "detalle": detalle}
    out["latencia"] = bench_latencia(modelo, ctx, datos)
    out["todo_ok"] = all(v["ok"] for v in out["comprobaciones"].values())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--muestra", default="dev")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--etiqueta", required=True)
    ap.add_argument("--feats", default=None, help="muestra de las features/vocabularios del modelo (por defecto, la misma que --muestra). "
                    "Un checkpoint de `completo` necesita las features de `completo`, aunque se compruebe con usuarios de `dev`")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    datos = cargar(PROCESADOS / a.muestra)
    if a.feats and a.feats != a.muestra:
        datos.feats = cargar_features(PROCESADOS / a.feats)
    modelo = DosTorres.desde_config(datos.feats, cfg)
    modelo.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=True)["modelo"])
    r = ejecutar(datos, modelo)
    r["checkpoint"], r["muestra"] = str(a.checkpoint.name), a.muestra
    (INFORMES / f"comprobaciones_{a.etiqueta}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Comprobaciones `{a.etiqueta}` (muestra `{a.muestra}`)", "", "| Comprobación | Resultado | Detalle |", "|---|:---:|---|"]
    md += [f"| {k} | {'OK' if v['ok'] else 'FALLA'} | {v['detalle']} |" for k, v in r["comprobaciones"].items()]
    md += ["", f"Latencia (CPU): {r['latencia']}"]
    (INFORMES / f"comprobaciones_{a.etiqueta}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    raise SystemExit(0 if r["todo_ok"] else 1)


if __name__ == "__main__":
    main()
