"""Entrenamiento del Two-Tower (plan_entrenamiento.md, tarea 3).

Un paso = un lote de ejemplos (un positivo de train cada uno, F3): torre de ítem sobre todo el catálogo -> usuarios
(historial de train sin el objetivo) -> softmax con los negativos del lote, logQ, máscara C14 y peso C4 x C9.
Al final de cada época evalúa en val (catálogo completo, máscara C14, historial = train) y aplica early stopping.
Colab se corta: `ultimo.pt` se escribe cada `pasos_checkpoint` pasos y al acabar cada época, y al relanzar el mismo
comando se reanuda solo. El orden de los lotes de la época e sale de (semilla, e): D10.

Uso: python -m src.entrenar --muestra dev --device cpu --salida modelos/
     python -m src.entrenar --muestra completo --device cuda --salida /content/drive/MyDrive/CPIA/modelos
Salida: <salida>/<muestra>_tau<tau>_s<semilla>/{ultimo.pt, mejor.pt, log.json, resumen.json}.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .datos import Datos, cargar
from .evaluacion.metricas import evaluar_por_lotes
from .modelo import Bolsa, DosTorres
from .perdida import Valorados, mascara_lote, perdida_softmax_logq
from .preprocesamiento.comun import PROCESADOS, RAIZ, marcar

TAREA = "CPIA entrenar"


class Contexto:
    """Todo lo que el bucle necesita, ya en el dispositivo: ejemplos, historiales por vista, máscaras y logQ."""

    def __init__(self, datos: Datos, device: str):
        self.datos, self.device = datos, device
        t = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=device)
        self.ej_user, self.ej_item, self.ej_peso = t(datos.ej_user, torch.int64), t(datos.ej_item, torch.int64), t(datos.ej_peso)
        self.logq = t(datos.logq)
        self.bolsas = {v: (Bolsa(*datos.hist[(v, "A")], device=device), Bolsa(*datos.hist[(v, "B")], device=device))
                       for v in ("train", "train_val")}
        self.valorados = {v: Valorados(*datos.valorados[v], datos.n_items, device=device) for v in ("train", "train_val")}


def vectores_usuarios_eval(modelo: DosTorres, ctx: Contexto, V, filas: np.ndarray, vista: str) -> torch.Tensor:
    A, B = ctx.bolsas[vista]
    return modelo.vectores_usuarios(V, torch.as_tensor(filas, device=ctx.device), A, B)


@torch.no_grad()
def evaluar_particion(modelo: DosTorres, ctx: Contexto, particion: int, ks, usuarios_max: int | None = None,
                      semilla: int = 0, tam_lote: int = 1024) -> dict[str, float]:
    """Métricas en val (1: historial = train) o test (2: historial = train + val, F8). Catálogo completo, sin
    lo valorado (C14). `usuarios_max`: submuestra fija de los elegibles (la misma en todas las épocas)."""
    d = ctx.datos
    vista = "train" if particion == 1 else "train_val"
    rel, elig = d.relevantes(particion)
    usuarios = np.unique(np.concatenate(list(elig.values())))
    if usuarios_max and len(usuarios) > usuarios_max:
        usuarios = np.sort(np.random.default_rng(semilla).choice(usuarios, usuarios_max, replace=False))
        elig = {k: np.intersect1d(v, usuarios) for k, v in elig.items()}
    modelo.eval()
    V = modelo.vectores_items()
    score = lambda filas: (vectores_usuarios_eval(modelo, ctx, V, filas, vista) @ V.T).cpu().numpy()
    return evaluar_por_lotes(score, usuarios, rel, elig, ks, d.pop_train, d.n_items, *d.valorados[vista], tam_lote=tam_lote)


def _escribir(ruta: Path, objeto) -> None:
    """torch.save atómico: si Colab muere a medias no queda un .pt roto."""
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    torch.save(objeto, tmp)
    os.replace(tmp, ruta)


def entrenar(datos: Datos, cfg: dict, device: str, salida: Path, semilla: int | None = None, tau: float | None = None,
             batch: int | None = None, epocas: int | None = None, pasos_por_epoca: int = 0, ctx: Contexto | None = None) -> dict:
    """Entrena (o reanuda) y devuelve el resumen. `salida` es la carpeta de esta ejecución."""
    e = cfg["entrenamiento"]
    semilla = cfg["semilla"] if semilla is None else semilla
    tau = e["tau"] if tau is None else tau
    batch = batch or e["batch"]["cuda" if str(device).startswith("cuda") else "cpu"]
    epocas = epocas or e["epocas"]
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(semilla)
    ctx = ctx or Contexto(datos, device)
    modelo = DosTorres.desde_config(datos.feats, cfg).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=e["lr"], weight_decay=e["weight_decay"])
    estado = {"epoca": 0, "paso": 0, "paso_global": 0, "mejor": -1.0, "malas": 0, "log": [], "segundos": 0.0}

    ultimo = salida / "ultimo.pt"
    if ultimo.exists():                                           # reanudar (Colab se corta)
        ck = torch.load(ultimo, map_location=device, weights_only=True)
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["opt"])
        estado = ck["estado"]
        print(f"Reanudo desde época {estado['epoca']}, paso {estado['paso']} (paso global {estado['paso_global']})")

    n_ej = len(ctx.ej_user)
    n_pasos = n_ej // batch
    if pasos_por_epoca:
        n_pasos = min(n_pasos, pasos_por_epoca)
    total = epocas * n_pasos
    A, B = ctx.bolsas["train"]
    t_ini = time.time() - estado["segundos"]

    def guardar(nombre: str = "ultimo.pt") -> None:
        _escribir(salida / nombre, {"modelo": modelo.state_dict(), "opt": opt.state_dict(), "estado": estado,
                                    "cfg": {"tau": tau, "batch": batch, "semilla": semilla}})

    while estado["epoca"] < epocas and estado["malas"] < e["paciencia"]:
        ep = estado["epoca"]
        orden = torch.randperm(n_ej, generator=torch.Generator().manual_seed(semilla * 1000 + ep)).to(ctx.device)
        modelo.train()
        suma_perdida, n_perdida = 0.0, 0
        while estado["paso"] < n_pasos:
            idx = orden[estado["paso"] * batch:(estado["paso"] + 1) * batch]
            u, t, w = ctx.ej_user[idx], ctx.ej_item[idx], ctx.ej_peso[idx]
            V = modelo.vectores_items()
            U = modelo.vectores_usuarios(V, u, A, B, excluir_item=t, excluir_peso=w)
            perdida = perdida_softmax_logq(U, V[t], ctx.logq[t], tau, w, mascara_lote(u, t, ctx.valorados["train"]))
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), e["clip_grad"])
            opt.step()
            estado["paso"] += 1
            estado["paso_global"] += 1
            suma_perdida += float(perdida.detach())
            n_perdida += 1
            if estado["paso"] % 50 == 0:
                marcar(TAREA, estado["paso_global"], total, f"época {ep + 1}/{epocas} pérdida {suma_perdida / n_perdida:.4f}")
            if estado["paso"] % e["pasos_checkpoint"] == 0:
                estado["segundos"] = time.time() - t_ini
                guardar()

        m = evaluar_particion(modelo, ctx, 1, cfg["evaluacion"]["ks"], e["n_usuarios_val"], semilla)
        estado["epoca"], estado["paso"] = ep + 1, 0
        valor = m[e["metrica_val"]]
        mejora = valor > estado["mejor"]
        estado["mejor"] = max(valor, estado["mejor"])
        estado["malas"] = 0 if mejora else estado["malas"] + 1
        estado["segundos"] = time.time() - t_ini
        estado["log"].append({"epoca": ep + 1, "perdida": suma_perdida / max(n_perdida, 1), "segundos": round(estado["segundos"], 1),
                              **{k: round(v, 5) for k, v in m.items() if k.split("@")[0] in ("recall", "ndcg", "cobertura")}})
        print(f"época {ep + 1}/{epocas}  pérdida {estado['log'][-1]['perdida']:.4f}  {e['metrica_val']} {valor:.4f}"
              f"{'  (mejor)' if mejora else ''}  [{estado['segundos'] / 60:.1f} min]")
        guardar()
        if mejora:
            guardar("mejor.pt")
        (salida / "log.json").write_text(json.dumps(estado["log"], indent=1, ensure_ascii=False), encoding="utf-8")

    resumen = {"muestra": datos.carpeta.name, "tau": tau, "batch": batch, "semilla": semilla, "epocas_hechas": estado["epoca"],
               "metrica_val": e["metrica_val"], "mejor": estado["mejor"], "parada_temprana": estado["malas"] >= e["paciencia"],
               "segundos": round(estado["segundos"], 1), "log": estado["log"]}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    marcar(TAREA, total, total, "hecho")
    return resumen


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--salida", type=Path, default=RAIZ / "modelos")
    ap.add_argument("--semilla", type=int)
    ap.add_argument("--tau", type=float, help="temperatura (T1: barrido 0.02, 0.05, 0.1)")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--pasos-por-epoca", type=int, default=0, help="recorta la época (pruebas rápidas)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    semilla = cfg["semilla"] if a.semilla is None else a.semilla
    tau = cfg["entrenamiento"]["tau"] if a.tau is None else a.tau
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda, pero no hay GPU (en Colab: Entorno de ejecución > T4)")
    datos = cargar(PROCESADOS / a.muestra)
    carpeta = a.salida / f"{a.muestra}_tau{tau:g}_s{semilla}"
    r = entrenar(datos, cfg, a.device, carpeta, semilla, tau, a.batch, a.epocas, a.pasos_por_epoca)
    print(f"Listo: mejor {r['metrica_val']} = {r['mejor']:.4f} en {r['epocas_hechas']} épocas -> {carpeta}")


if __name__ == "__main__":
    main()
