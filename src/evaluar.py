"""Evaluación final (plan_entrenamiento.md, tarea 4): test con historial train + val (F8) y máscara train + val (C14),
con las dos definiciones de relevante (C7: `pos` y `r4`), frente a los baselines. Escribe reports/resultados_<muestra>.md.
Pendiente: dev_frio (G6) y la demo con Letterboxd.

Uso: python -m src.evaluar --muestra dev --checkpoint modelos/dev_tau0.05_s42/mejor.pt [--sin-ease] [--max-items-ease 10000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .baselines import EASE, evaluar_ease, evaluar_popularidad
from .datos import cargar
from .entrenar import Contexto, evaluar_particion
from .modelo import DosTorres
from .preprocesamiento.comun import INFORMES, PROCESADOS, RAIZ

NOMBRES = {"pop": "Popularidad", "ease": "EASE", "modelo": "Two-Tower"}


def tabla(res: dict[str, dict], ks, definicion: str) -> str:
    cols = list(res)
    fila = lambda etiqueta, clave: f"| {etiqueta} | " + " | ".join(f"{res[c][clave]:.4f}" for c in cols) + " |"
    l = ["| Métrica | " + " | ".join(NOMBRES[c] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for k in ks:
        l.append(fila(f"Recall@{k}", f"recall@{k}/{definicion}"))
    for k in ks:
        l.append(fila(f"NDCG@{k}", f"ndcg@{k}/{definicion}"))
    l.append(fila(f"Cobertura@{ks[1]}", f"cobertura@{ks[1]}"))
    l.append(fila(f"Novedad@{ks[1]}", f"novedad@{ks[1]}"))
    return "\n".join(l)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--muestra", required=True, help="dev o completo (carpeta de data/processed)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sin-ease", action="store_true")
    ap.add_argument("--max-items-ease", type=int, default=10000)
    ap.add_argument("--lam-ease", type=float, default=500.0)
    ap.add_argument("--etiqueta", default="", help="sufijo de los informes (p. ej. _s43), para no sobrescribir")
    ap.add_argument("--usuarios-max", type=int, default=0, help="submuestra de usuarios (0 = todos los elegibles)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    ks, umax, semilla = cfg["evaluacion"]["ks"], a.usuarios_max or None, cfg["semilla"]
    datos = cargar(PROCESADOS / a.muestra)
    modelo = DosTorres.desde_config(datos.feats, cfg).to(a.device)
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    modelo.load_state_dict(ck["modelo"])
    ctx = Contexto(datos, a.device)
    ease = None if a.sin_ease else EASE(datos, a.lam_ease, a.max_items_ease)

    salida = {}
    for nombre, particion in (("val", 1), ("test", 2)):
        res = {"pop": evaluar_popularidad(datos, particion, ks, umax, semilla),
               "modelo": evaluar_particion(modelo, ctx, particion, ks, umax, semilla)}
        if ease is not None:
            res["ease"] = evaluar_ease(datos, particion, ks, umax, semilla, modelo=ease)
        salida[nombre] = dict(sorted(res.items(), key=lambda kv: ["pop", "ease", "modelo"].index(kv[0])))

    md = [f"# Resultados en `{a.muestra}`", "",
          f"Checkpoint `{a.checkpoint.name}` (época {ck['estado']['epoca']}, τ = {ck['cfg']['tau']}, semilla {ck['cfg']['semilla']}). "
          "Catálogo completo, sin lo valorado (C14). Test: historial = train + val (F8). "
          + ("EASE" + (f" sobre las {a.max_items_ease} películas más populares, λ = {a.lam_ease:g}" if ease else " omitido") + ".")]
    for nombre in ("val", "test"):
        for d, titulo in (("pos", "positivo (C2)"), ("r4", "rating ≥ 4 (C7)")):
            md += ["", f"## {nombre.capitalize()}: relevante = {titulo}", "", tabla(salida[nombre], ks, d)]
    ruta = INFORMES / f"resultados_{a.muestra}{a.etiqueta}.md"
    ruta.write_text("\n".join(md) + "\n", encoding="utf-8")
    (INFORMES / f"resultados_{a.muestra}{a.etiqueta}.json").write_text(json.dumps(salida, indent=1), encoding="utf-8")
    print(f"Escrito {ruta}")


if __name__ == "__main__":
    main()
