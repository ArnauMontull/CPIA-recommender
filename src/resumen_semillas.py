"""Media ± desviación del Two-Tower entre semillas (D10), a partir de los informes `resultados_<muestra>[_s<N>].md`.

Cada informe tiene una tabla por (partición, definición de relevante); la columna del Two-Tower es la última. EASE y popularidad se
ajustan una sola vez (no dependen de la semilla): se toman del informe que los trae (el de la semilla de referencia).

Uso: python -m src.resumen_semillas reports/resultados_completo.md reports/resultados_completo_s43.md ... [--salida reports/semillas_completo.md]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SECCION = re.compile(r"^## (Val|Test): relevante = (.+)$")


def leer_tablas(texto: str) -> dict[tuple[str, str], dict[str, list[float]]]:
    """{(seccion, metrica): [valores por columna]} con la sección = «Test: relevante = positivo (C2)», etc."""
    out: dict[tuple[str, str], list[float]] = {}
    seccion = None
    for linea in texto.splitlines():
        m = SECCION.match(linea)
        if m:
            seccion = f"{m.group(1)}: {m.group(2)}"
        elif seccion and linea.startswith("| ") and not linea.startswith("| Métrica") and not linea.startswith("|---"):
            c = [x.strip() for x in linea.strip().strip("|").split("|")]
            out[(seccion, c[0])] = [float(x) for x in c[1:]]
    return out


def resumir(informes: list[Path], seccion: str = "Test: positivo (C2)") -> str:
    tablas = [leer_tablas(Path(p).read_text(encoding="utf-8")) for p in informes]
    claves = [k for k in tablas[0] if k[0] == seccion]
    ref = tablas[0]
    con_ease = len(next(iter(ref.values()))) == 3
    lineas = [f"# Two-Tower entre {len(informes)} semillas ({seccion})", "",
              "| Métrica | Popularidad | EASE | Two-Tower (media ± desv.) | Mín | Máx |" if con_ease
              else "| Métrica | Popularidad | Two-Tower (media ± desv.) | Mín | Máx |",
              "|---|---:|---:|---:|---:|---:|" if con_ease else "|---|---:|---:|---:|---:|"]
    for k in claves:
        v = np.array([t[k][-1] for t in tablas])
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        base = f"{ref[k][0]:.4f} | {ref[k][1]:.4f} | " if con_ease else f"{ref[k][0]:.4f} | "
        lineas.append(f"| {k[1]} | {base}{v.mean():.4f} ± {sd:.4f} | {v.min():.4f} | {v.max():.4f} |")
    return "\n".join(lineas) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("informes", nargs="+", type=Path)
    ap.add_argument("--seccion", default="Test: positivo (C2)")
    ap.add_argument("--salida", type=Path)
    a = ap.parse_args(argv)
    md = resumir(a.informes, a.seccion)
    if a.salida:
        a.salida.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
