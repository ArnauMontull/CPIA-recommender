"""Importa los resultados de una semilla descargados de Drive (zip con mejor.pt, log.json, resumen.json y resultados_completo_s<N>.*).

Extrae el zip en una carpeta nueva y vacía (nunca en el repo directamente), admite solo los nombres esperados y coloca:
  mejor.pt                      -> modelos/completo_tau0.05_s<N>/mejor.pt   (fuera de git)
  log.json                      -> reports/semillas/s<N>_log.json
  resultados_completo_s<N>.md   -> reports/
  resultados_completo_s<N>.json -> reports/

Uso: python scripts/importar_semilla.py "<ruta del zip>" 43
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def importar(zip_ruta: Path, semilla: int, raiz: Path = RAIZ) -> list[str]:
    esperados = {"mejor.pt", "log.json", f"resultados_completo_s{semilla}.md", f"resultados_completo_s{semilla}.json"}
    hecho = []
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_ruta) as z:
        tmp = Path(tmp)
        for info in z.infolist():
            nombre = Path(info.filename).name                       # sin carpetas: evita rutas con ../
            if info.is_dir() or nombre not in esperados:
                continue
            (tmp / nombre).write_bytes(z.read(info))
        faltan = esperados - {p.name for p in tmp.iterdir()}
        if faltan:
            raise SystemExit(f"Faltan en el zip: {sorted(faltan)}")
        destinos = {
            "mejor.pt": raiz / "modelos" / f"completo_tau0.05_s{semilla}" / "mejor.pt",
            "log.json": raiz / "reports" / "semillas" / f"s{semilla}_log.json",
            f"resultados_completo_s{semilla}.md": raiz / "reports" / f"resultados_completo_s{semilla}.md",
            f"resultados_completo_s{semilla}.json": raiz / "reports" / f"resultados_completo_s{semilla}.json",
        }
        for nombre, destino in destinos.items():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp / nombre, destino)
            hecho.append(str(destino.relative_to(raiz)))
    return hecho


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("zip", type=Path)
    ap.add_argument("semilla", type=int)
    a = ap.parse_args()
    print("\n".join(importar(a.zip, a.semilla)))
    sys.exit(0)
