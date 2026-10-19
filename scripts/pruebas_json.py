"""Ejecuta toda la suite de pytest y guarda los recuentos en reports/pruebas.json (los lee src/informe_final.py).

Uso: python scripts/pruebas_json.py
Sale con 0 si no hay fallos; si falla algo, también escribe qué pruebas.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main() -> int:
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "tests", "-p", "no:warnings", "-rfE"],
                       cwd=RAIZ, capture_output=True, text=True)
    resumen = (r.stdout.strip().splitlines() or [""])[-1]
    n = lambda palabra: int(m.group(1)) if (m := re.search(rf"(\d+) {palabra}", resumen)) else 0
    datos = {"pasados": n("passed"), "saltados": n("skipped"), "fallos": n("failed") + n("error"), "resumen": resumen,
             "fallidas": re.findall(r"^(?:FAILED|ERROR) (\S+)", r.stdout, re.M)}
    (RAIZ / "reports" / "pruebas.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(datos, ensure_ascii=False))
    return 0 if datos["fallos"] == 0 and r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
