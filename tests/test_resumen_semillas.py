"""Agregación entre semillas a partir de informes markdown."""
import pytest

from src.resumen_semillas import leer_tablas, resumir

CON_EASE = """# Resultados
## Test: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0500 | 0.1000 | 0.1200 |
| NDCG@20 | 0.0600 | 0.1300 | 0.1400 |

## Val: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0000 | 0.0000 | 0.9999 |
"""
SIN_EASE = """## Test: relevante = positivo (C2)

| Métrica | Popularidad | Two-Tower |
|---|---:|---:|
| Recall@10 | 0.0500 | 0.1400 |
| NDCG@20 | 0.0600 | 0.1500 |
"""


def test_leer_tablas_separa_secciones():
    t = leer_tablas(CON_EASE)
    assert t[("Test: positivo (C2)", "Recall@10")] == [0.05, 0.1, 0.12]
    assert t[("Val: positivo (C2)", "Recall@10")][-1] == 0.9999


def test_media_y_desviacion_entre_semillas(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text(CON_EASE, encoding="utf-8")
    b.write_text(SIN_EASE, encoding="utf-8")
    md = resumir([a, b])
    # Two-Tower: 0.12 y 0.14 -> media 0.1300, desviación muestral 0.0141
    assert "| Recall@10 | 0.0500 | 0.1000 | 0.1300 ± 0.0141 | 0.1200 | 0.1400 |" in md
    assert "Val" not in md.split("\n")[0] and "0.9999" not in md
