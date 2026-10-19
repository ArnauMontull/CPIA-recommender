"""Generador de informes: ensambla resultados sintéticos y produce LaTeX y HTML coherentes, con glosario completo."""
import json
import re

from src.glosario import GLOSARIO, GRUPOS
from src.informe_final import METRICAS, ensamblar, generar, tex_gl

MD42 = """# Resultados
## Test: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0500 | 0.1000 | 0.1200 |
| Recall@200 | 0.3300 | 0.6000 | 0.6300 |
| NDCG@20 | 0.0600 | 0.1300 | 0.1400 |
"""
MD43 = """## Test: relevante = positivo (C2)

| Métrica | Popularidad | Two-Tower |
|---|---:|---:|
| Recall@10 | 0.0500 | 0.1400 |
| Recall@200 | 0.3300 | 0.6500 |
| NDCG@20 | 0.0600 | 0.1500 |
"""


def _reports(tmp_path):
    r = tmp_path / "reports"
    (r / "semillas").mkdir(parents=True)
    (r / "resultados_completo.md").write_text(MD42, encoding="utf-8")
    (r / "resultados_completo_s43.md").write_text(MD43, encoding="utf-8")
    (r / "semillas" / "s42_log.json").write_text(json.dumps([{"epoca": 1, "perdida": 6.5, "recall@200/pos": 0.6, "segundos": 300}]), encoding="utf-8")
    return r


def test_ensamblar_media_y_desviacion_entre_semillas(tmp_path):
    d = ensamblar(_reports(tmp_path))
    assert d["semillas"] == ["42", "43"]
    v = d["metodos"]["Test: positivo (C2)"]["Recall@200"]
    assert abs(v["Two-Tower"]["media"] - 0.64) < 1e-9 and abs(v["Two-Tower"]["sd"] - 0.0141421) < 1e-6
    assert v["EASE"] == 0.6 and v["Popularidad"] == 0.33        # de la semilla de referencia
    assert d["curvas"]["42"][0]["recall200"] == 0.6


def test_generar_tex_y_html(tmp_path):
    out = tmp_path / "salida"
    res = generar(ensamblar(_reports(tmp_path)), out, compilar=False)
    t = (out / "generado.tex").read_text(encoding="utf-8")
    assert "\section{Glosario}" in t and "\section{Resultados principales}" in t and "0.6400" in t
    h = (out / "informe_interactivo.html").read_text(encoding="utf-8")
    datos = json.loads(re.search(r'<script id="datos" type="application/json">(.*?)</script>', h, re.S).group(1))
    assert len(datos["glosario"]) == len(GLOSARIO) and datos["semillas"] == ["42", "43"]
    assert "@@" not in h and "@@" not in (out / "informe_resultados.tex").read_text(encoding="utf-8")


def test_el_glosario_define_cada_metrica_y_grupo():
    texto = " ".join(t + " " + d for _, t, d in GLOSARIO).lower()
    familias = {m.split("@")[0].lower() for m in METRICAS}
    assert familias <= {w for w in re.findall(r"[a-záéíóú]+", texto)}          # recall, ndcg, cobertura, novedad
    for termino in ("train", "val", "test", "bolsa", "elegible", "τ", "ease", "popularidad", "bootstrap", "lift", "mmr", "logq",
                    "checkpoint", "semilla", "máscara", "película fría", "usuario frío", "watchlist", "latencia", "fuga"):
        assert termino in texto, termino
    assert {g for g, _, _ in GLOSARIO} == set(GRUPOS)
    assert len({t for _, t, _ in GLOSARIO}) == len(GLOSARIO)                    # sin términos repetidos


def test_tex_gl_traduce_simbolos():
    s = tex_gl("τ ≥ 0,05 y 10⁻³ en train_val")
    assert r"$\tau$" in s and r"$\geq$" in s and "$10^{-3}$" in s and r"train\_val" in s
