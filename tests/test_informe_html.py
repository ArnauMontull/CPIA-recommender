"""El informe interactivo se ejecuta sin errores de JavaScript (con un DOM mínimo simulado) y pinta todas las secciones."""
import re
import shutil
import subprocess

import pytest

from src.informe_final import generar

NODE = shutil.which("node")
MD42 = """## Test: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0500 | 0.1000 | 0.1200 |
| Recall@200 | 0.3300 | 0.6000 | 0.6300 |
"""
HARNESS = """
const DATOS = %s;
const els = {};
const mk = id => els[id] || (els[id] = {id, value: '', innerHTML: '', textContent: '', children: [], style: {}, setAttribute(){}, appendChild(c){this.children.push(c)}, onchange:null, oninput:null});
global.document = {getElementById: id => id === 'datos' ? {textContent: JSON.stringify(DATOS)} : mk(id),
  createElementNS: () => ({setAttribute(){}, appendChild(){}, style:{}, set textContent(v){}, get textContent(){return ''}})};
global.innerWidth = 1000;
mk('selPart').value='Test'; mk('selDef').value='positivo (C2)'; mk('selFam').value='Recall'; mk('selSem').value='Recall@200'; mk('selCur').value='recall200'; mk('busca').value='';
"""


@pytest.mark.skipif(NODE is None, reason="sin Node.js")
def test_el_javascript_del_informe_se_ejecuta(tmp_path):
    rep = tmp_path / "reports"
    rep.mkdir()
    (rep / "resultados_completo.md").write_text(MD42, encoding="utf-8")
    from src.informe_final import ensamblar
    generar(ensamblar(rep), tmp_path / "out")
    h = (tmp_path / "out" / "informe_interactivo.html").read_text(encoding="utf-8")
    datos = re.search(r'<script id="datos" type="application/json">(.*?)</script>', h, re.S).group(1)
    js = re.findall(r"<script>(.*?)</script>", h, re.S)[-1]
    f = tmp_path / "t.js"
    f.write_text(HARNESS % datos + js + "\nconsole.log(els.tabM.innerHTML.length, els.glo.innerHTML.length, els.tiles.innerHTML.length);", encoding="utf-8")
    r = subprocess.run([NODE, str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:800]
    tabla, glo, tiles = (int(x) for x in r.stdout.split())
    assert tabla > 100 and glo > 5000 and tiles > 100
