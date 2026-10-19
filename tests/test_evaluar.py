"""evaluar.py de extremo a extremo con el dataset sintético y un checkpoint recién entrenado."""
import copy

from src import evaluar as ev
from src.datos import cargar
from src.entrenar import entrenar

from . import sintetico


def test_evaluar_escribe_el_informe(tmp_path, monkeypatch):
    c = sintetico.crear(tmp_path)
    cfg = copy.deepcopy(sintetico.CFG)
    cfg["entrenamiento"].update(batch={"cpu": 16, "cuda": 16}, n_usuarios_val=0, metrica_val="recall@10/pos")
    cfg["evaluacion"]["ks"] = [5, 10, 20, 30]
    entrenar(cargar(c), cfg, "cpu", tmp_path / "run", epocas=1)
    informes = tmp_path / "informes"
    informes.mkdir()
    monkeypatch.setattr(ev, "INFORMES", informes)
    monkeypatch.setattr(ev, "PROCESADOS", tmp_path / "processed")
    monkeypatch.setattr(ev.yaml, "safe_load", lambda _t: cfg)
    ev.main(["--muestra", "sint", "--checkpoint", str(tmp_path / "run" / "mejor.pt"), "--max-items-ease", "20", "--lam-ease", "10"])
    md = (informes / "resultados_sint.md").read_text(encoding="utf-8")
    assert "## Test: relevante = positivo (C2)" in md and "| Recall@5 |" in md and "EASE" in md and "Two-Tower" in md
