"""Entrenamiento de extremo a extremo en un dataset sintético: corre, mide en val, guarda y reanuda."""
import copy
import json

import numpy as np
import pytest
import torch

from src.datos import cargar
from src.entrenar import Contexto, entrenar, evaluar_particion

from . import sintetico


@pytest.fixture()
def datos(tmp_path):
    return cargar(sintetico.crear(tmp_path))


def cfg_rapida():
    cfg = copy.deepcopy(sintetico.CFG)
    cfg["entrenamiento"].update(batch={"cpu": 16, "cuda": 16}, pasos_checkpoint=2, n_usuarios_val=0)
    cfg["evaluacion"]["ks"] = [5, 10]
    cfg["entrenamiento"]["metrica_val"] = "recall@10/pos"
    return cfg


def test_datos_cargan_con_las_formas_esperadas(datos):
    assert datos.feats["texto"].shape == (30, 384)
    assert datos.feats["generos"].shape == (30, 16) and datos.feats["numerico"].shape == (30, 8)
    assert len(datos.ej_user) == len(datos.ej_item) == len(datos.ej_peso) > 0
    rel, elig = datos.relevantes(1)
    assert set(rel) == set(elig) == {"pos", "r4"}


def test_una_epoca_escribe_checkpoints_log_y_resumen(datos, tmp_path):
    r = entrenar(datos, cfg_rapida(), "cpu", tmp_path / "run", epocas=2)
    assert r["epocas_hechas"] == 2 and np.isfinite(r["mejor"])
    for f in ("ultimo.pt", "mejor.pt", "log.json", "resumen.json"):
        assert (tmp_path / "run" / f).exists()
    log = json.loads((tmp_path / "run" / "log.json").read_text(encoding="utf-8"))
    assert [x["epoca"] for x in log] == [1, 2] and all(np.isfinite(x["perdida"]) for x in log)


def test_reanuda_desde_el_ultimo_checkpoint(datos, tmp_path):
    cfg = cfg_rapida()
    entrenar(datos, cfg, "cpu", tmp_path / "run", epocas=1)
    r = entrenar(datos, cfg, "cpu", tmp_path / "run", epocas=2)      # mismo directorio: continúa en la época 2
    assert r["epocas_hechas"] == 2 and [x["epoca"] for x in r["log"]] == [1, 2]
    ck = torch.load(tmp_path / "run" / "ultimo.pt", weights_only=True)
    assert ck["estado"]["epoca"] == 2 and ck["estado"]["paso"] == 0


def test_la_perdida_baja_al_entrenar(datos, tmp_path):
    cfg = cfg_rapida()
    cfg["entrenamiento"]["paciencia"] = 99
    r = entrenar(datos, cfg, "cpu", tmp_path / "run", epocas=6)
    assert r["log"][-1]["perdida"] < r["log"][0]["perdida"]


def test_evaluar_val_y_test_devuelven_metricas(datos):
    cfg = cfg_rapida()
    ctx = Contexto(datos, "cpu")
    from src.modelo import DosTorres
    m = DosTorres.desde_config(datos.feats, cfg)
    for particion in (1, 2):
        r = evaluar_particion(m, ctx, particion, [5, 10])
        assert 0.0 <= r["recall@10/pos"] <= 1.0 and 0.0 <= r["cobertura@10"] <= 1.0
