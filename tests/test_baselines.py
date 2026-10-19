"""Baseline de popularidad: devuelve las mismas claves que el modelo y no recomienda lo ya valorado."""
import numpy as np
import pytest

from src.baselines import evaluar_popularidad
from src.datos import cargar

from . import sintetico


@pytest.fixture()
def datos(tmp_path):
    return cargar(sintetico.crear(tmp_path))


def test_popularidad_val_y_test(datos):
    for particion in (1, 2):
        r = evaluar_popularidad(datos, particion, [5, 10])
        assert 0.0 <= r["recall@10/pos"] <= 1.0 and r["recall@10/pos"] >= r["recall@5/pos"]
        assert 0.0 < r["cobertura@5"] <= 1.0


def test_popularidad_igual_para_todos_salvo_la_mascara(datos):
    # con k = catálogo completo se recupera todo lo no valorado: recall = 1 (todos los relevantes de val no están en train)
    r = evaluar_popularidad(datos, 1, [datos.n_items])
    assert r[f"recall@{datos.n_items}/pos"] == pytest.approx(1.0)


def test_ease_devuelve_metricas_y_no_recomienda_lo_valorado(datos):
    from src.baselines import EASE, evaluar_ease
    m = EASE(datos, lam=10.0, max_items=20)
    assert m.B.shape == (20, 20) and (m.B.diagonal() == 0).all()
    s = m.scores(np.array([0, 1]), "train")
    assert s.shape == (2, datos.n_items) and np.isinf(s[:, np.setdiff1d(np.arange(datos.n_items), m.items)]).all()
    for particion in (1, 2):
        r = evaluar_ease(datos, particion, [5, 10], modelo=m)
        assert 0.0 <= r["recall@10/pos"] <= 1.0
