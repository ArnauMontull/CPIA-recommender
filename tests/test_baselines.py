"""Baseline de popularidad: devuelve las mismas claves que el modelo y no recomienda lo ya valorado."""
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
