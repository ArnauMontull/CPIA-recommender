"""Bootstrap de usuarios: IC de una diferencia a mano y ejecución completa sobre datos sintéticos."""
import numpy as np
import torch

from src.baselines import EASE
from src.bootstrap import ejecutar, ic_diferencia
from src.datos import cargar
from src.entrenar import Contexto
from src.modelo import DosTorres

from . import sintetico


def test_ic_de_una_diferencia_constante_es_un_punto():
    a, b = np.full(500, 0.7), np.full(500, 0.5)
    d, lo, hi = ic_diferencia(a, b, n=200)
    assert abs(d - 0.2) < 1e-9 and abs(lo - 0.2) < 1e-9 and abs(hi - 0.2) < 1e-9


def test_ic_excluye_el_cero_si_hay_diferencia_clara_y_lo_incluye_si_no():
    rng = np.random.default_rng(0)
    b = rng.normal(0.5, 0.2, 2000)
    d, lo, hi = ic_diferencia(b + 0.1, b + rng.normal(0, 0.05, 2000), n=500)       # +0,1 de media
    assert lo > 0
    d, lo, hi = ic_diferencia(b, b + rng.normal(0, 0.3, 2000), n=500)              # diferencias de media 0
    assert lo < 0 < hi


def test_ejecutar_en_sintetico(tmp_path):
    torch.manual_seed(0)
    datos = cargar(sintetico.crear(tmp_path, n_usuarios=80, n_items=40, por_usuario=24))
    m = DosTorres.desde_config(datos.feats, sintetico.CFG)
    r = ejecutar(datos, m, Contexto(datos, "cpu"), EASE(datos, 10.0, 30), None, n=100)
    assert r["n_usuarios"] > 10 and "Two-Tower − EASE | recall@20" in r["diferencias"]
    v = r["diferencias"]["Two-Tower − Popularidad | ndcg@200"]
    assert v["ic_inf"] <= v["dif"] <= v["ic_sup"]
