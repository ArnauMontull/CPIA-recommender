"""Métricas de watchlist: aciertos, precisión, recall y lift sobre el azar, calculados a mano."""
import numpy as np
import pytest

from src.evaluar_watchlist import metricas_watchlist


def test_a_mano():
    # 10 películas, scores 9..0 (la 0 es la mejor); la 3 y la 8 ya vistas (excluidas); watchlist elegible = {1, 5, 9}
    s = np.arange(10, 0, -1, dtype=float)
    m = metricas_watchlist(s, excluidos=np.array([3, 8]), watchlist=np.array([1, 5, 9]), ks=(2, 5, 8))
    assert m["candidatos"] == 8 and m["azar"] == pytest.approx(3 / 8)
    # ranking sin excluidas: 0,1,2,4,5,6,7,9 -> en watchlist: posiciones 2, 5, 8
    assert (m["hits@2"], m["hits@5"], m["hits@8"]) == (1, 2, 3)
    assert m["precision@5"] == pytest.approx(0.4) and m["recall@5"] == pytest.approx(2 / 3)
    assert m["lift@5"] == pytest.approx(0.4 / (3 / 8))


def test_los_excluidos_nunca_acierta():
    s = np.arange(5, 0, -1, dtype=float)
    m = metricas_watchlist(s, excluidos=np.array([0]), watchlist=np.array([0, 1]), ks=(3,))   # la 0 no puede contar
    assert m["hits@3"] == 1
