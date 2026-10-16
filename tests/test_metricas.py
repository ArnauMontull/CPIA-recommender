"""Tests de src/evaluacion/metricas.py con casos hechos a mano (3 usuarios, 6 películas)."""
import numpy as np
import pytest

from src.evaluacion import metricas as m

N = 6
# Scores (filas = usuarios 0, 1, 2)
SC = np.array([
    [0.9, 0.8, 0.7, 0.1, 0.2, 0.3],   # orden: 0,1,2,5,4,3
    [0.1, 0.9, 0.8, 0.7, 0.6, 0.5],   # orden: 1,2,3,4,5,0
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # todo empatado
], dtype=np.float32)
# train (a excluir): u0 ya vio la 0; u1 y u2 nada
EXC = m.csr_desde_pares([0], [0], 3)
# relevantes (definición «train»): u0 {1, 3}; u1 {2}; u2 {} (no elegible)
REL = m.csr_desde_pares([0, 0, 1], [1, 3, 2], 3)


def test_csr_desde_pares():
    ip, ix = m.csr_desde_pares([2, 0, 0, 0], [5, 3, 1, 1], 4)
    assert ip.tolist() == [0, 2, 2, 3, 3]
    assert ix.tolist() == [1, 3, 5]


def test_top_k_excluye_lo_valorado():
    tk = m.top_k(SC, 3, *EXC)
    assert tk[0].tolist() == [1, 2, 5]          # la 0 (score 0,9) está excluida
    assert tk[1].tolist() == [1, 2, 3]
    assert m.top_k(SC, 6, *EXC)[0][-1] == 0     # la excluida queda la última


def test_top_k_con_filas_globales():
    # El usuario 1 solo en el lote: su fila no debe recibir la máscara del usuario 0
    assert m.top_k(SC[[1]], 2, *EXC, filas=[1])[0].tolist() == [1, 2]
    # El usuario 0 en la fila 0: sí
    assert m.top_k(SC[[0]], 2, *EXC, filas=[0])[0].tolist() == [1, 2]
    # Fila 0 del lote = usuario 0 pero pasando el score del usuario 1: se excluye la 0 de u0
    sc = np.array([[0.9, 0.1, 0.2, 0.3, 0.4, 0.5]])
    assert m.top_k(sc, 2, *EXC, filas=[0])[0].tolist() == [5, 4]
    assert m.top_k(sc, 2, *EXC, filas=[1])[0].tolist() == [0, 5]


def test_top_k_empates_menor_indice():
    assert m.top_k(SC[[2]], 3, *EXC, filas=[2])[0].tolist() == [0, 1, 2]


def test_ndcg_un_relevante_en_posicion_2():
    tk = np.array([[4, 2, 0]])  # el relevante (2) está en la posición 2 (índice 1)
    ip, ix = m.csr_desde_pares([0], [2], 1)
    assert m.ndcg_at_k(tk, ip, ix, 3)[0] == pytest.approx(1 / np.log2(3))


def test_recall_y_ndcg_con_exclusion():
    tk = m.top_k(SC, 3, *EXC)
    r = m.recall_at_k(tk, *REL, 3)
    # u0: top = [1,2,5], relevantes {1,3} -> 1/2; u1: top = [1,2,3], relevante {2} -> 1; u2: NaN
    assert r[0] == pytest.approx(0.5) and r[1] == pytest.approx(1.0) and np.isnan(r[2])
    n = m.ndcg_at_k(tk, *REL, 3)
    assert n[0] == pytest.approx(1.0 / (1 + 1 / np.log2(3)))   # DCG 1, IDCG 1 + 1/log2(3)
    assert n[1] == pytest.approx(1 / np.log2(3))               # relevante en la posición 2
    # Sin la exclusión, la 0 ocuparía el primer puesto
    sin = m.top_k(SC, 3, *m.csr_desde_pares([], [], 3))
    assert sin[0].tolist() == [0, 1, 2]


def test_techo_recall():
    # |R| = 2 y 1 (u2 se ignora): k=1 -> media(1/2, 1) = 0,75; k=5 -> 1
    assert m.techo_recall(*REL, 1, [0, 1, 2]) == pytest.approx(0.75)
    assert m.techo_recall(*REL, 5, [0, 1, 2]) == pytest.approx(1.0)


def test_cobertura_y_novedad():
    tk = np.array([[1, 2], [2, 3]])
    assert m.cobertura(tk, N) == pytest.approx(3 / 6)
    pop = np.array([0, 3, 1, 0, 0, 0])   # +1 -> [1,4,2,1,1,1], suma 10
    esperado = np.mean([-np.log2(0.4), -np.log2(0.2), -np.log2(0.2), -np.log2(0.1)])
    assert m.novedad(tk, pop) == pytest.approx(esperado)


def _args():
    rel = {"train": REL, "r4": m.csr_desde_pares([0, 1], [3, 2], 3)}
    elig = {"train": np.array([0, 1]), "r4": np.array([0, 1, 2])}
    return rel, elig, np.array([0, 3, 1, 0, 0, 0])


def test_evaluar_solo_elegibles():
    rel, elig, pop = _args()
    tk = m.top_k(SC, 3, *EXC)
    res = m.evaluar(tk, rel, elig, [3], pop, N)
    # train: u2 no elegible -> media de u0 y u1
    assert res["recall@3/train"] == pytest.approx((0.5 + 1.0) / 2)
    assert res["ndcg@3/train"] == pytest.approx((1 / (1 + 1 / np.log2(3)) + 1 / np.log2(3)) / 2)
    # r4: u0 {3}: top [1,2,5] -> 0; u1 {2} -> 1; u2 elegible pero sin relevantes: NaN, se ignora
    assert res["recall@3/r4"] == pytest.approx((0 + 1.0) / 2)
    assert res["cobertura@3"] == pytest.approx(len({1, 2, 5, 3, 0}) / N)
    assert "novedad@3" in res and "techo_recall@3/train" in res


def test_evaluar_por_lotes_igual_que_evaluar():
    rel, elig, pop = _args()
    ks = [2, 3]
    tk = m.top_k(SC, 3, *EXC)
    todo = m.evaluar(tk, rel, elig, ks, pop, N)
    lotes = m.evaluar_por_lotes(lambda f: SC[f], [0, 1, 2], rel, elig, ks, pop, N, *EXC, tam_lote=2)
    assert todo.keys() == lotes.keys()
    for c in todo:
        assert lotes[c] == pytest.approx(todo[c]), c
