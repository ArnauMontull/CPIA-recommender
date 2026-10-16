"""Tests del experimento de arranque en frío (G6) con datos sintéticos hechos a mano."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocesamiento import frio as F

CFG = {
    "split": {"frac_val": 0.10, "frac_test": 0.10, "min_por_particion": 1},
    "senal": {"tope_umbral": 4.0, "margen_nota_baja": 1.0, "peso_primer_dia": 0.75, "segundos_primer_dia": 86400,
              "umbral_fijo_eval": 4.0, "suavizado_logq": 1.0},
    "historial": {"peso_neutra": 0.25},
    "letterboxd": {"min_notas_sin_encoger": 20, "lambda_encogimiento": 10},
    "frio": {"frac_usuarios": 0.10, "frac_historia": 0.80},
}
N_ITEMS = 12
RATING_FRIO = [5, 4, 3, 2, 1, 5, 4, 3, 2, 1]


def datos():
    """Usuarios 1-3 normales: 12 ratings de 3.0 (items 0..11, ts 1..12) -> 11 train (el test pasa a train) + 1 val (item 10).
    Usuario 0 frío: 10 ratings (items 0..9, ts 100+i) -> 8 dados + 2 test. Usuario 4 frío con 4 ratings -> fuera."""
    filas = [(0, i, RATING_FRIO[i], 100 + i) for i in range(10)]
    filas += [(u, i, 3.0, 1 + i) for u in (1, 2, 3) for i in range(12)]
    filas += [(4, i, 4.0, 50 + i) for i in range(4)]
    r = pd.DataFrame(filas, columns=["user_idx", "item_idx", "rating", "timestamp"])
    return r.astype({"rating": "float32", "user_idx": "int32", "item_idx": "int32"})


FRIOS = np.array([True, False, False, False, True])


@pytest.fixture(scope="module")
def res():
    return F.construir_frio(datos(), FRIOS, CFG, 5, N_ITEMS)


def test_ningun_rating_de_un_frio_en_train(res):
    s = res["senal"]
    assert set(s[s.user_idx == 0].particion) == {2, 3}
    assert not ((s.particion == 0) & FRIOS[s.user_idx.to_numpy()]).any()
    assert res["meta"]["filas_train"] == 33          # 3 normales x 11 (su test pasa a train)
    # los fríos no aportan nada a la máscara C14 de train
    assert res["indptr"][1] - res["indptr"][0] == 0


def test_historia_dada_antes_que_el_test(res):
    g = res["senal"][res["senal"].user_idx == 0]
    assert (g.particion == 3).sum() == 8 and (g.particion == 2).sum() == 2
    assert g[g.particion == 3].timestamp.max() < g[g.particion == 2].timestamp.min()
    assert g[g.particion == 2].item_idx.tolist() == [8, 9]


def test_media_u_de_un_frio_es_la_de_su_historia_encogida(res):
    # historia dada = ratings 5,4,3,2,1,5,4,3 -> suma 27, n = 8 < 20; mu = 3.0 (todo train es 3.0)
    esperado = (27 + 10 * 3.0) / (8 + 10)
    assert res["meta"]["mu_global_train"] == pytest.approx(3.0)
    assert res["usuarios"].media_u[0] == pytest.approx(esperado)
    # las etiquetas de sus filas usan esa media (3.1667): 4 y 5 positivos; 3 no; 2 y 1 notas bajas (<= 2.1667)
    g = res["senal"][res["senal"].user_idx == 0].set_index("item_idx")
    assert g.positivo[[0, 1, 5, 6]].all() and not g.positivo[[2, 7]].any()
    assert g.nota_baja[[3, 4, 8, 9]].all() and not g.nota_baja[[2, 7]].any()
    # normales: media cruda de su train (3.0)
    assert res["usuarios"].media_u[1] == pytest.approx(3.0)


def test_logq_sin_los_frios(res):
    # positivos de train: los 3 normales (3.0 >= min(4, 3.0)) en items 0..9 y 11 -> 3 cada uno; el item 10 (val) 0
    # total 33: P(i) = (3 + 1) / (33 + 12), P(10) = 1 / 45. Con los fríos, los items 0..7 tendrían más.
    esperado = np.full(N_ITEMS, np.log(4 / 45))
    esperado[10] = np.log(1 / 45)
    np.testing.assert_allclose(res["logq"], esperado, rtol=1e-6)
    assert np.exp(res["logq"]).sum() == pytest.approx(1.0, rel=1e-5)


def test_eleccion_de_frios_estable_y_por_hash():
    ids = np.arange(1000, 1200)
    a = F.elegir_frios(ids, 0.10, 42)
    b = F.elegir_frios(ids, 0.10, 42)
    assert (a == b).all() and a.sum() == 20
    # no depende del orden de entrada: el mismo userId es frío o no
    perm = np.random.default_rng(0).permutation(len(ids))
    c = F.elegir_frios(ids[perm], 0.10, 42)
    assert (c == a[perm]).all()
    # otra semilla elige otros; y el hash es determinista (valor fijado a mano de splitmix64)
    assert (F.elegir_frios(ids, 0.10, 43) != a).any()


def test_hash_splitmix64_valor_conocido():
    # splitmix64 con estado inicial 0: la primera salida es 0xE220A8397B1DCDAF (referencia publicada del algoritmo)
    assert int(F.hash_estable(np.array([0]), 0)[0]) == 0xE220A8397B1DCDAF


def test_frio_con_menos_de_5_ratings_queda_fuera(res):
    s = res["senal"]
    assert 4 not in set(s.user_idx)
    assert res["meta"]["frios_fuera_menos_de_min"] == 1 and res["meta"]["ratings_frios_fuera"] == 4
    assert res["meta"]["usuarios_frios_en_experimento"] == 1
    assert not res["usuarios"].frio[4] and not res["usuarios"].elegible_test[4]


def test_d8_y_elegibilidad(res):
    # el val de cada normal es el item 10, que no está en ningún train -> D8 lo quita (3 filas); los tests de u0 (items 8, 9) sí están
    assert res["meta"]["quitadas_d8_val"] == 3 and res["meta"]["quitadas_d8_test"] == 0
    assert res["meta"]["filas_val"] == 0 and res["meta"]["filas_test"] == 2
    u = res["usuarios"]
    # u0: positivos en la dada (items 0,1,5,6) pero ninguno en test (notas bajas 2 y 1) -> no elegible
    assert u.n_pos_dada[0] == 4 and not u.elegible_test[0]


def test_historiales_y_mascara_dada(res):
    ip, ix = res["mascaras"]["dada"]
    assert ix[ip[0]:ip[1]].tolist() == list(range(8))          # C14 del test del frío = su historia dada
    assert ip[2] - ip[1] == 0                                  # los normales no tienen historia dada
    for bolsa in "AB":
        a, i, w = res["vistas"][("dada", bolsa)]
        assert a[2] == a[1]
    ipA, ixA, wA = res["vistas"][("dada", "A")]
    ipB, ixB, wB = res["vistas"][("dada", "B")]
    # bolsa B de la dada: notas <= 3.1667-1 = 2.1667 -> items 3 (2.0) y 4 (1.0); bolsa A: 0,1,2,5,6,7
    assert ixB[ipB[0]:ipB[1]].tolist() == [3, 4]
    assert ixA[ipA[0]:ipA[1]].tolist() == [0, 1, 2, 5, 6, 7]
    # el historial de train del frío está vacío
    assert res["vistas"][("train", "A")][0][1] == 0
