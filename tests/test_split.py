"""Tests de la etapa 1 (split) con DataFrames sintéticos hechos a mano."""
from __future__ import annotations

import pandas as pd

from src.preprocesamiento.split import (
    asignar_indices, marcar_primer_dia, particionar, quitar_frias, usuarios_corte_empate)


def _r(filas):
    """filas = [(user_idx, item_idx, timestamp)] -> DataFrame con rating fijo."""
    return pd.DataFrame(filas, columns=["user_idx", "item_idx", "timestamp"]).assign(rating=3.0)


def test_diez_ratings_son_8_1_1():
    r = particionar(_r([(0, i, 100 + i) for i in range(10)]), 0.1, 0.1, 1)
    assert r.particion.tolist() == [0] * 8 + [1, 2]


def test_empates_se_resuelven_por_movieid():
    # los 10 al mismo segundo y escritos en orden inverso: manda item_idx (= orden de movieId)
    r = particionar(_r([(0, i, 500) for i in range(9, -1, -1)]), 0.1, 0.1, 1)
    assert r.item_idx.tolist() == list(range(10))
    assert r.particion.tolist() == [0] * 8 + [1, 2]
    assert usuarios_corte_empate(r) == 1


def test_corte_sin_empate_no_cuenta():
    r = particionar(_r([(0, i, 100 + i) for i in range(10)]), 0.1, 0.1, 1)
    assert usuarios_corte_empate(r) == 0


def test_ningun_val_test_anterior_a_train():
    # tres usuarios de 20 ratings con marcas de tiempo desordenadas en la entrada: n=20 -> 16/2/2
    filas = [(u, i, 1000 - 7 * ((i * 5 + u) % 23)) for u in range(3) for i in range(20)]
    r = particionar(_r(filas), 0.1, 0.1, 1)
    for _, g in r.groupby("user_idx"):
        assert g[g.particion == 0].timestamp.max() <= g[g.particion == 1].timestamp.min()
        assert g[g.particion == 1].timestamp.max() <= g[g.particion == 2].timestamp.min()
        assert g.particion.value_counts().to_dict() == {0: 16, 1: 2, 2: 2}


def test_pelicula_solo_en_val_o_test_sale():
    # usuario 0: items 0..9; el 8 solo está en su val y el 9 solo en su test
    r = quitar_frias(particionar(_r([(0, i, 100 + i) for i in range(10)]), 0.1, 0.1, 1))
    assert r.item_idx.tolist() == list(range(8))
    assert r.particion.tolist() == [0] * 8


def test_pelicula_con_rating_en_train_de_otro_usuario_se_queda():
    # usuario 1 valora el 9 al principio (train); el 8 sigue siendo frío
    u1 = [(1, 9, 100)] + [(1, i, 101 + i) for i in range(9)]  # train: 9,0..6; val: 7; test: 8
    r = quitar_frias(particionar(_r([(0, i, 100 + i) for i in range(10)] + u1), 0.1, 0.1, 1))
    assert len(r) == 20 - 2  # fuera solo el 8: val de u0 y test de u1; el 9 de u0 (test) y el 7 de u1 (val) se quedan
    assert r[(r.user_idx == 0) & (r.particion == 2)].item_idx.tolist() == [9]
    assert 8 not in r[r.particion > 0].item_idx.tolist()


def test_usuario_con_2_ratings_todo_train():
    r = particionar(_r([(0, 0, 10), (0, 1, 20)]), 0.1, 0.1, 1)
    assert r.particion.tolist() == [0, 0]


def test_usuario_con_3_ratings_es_1_1_1():
    r = particionar(_r([(0, 0, 10), (0, 1, 20), (0, 2, 30)]), 0.1, 0.1, 1)
    assert r.particion.tolist() == [0, 1, 2]


def test_redondeo_de_25_ratings():
    # 0,1 * 25 = 2,5 -> rint = 2 (par); 21 en train
    r = particionar(_r([(0, i, i) for i in range(25)]), 0.1, 0.1, 1)
    assert r.particion.value_counts().to_dict() == {0: 21, 1: 2, 2: 2}


def test_usuarios_independientes():
    # u0: 10 ratings (8/1/1); u1: 2 ratings (todo train)
    r = particionar(_r([(0, i, 100 + i) for i in range(10)] + [(1, 0, 5), (1, 1, 6)]), 0.1, 0.1, 1)
    assert r[r.user_idx == 0].particion.tolist() == [0] * 8 + [1, 2]
    assert r[r.user_idx == 1].particion.tolist() == [0, 0]


def test_primer_dia():
    r = _r([(0, 0, 1000), (0, 1, 1000 + 86399), (0, 2, 1000 + 86400), (1, 0, 5), (1, 1, 5 + 86400)])
    r = marcar_primer_dia(r, 86400)
    assert r.primer_dia.tolist() == [True, True, False, True, False]


def test_asignar_indices():
    movies = pd.DataFrame({"movieId": [50, 10, 30]})
    ratings = pd.DataFrame({"userId": [7, 3, 7], "movieId": [30, 10, 50], "rating": 3.0, "timestamp": 1})
    r, mp, mu = asignar_indices(movies, ratings)
    assert mp.to_dict("list") == {"movieId": [10, 30, 50], "item_idx": [0, 1, 2]}
    assert mu.to_dict("list") == {"userId": [3, 7], "user_idx": [0, 1]}
    assert r.item_idx.tolist() == [1, 0, 2]
    assert r.user_idx.tolist() == [1, 0, 1]
