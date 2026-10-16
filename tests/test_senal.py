"""Tests de la etapa 2 (señal) con datos sintéticos hechos a mano (Ana, Bruno y Carla del planning de C)."""
import numpy as np
import pandas as pd
import pytest

from src.preprocesamiento import senal as S

CFG = {"tope_umbral": 4.0, "margen_nota_baja": 1.0, "peso_primer_dia": 0.75,
       "segundos_primer_dia": 86400, "umbral_fijo_eval": 4.0, "suavizado_logq": 1.0}
# items: 0 Pulp, 1 Matrix, 2 Shrek, 3 Titanic, 4 Notting Hill, 5 Batman & Robin
N = 6
COLS = ["user_idx", "item_idx", "rating", "timestamp", "particion", "primer_dia"]
# usuario 0 Ana, 1 Bruno, 2 Carla; todo en train
NOTAS = {
    0: {1: 5.0, 2: 4.5, 3: 5.0, 4: 4.0, 5: 3.5},
    1: {0: 4.0, 1: 3.5, 2: 3.0, 3: 1.5, 4: 2.0},
    2: {0: 5.0, 1: 4.0, 2: 4.0, 3: 3.0, 5: 1.0},
}


def tipar(df):
    return df.astype({"rating": "float32", "user_idx": "int32", "item_idx": "int32", "particion": "int8"})


def tres_usuarios() -> pd.DataFrame:
    filas = [(u, i, r, 1000 + i, 0, False) for u, d in NOTAS.items() for i, r in d.items()]
    return tipar(pd.DataFrame(filas, columns=COLS))


def medias(df, n=3):
    return S.medias_train(df.user_idx.to_numpy(), df.rating.to_numpy(), df.particion.to_numpy(), n)


def positivos(df, u):
    e = S.etiquetar(df, medias(df), CFG)
    return set(e[(e.user_idx == u) & e.positivo].item_idx)


def test_medias_ana_bruno_carla():
    assert medias(tres_usuarios()) == pytest.approx([4.4, 2.8, 3.4])


def test_positivos_min_4_media():
    df = tres_usuarios()
    assert positivos(df, 0) == {1, 2, 3, 4}   # Ana: umbral 4,0, sin Batman & Robin
    assert positivos(df, 1) == {0, 1, 2}      # Bruno: umbral 2,8: Pulp, Matrix, Shrek
    assert positivos(df, 2) == {0, 1, 2}      # Carla: umbral 3,4


def test_nota_baja_bruno():
    e = S.etiquetar(tres_usuarios(), medias(tres_usuarios()), CFG)
    # Bruno: rating <= 2,8 - 1 = 1,8 -> solo Titanic (1,5)
    assert set(e[(e.user_idx == 1) & e.nota_baja].item_idx) == {3}


def test_positivo_gana_a_nota_baja():
    # Usuario con todo 5,0 en train: un 4,0 en test es >= 4 (positivo) y <= 5 - 1 (nota baja); cuenta solo como positivo.
    df = tipar(pd.DataFrame([(0, 0, 5.0, 1, 0, False), (0, 1, 5.0, 2, 0, False), (0, 2, 4.0, 3, 2, False)], columns=COLS))
    e = S.etiquetar(df, medias(df, 1), CFG)
    assert e.positivo.iloc[2] and not e.nota_baja.iloc[2]


def test_peso_cinco_primer_dia():
    # Ana, Matrix 5,0 en el primer día: (1 + 5 - 4,4) * 0,75 = 1,2; Titanic 5,0 otro día: 1,6; Batman no positivo: 0
    df = tres_usuarios()
    df.loc[(df.user_idx == 0) & (df.item_idx == 1), "primer_dia"] = True
    m = medias(df)
    e = S.etiquetar(df, m, CFG)
    w = S.pesos(e, m, CFG)
    e["w"] = w
    ana = e[e.user_idx == 0].set_index("item_idx").w
    assert ana[1] == pytest.approx(1.2, rel=1e-5)
    assert ana[3] == pytest.approx(1.6, rel=1e-5)
    assert ana[5] == 0
    assert w.dtype == np.float32


def test_logq_suma_uno_y_valor():
    df = tres_usuarios()
    e = S.etiquetar(df, medias(df), CFG)
    lq = S.logq(e.item_idx.to_numpy(), e.positivo.to_numpy(), e.particion.to_numpy(), N, 1.0)
    assert lq.dtype == np.float32
    assert np.exp(lq.astype("float64")).sum() == pytest.approx(1.0, abs=1e-6)
    # positivos por película: Pulp 2, Matrix 3, Shrek 3, Titanic 1, Notting 1, Batman 0; total 10; N=6, s=1 -> /16
    assert np.exp(lq[0]) == pytest.approx(3 / 16, rel=1e-5)
    assert np.exp(lq[1]) == pytest.approx(4 / 16, rel=1e-5)
    assert np.exp(lq[5]) == pytest.approx(1 / 16, rel=1e-5)


def con_val_test() -> pd.DataFrame:
    """Ana: train 4,0 5,0 3,0 (media 4,0); val 4,5; test 2,0. Bruno: train 4,0 2,0 (media 3,0); val 3,0; test 1,0."""
    filas = [
        (0, 0, 4.0, 1, 0, True), (0, 1, 5.0, 2, 0, False), (0, 2, 3.0, 3, 0, False),
        (0, 3, 4.5, 4, 1, False), (0, 4, 2.0, 5, 2, False),
        (1, 0, 4.0, 1, 0, True), (1, 1, 2.0, 2, 0, False),
        (1, 3, 3.0, 3, 1, False), (1, 4, 1.0, 4, 2, False),
    ]
    return tipar(pd.DataFrame(filas, columns=COLS))


def test_no_fuga_de_test():
    df = con_val_test()
    a = S.construir(df, CFG, 2, N)
    df2 = df.copy()
    df2.loc[(df2.user_idx == 0) & (df2.particion == 2), "rating"] = 5.0   # el test de Ana pasa de 2,0 a 5,0
    b = S.construir(df2, CFG, 2, N)
    assert a["usuarios"].media_u.tolist() == b["usuarios"].media_u.tolist()
    t = (df.particion == 0).to_numpy()
    assert (a["senal"].positivo[t] == b["senal"].positivo[t]).all()
    assert (a["senal"].peso == b["senal"].peso).all()
    assert np.array_equal(a["logq"], b["logq"])
    assert np.array_equal(a["indptr"], b["indptr"]) and np.array_equal(a["indices"], b["indices"])
    # la etiqueta del propio test sí cambia (fila 4: Ana, item 4)
    assert not a["senal"].positivo.iloc[4] and b["senal"].positivo.iloc[4]


def test_mascara_csr():
    df = con_val_test()
    indptr, indices = S.mascara_valorados(df.user_idx.to_numpy(), df.item_idx.to_numpy(),
                                          df.particion.to_numpy(), 2, N)
    assert indptr.dtype == np.int64 and indices.dtype == np.int32
    assert indptr.tolist() == [0, 3, 5]
    assert indices[indptr[0]:indptr[1]].tolist() == [0, 1, 2]   # val/test (items 3 y 4) no entran
    assert indices[indptr[1]:indptr[2]].tolist() == [0, 1]


def test_elegibles_c8_y_relevante_4():
    r = S.construir(con_val_test(), CFG, 2, N)
    u = r["usuarios"]
    # Ana (umbral 4,0): val 4,5 positivo; test 2,0 no. Bruno (umbral 3,0): val 3,0 positivo; test 1,0 no.
    assert u.elegible_val.tolist() == [True, True]
    assert u.elegible_test.tolist() == [False, False]   # C8: sin positivos en test -> fuera de las métricas
    # relevante_4 (>= 4,0): Ana val 4,5 sí; Bruno val 3,0 no
    assert u.elegible_val_4.tolist() == [True, False]
    assert u.elegible_test_4.tolist() == [False, False]
    assert r["senal"].relevante_4.tolist() == [True, True, False, True, False, True, False, False, False]
    assert u.n_train.tolist() == [3, 2] and u.n_pos_train.tolist() == [2, 1]


def test_sin_positivos_en_train_no_elegible():
    # media 3,0 con train 1,0 y 5,0: el umbral es 3,0 y solo el 5,0 es positivo; el test 5,0 es positivo
    filas = [(0, 0, 1.0, 1, 0, False), (0, 1, 5.0, 2, 0, False), (0, 2, 5.0, 3, 2, False),
             (1, 0, 3.0, 1, 0, False), (1, 1, 3.0, 2, 0, False), (1, 2, 5.0, 3, 2, False)]
    u = S.construir(tipar(pd.DataFrame(filas, columns=COLS)), CFG, 2, N)["usuarios"]
    assert u.n_pos_train.tolist() == [1, 2]
    assert u.elegible_test.tolist() == [True, True]
    # sin positivo en train: todo train por debajo del umbral es imposible con media_u; se fuerza con un usuario sin train
    filas = [(0, 2, 5.0, 3, 2, False)]
    u = S.construir(tipar(pd.DataFrame(filas, columns=COLS)), CFG, 1, N)["usuarios"]
    assert u.n_pos_train.tolist() == [0]
    assert u.elegible_test.tolist() == [False] and u.elegible_test_4.tolist() == [False]
