"""Tests de la etapa de historiales (bloque F) con datos sintéticos hechos a mano."""
import numpy as np
import pandas as pd
import pytest

from src.preprocesamiento import historial as H
from src.preprocesamiento import senal as S

CS = {"tope_umbral": 4.0, "margen_nota_baja": 1.0, "peso_primer_dia": 0.75,
      "segundos_primer_dia": 86400, "umbral_fijo_eval": 4.0, "suavizado_logq": 1.0}
CH = {"peso_neutra": 0.25}
N = 6
COLS = ["user_idx", "item_idx", "rating", "timestamp", "particion", "primer_dia"]


def preparar(filas, n_usuarios):
    df = pd.DataFrame(filas, columns=COLS).astype(
        {"rating": "float32", "user_idx": "int32", "item_idx": "int32", "particion": "int8"})
    media = S.medias_train(df.user_idx.to_numpy(), df.rating.to_numpy(), df.particion.to_numpy(), n_usuarios)
    df = S.etiquetar(df, media, CS)
    df["peso"] = S.pesos(df, media, CS)
    return df, media


def usuario0():
    # train: item0 5.0 (dia1), item1 4.0, item2 3.0 (neutra), item3 1.0 (baja, dia1); media_u = 3.25
    # val: item4 5.0 ; test: item5 5.0
    return [(0, 0, 5.0, 100, 0, True), (0, 1, 4.0, 200000, 0, False), (0, 2, 3.0, 200001, 0, False),
            (0, 3, 1.0, 100000, 0, True), (0, 4, 5.0, 300000, 1, False), (0, 5, 5.0, 400000, 2, False)]


def test_pesos_positivo_neutra_y_baja_primer_dia_y_otro():
    df, media = preparar(usuario0(), 1)
    assert media[0] == pytest.approx(3.25)
    w, es_b = H.pesos_historial(df, media, CS, CH)
    assert w[0] == pytest.approx(2.75 * 0.75)    # positivo 5.0 del primer dia: (1+5-3.25)*0.75
    assert w[1] == pytest.approx(1.75)           # positivo 4.0 de otro dia: 1+0.75
    assert w[2] == pytest.approx(0.25)           # neutra de otro dia (3.0: ni >=3.25 ni <=2.25)
    assert w[3] == pytest.approx(0.75)           # nota baja del primer dia: 1*0.75
    assert list(es_b) == [False, False, False, True, False, False]
    assert w[0] == pytest.approx(df.peso[0]) and w[1] == pytest.approx(df.peso[1])   # = peso de senal


def test_neutra_primer_dia_y_baja_otro_dia():
    filas = [(0, 0, 2.5, 10, 0, True), (0, 1, 1.0, 900000, 0, False), (0, 2, 5.0, 900001, 0, False),
             (0, 3, 3.5, 900002, 0, False)]   # media 3.0; item0 neutra, item1 baja (1<=2), item2 positivo
    df, media = preparar(filas, 1)
    w, es_b = H.pesos_historial(df, media, CS, CH)
    assert w[0] == pytest.approx(0.25 * 0.75) and w[1] == pytest.approx(1.0)
    assert w[2] == pytest.approx(3.0) and not es_b[0] and es_b[1]


def vistas_u0():
    df, media = preparar(usuario0(), 1)
    return H.construir_vistas(df, media, CS, CH, 1, N)


def test_val_en_train_val_no_en_train_y_test_en_ninguna():
    v = vistas_u0()

    def items(vista):
        return set(v[(vista, "A")][1]) | set(v[(vista, "B")][1])
    assert items("train") == {0, 1, 2, 3}
    assert items("train_val") == {0, 1, 2, 3, 4}
    assert 4 not in items("train") and 5 not in items("train_val")
    ip, ix, pw = v[("train_val", "A")]
    assert pw[list(ix).index(4)] == pytest.approx(2.75)   # val 5.0 con media de train 3.25
    assert set(v[("train", "A")][1]) == {0, 1, 2} and list(v[("train", "B")][1]) == [3]


def test_media_bolsa_quitando_objetivo_igual_a_mano():
    v = vistas_u0()
    V = np.random.default_rng(0).normal(size=(N, 3)).astype("float32")
    ip, ix, pw = v[("train", "A")]
    wa = {0: 2.75 * 0.75, 1: 1.75, 2: 0.25}
    m, vacia = H.media_bolsa(V, ip, ix, pw, np.array([0]), np.array([0]), np.array([wa[0]]))
    esperado = (wa[1] * V[1] + wa[2] * V[2]) / (wa[1] + wa[2])
    assert not vacia[0] and m[0] == pytest.approx(esperado, rel=1e-5)
    m2, _ = H.media_bolsa(V, ip, ix, pw, np.array([0]))
    assert m2[0] == pytest.approx(sum(wa[i] * V[i] for i in wa) / sum(wa.values()), rel=1e-5)
    mb, vb = H.media_bolsa(V, *v[("train", "B")], np.array([0]))
    assert mb[0] == pytest.approx(V[3], rel=1e-5) and not vb[0]


def test_un_solo_positivo_y_nada_mas_historial_vacio():
    filas = [(0, 0, 5.0, 10, 0, False), (1, 1, 4.0, 10, 0, False), (1, 2, 5.0, 11, 0, False)]
    df, media = preparar(filas, 2)
    v = H.construir_vistas(df, media, CS, CH, 2, N)
    V = np.ones((N, 2), dtype="float32")
    ip, ix, pw = v[("train", "A")]
    ej = H.ejemplos_train(df)
    assert list(ej.user_idx) == [0, 1, 1]
    m, vacia = H.media_bolsa(V, ip, ix, pw, ej.user_idx.to_numpy(), ej.item_idx.to_numpy(), ej.peso.to_numpy())
    assert list(vacia) == [True, False, False]
    assert (m[0] == 0).all()
    assert H.resumen(v, ej, 2)["ejemplos_con_A_vacia_sin_objetivo"] == 1
    mb, vb = H.media_bolsa(V, *v[("train", "B")], np.array([0]))
    assert vb[0] and (mb[0] == 0).all()


def test_csr_ordenado_y_consistente_con_indptr():
    filas = [(1, 4, 5.0, 10, 0, False), (0, 3, 4.0, 11, 0, False), (1, 0, 4.0, 12, 0, False),
             (0, 1, 4.0, 13, 0, False), (2, 2, 4.0, 14, 0, False)]
    df, media = preparar(filas, 3)
    v = H.construir_vistas(df, media, CS, CH, 3, N)
    ip, ix, pw = v[("train", "A")]
    assert ip.dtype == np.int64 and ix.dtype == np.int32 and pw.dtype == np.float32
    assert list(ip) == [0, 2, 4, 5] and ip[-1] == len(ix) == len(pw)
    for u in range(3):
        seg = ix[ip[u]:ip[u + 1]]
        assert list(seg) == sorted(seg)
    assert list(ix[ip[1]:ip[2]]) == [0, 4]
