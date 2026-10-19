"""Pérdida softmax + logQ + máscara C14 + peso, comprobada a mano con un lote de 3."""
import math

import numpy as np
import torch

from src.perdida import Valorados, mascara_lote, perdida_softmax_logq


def test_lote_de_tres_a_mano():
    U = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    Vb = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
    logq = torch.tensor([0.0, math.log(2), 0.0])
    peso = torch.tensor([1., 2., 1.])
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 2] = True                                    # el usuario 0 ya había valorado la película de la columna 2
    e = math.e
    ce0 = math.log(e + 0.5) - 1.0                        # logits fila 0: [1, -ln2, -inf]
    ce1 = math.log(1 + e / 2 + e) - (1 - math.log(2))    # fila 1: [0, 1-ln2, 1]; positivo = columna 1
    ce2 = math.log(e + 0.5 + 1) - 0.0                    # fila 2: [1, -ln2, 0]; positivo = columna 2
    esperado = (1 * ce0 + 2 * ce1 + 1 * ce2) / 4
    assert abs(float(perdida_softmax_logq(U, Vb, logq, 1.0, peso, mask)) - esperado) < 1e-5


def test_la_mascara_cambia_la_perdida_y_logq_corrige_el_sesgo():
    U = torch.tensor([[1., 0.], [0., 1.]])
    Vb = torch.tensor([[0.6, 0.8], [0.8, 0.6]])
    peso = torch.ones(2)
    sin = perdida_softmax_logq(U, Vb, torch.zeros(2), 0.5, peso)
    con = perdida_softmax_logq(U, Vb, torch.zeros(2), 0.5, peso, torch.tensor([[False, True], [True, False]]))
    assert con < sin                                      # quitar negativos solo puede bajar la pérdida
    # restar logQ de un negativo popular lo hace más fuerte
    popular = perdida_softmax_logq(U, Vb, torch.tensor([0.0, -3.0]), 0.5, peso)
    assert popular != sin


def test_valorados_busca_pares_usuario_pelicula():
    # usuario 0: {1, 4}; usuario 1: {}; usuario 2: {0, 4}
    v = Valorados(np.array([0, 2, 2, 4]), np.array([1, 4, 0, 4]), n_items=5)
    m = v.contiene(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 4]))
    assert m.tolist() == [[False, True, True], [False, False, False], [True, False, True]]


def test_mascara_lote_respeta_la_diagonal_y_enmascara_duplicados():
    # usuarios 0 y 1 con el mismo objetivo (película 3): el ejemplo del otro no es un negativo válido
    v = Valorados(np.array([0, 1, 2]), np.array([3, 3]), n_items=5)
    m = mascara_lote(torch.tensor([0, 1]), torch.tensor([3, 3]), v)
    assert m.tolist() == [[False, True], [True, False]]
