"""Torres del modelo: la media en torch = la de numpy (historial.media_bolsa), formas, normas, dropout del ID y gradientes."""
import numpy as np
import pytest
import torch

from src.datos import cargar
from src.modelo import Bolsa, DosTorres, media_bolsa
from src.preprocesamiento.historial import media_bolsa as media_numpy

from . import sintetico


@pytest.fixture(scope="module")
def datos(tmp_path_factory):
    return cargar(sintetico.crear(tmp_path_factory.mktemp("sint")))


def test_media_bolsa_torch_igual_a_numpy(datos):
    ip, ix, pw = datos.hist[("train", "A")]
    V = np.random.default_rng(1).normal(size=(datos.n_items, 8)).astype("float32")
    filas = np.array([0, 5, 7, 3])
    # el objetivo de cada fila: un item que realmente está en su bolsa, con su peso
    t = np.array([ix[ip[f]] for f in filas])
    w = np.array([pw[ip[f]] for f in filas])
    esperado, vacia_e = media_numpy(V, ip, ix, pw, filas, t, w)
    media, vacia, n = media_bolsa(torch.tensor(V), Bolsa(ip, ix, pw), torch.tensor(filas), torch.tensor(t), torch.tensor(w))
    np.testing.assert_allclose(media.numpy(), esperado, atol=1e-5)
    assert (vacia.numpy() == vacia_e).all()
    assert n.tolist() == [int(ip[f + 1] - ip[f]) - 1 for f in filas]
    # sin exclusión
    esperado, _ = media_numpy(V, ip, ix, pw, filas)
    media, _, _ = media_bolsa(torch.tensor(V), Bolsa(ip, ix, pw), torch.tensor(filas))
    np.testing.assert_allclose(media.numpy(), esperado, atol=1e-5)


def test_bolsa_vacia_da_vector_cero_y_bandera():
    V = torch.ones(4, 3)
    b = Bolsa(np.array([0, 0, 1]), np.array([2]), np.array([0.5], dtype="float32"))     # usuario 0 sin nada; usuario 1 solo el objetivo
    media, vacia, n = media_bolsa(V, b, torch.tensor([0, 1]), torch.tensor([0, 2]), torch.tensor([0.0, 0.5]))
    assert vacia.tolist() == [True, True]
    assert (media == 0).all()


def test_formas_y_normas(datos):
    torch.manual_seed(0)
    m = DosTorres.desde_config(datos.feats, sintetico.CFG).eval()
    V = m.vectores_items()
    assert V.shape == (datos.n_items, 128)
    torch.testing.assert_close(V.norm(dim=1), torch.ones(datos.n_items))
    A, B = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    U = m.vectores_usuarios(V, torch.arange(10), A, B)
    assert U.shape == (10, 128)
    torch.testing.assert_close(U.norm(dim=1), torch.ones(10))


def test_dropout_del_id_solo_en_entrenamiento(datos):
    torch.manual_seed(0)
    m = DosTorres.desde_config(datos.feats, sintetico.CFG)
    m.eval()
    torch.testing.assert_close(m.vectores_items(), m.vectores_items())
    m.train()
    assert not torch.allclose(m.vectores_items(), m.vectores_items())


def test_id_de_pelicula_sin_train_no_cuenta(datos):
    feats = dict(datos.feats)
    feats["en_train"] = np.zeros(datos.n_items, dtype=bool)
    feats["en_train"][:5] = True
    m = DosTorres.desde_config(feats, sintetico.CFG).eval()
    antes = m.vectores_items().detach().clone()
    with torch.no_grad():
        m.items.emb_id.weight += 1.0
    despues = m.vectores_items()
    torch.testing.assert_close(despues[5:], antes[5:])          # sin ID entrenado: el vector solo depende del contenido
    assert not torch.allclose(despues[:5], antes[:5])


def test_gradiente_llega_a_la_torre_de_items_por_el_historial(datos):
    m = DosTorres.desde_config(datos.feats, sintetico.CFG).eval()
    A, B = Bolsa(*datos.hist[("train", "A")]), Bolsa(*datos.hist[("train", "B")])
    V = m.vectores_items()
    m.vectores_usuarios(V, torch.arange(6), A, B).sum().backward()
    assert m.items.texto_denso.weight.grad.abs().sum() > 0
    assert m.usuario.mlp[0].weight.grad.abs().sum() > 0
