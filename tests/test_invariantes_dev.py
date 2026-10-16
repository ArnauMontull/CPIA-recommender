"""Invariantes de la pipeline sobre las salidas reales de dev (H11): sin fugas y todo alineado.

Se salta si no existe data/processed/dev/ (p. ej. en un clon sin datos). Para generarla:
split → senal → historial → features_peliculas con --muestra dev.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocesamiento.comun import PROCESADOS

D = PROCESADOS / "dev"
pytestmark = pytest.mark.skipif(not (D / "meta_historial.json").exists(),
                                reason="sin salidas de dev")


@pytest.fixture(scope="module")
def s():
    return pd.read_parquet(D / "senal.parquet")


@pytest.fixture(scope="module")
def u():
    return pd.read_parquet(D / "usuarios.parquet").set_index("user_idx")


@pytest.fixture(scope="module")
def n_items():
    return len(pd.read_parquet(D / "mapeo_peliculas.parquet"))


def test_orden_temporal(s):
    g = s.groupby(["user_idx", "particion"]).timestamp
    mx, mn = g.max().unstack(), g.min().unstack()
    assert ((mx[0] <= mn[1]) | mn[1].isna()).all() and ((mx[1] <= mn[2]) | mn[2].isna()).all()


def test_sin_peliculas_frias_en_evaluacion(s):
    en_train = set(s.loc[s.particion == 0, "item_idx"])
    assert s.loc[s.particion > 0, "item_idx"].isin(en_train).all()


def test_media_u_solo_train(s, u):
    mu = s[s.particion == 0].groupby("user_idx").rating.mean()
    assert np.allclose(u.media_u.loc[mu.index].to_numpy(), mu.to_numpy(), atol=1e-5)


def test_etiquetas(s, u):
    m = s.user_idx.map(u.media_u).to_numpy()
    assert (s.positivo.to_numpy() == (s.rating.to_numpy() >= np.minimum(4.0, m) - 1e-6)).mean() > 0.9999
    assert not (s.positivo & s.nota_baja).any()


def test_logq_con_positivos_de_train(s, n_items):
    lq = np.load(D / "logq.npy").astype(np.float64)
    cnt = np.bincount(s.loc[(s.particion == 0) & s.positivo, "item_idx"], minlength=n_items)
    assert np.allclose(np.exp(lq), (cnt + 1) / (cnt.sum() + n_items), rtol=1e-3)


def test_mascaras_e_historiales(s):
    n_train, n_tv = int((s.particion == 0).sum()), int((s.particion <= 1).sum())
    assert np.load(D / "valorados_train_indptr.npy")[-1] == n_train
    assert np.load(D / "valorados_train_val_indptr.npy")[-1] == n_tv
    a, b = (np.load(D / f"hist_train_{k}_indptr.npy")[-1] for k in "AB")
    assert a + b == n_train
    e = pd.read_parquet(D / "ejemplos_train.parquet")
    assert len(e) == int(((s.particion == 0) & s.positivo).sum())


def test_features_alineadas(n_items):
    f = pd.read_parquet(D / "peliculas_features.parquet")
    m = pd.read_parquet(D / "mapeo_peliculas.parquet").sort_values("item_idx")
    assert (f.item_idx.to_numpy() == np.arange(n_items)).all()
    assert (f.movieId.to_numpy() == m.movieId.to_numpy()).all()
    E = np.load(PROCESADOS / "emb_texto.npy", mmap_mode="r")
    assert E.shape == (n_items, 384)
    assert int(np.isclose(np.abs(np.asarray(E, dtype=np.float32)).sum(1), 0).sum()) == int(f.sin_texto.sum())
