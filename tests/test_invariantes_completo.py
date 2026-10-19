"""Invariantes sobre las salidas reales del conjunto `completo` (32 M de ratings), sin fugas y todo alineado.

Para no cargar 1 GB en RAM: las comprobaciones por usuario se hacen con 1 de cada 20 usuarios (`user_idx % 20 == 0`, fijo) y las
globales (logQ, recuentos) leyendo solo las columnas necesarias. Se salta si no existe data/processed/completo/.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from src.preprocesamiento.comun import PROCESADOS

D = PROCESADOS / "completo"
pytestmark = pytest.mark.skipif(not (D / "meta_historial.json").exists(), reason="sin salidas de completo")
PASO = 20


@pytest.fixture(scope="module")
def sub():
    """Ratings de 1 de cada PASO usuarios, con todas las columnas de senal."""
    n_u = len(pd.read_parquet(D / "usuarios.parquet", columns=["user_idx"]))
    elegidos = np.arange(0, n_u, PASO, dtype="int32").tolist()
    t = ds.dataset(D / "senal.parquet", format="parquet").to_table(filter=ds.field("user_idx").isin(elegidos))
    return t.to_pandas()


@pytest.fixture(scope="module")
def usuarios():
    return pd.read_parquet(D / "usuarios.parquet").set_index("user_idx")


@pytest.fixture(scope="module")
def n_items():
    return len(pd.read_parquet(D / "mapeo_peliculas.parquet"))


@pytest.fixture(scope="module")
def cols():
    """Solo item_idx, positivo, particion (190 MB) para los recuentos globales."""
    return pd.read_parquet(D / "senal.parquet", columns=["item_idx", "positivo", "particion"])


def test_hay_muchos_datos(sub):
    assert len(sub) > 1_000_000 and sub.user_idx.nunique() > 8_000


def test_orden_temporal_por_usuario(sub):
    g = sub.groupby(["user_idx", "particion"]).timestamp
    mx, mn = g.max().unstack(), g.min().unstack()
    assert ((mx[0] <= mn[1]) | mn[1].isna()).all() and ((mx[1] <= mn[2]) | mn[2].isna()).all()


def test_sin_peliculas_frias_en_evaluacion(cols):
    en_train = np.zeros(cols.item_idx.max() + 1, dtype=bool)
    en_train[cols.item_idx[cols.particion == 0]] = True
    assert en_train[cols.item_idx[cols.particion > 0]].all()


def test_media_u_solo_train(sub, usuarios):
    mu = sub[sub.particion == 0].groupby("user_idx").rating.mean()
    assert np.allclose(usuarios.media_u.loc[mu.index].to_numpy(), mu.to_numpy(), atol=1e-5)


def test_etiquetas(sub, usuarios):
    m = sub.user_idx.map(usuarios.media_u).to_numpy()
    assert (sub.positivo.to_numpy() == (sub.rating.to_numpy() >= np.minimum(4.0, m) - 1e-6)).mean() > 0.9999
    assert not (sub.positivo & sub.nota_baja).any()


def test_pesos_solo_en_positivos_de_train(sub):
    w = sub.peso.to_numpy()
    assert (w[~((sub.particion == 0) & sub.positivo).to_numpy()] == 0).all()
    assert (w[((sub.particion == 0) & sub.positivo).to_numpy()] > 0).all()


def test_logq_con_positivos_de_train(cols, n_items):
    lq = np.load(D / "logq.npy").astype(np.float64)
    cnt = np.bincount(cols.item_idx[(cols.particion == 0) & cols.positivo], minlength=n_items)
    assert np.allclose(np.exp(lq), (cnt + 1) / (cnt.sum() + n_items), rtol=1e-3)
    assert abs(np.exp(lq).sum() - 1) < 1e-5


def test_mascaras_y_ejemplos_coinciden_con_los_recuentos(cols):
    n_train, n_tv = int((cols.particion == 0).sum()), int((cols.particion <= 1).sum())
    assert np.load(D / "valorados_train_indptr.npy")[-1] == n_train
    assert np.load(D / "valorados_train_val_indptr.npy")[-1] == n_tv
    a, b = (np.load(D / f"hist_train_{k}_indptr.npy")[-1] for k in "AB")
    assert a + b == n_train
    assert len(pd.read_parquet(D / "ejemplos_train.parquet")) == int(((cols.particion == 0) & cols.positivo).sum())


def test_historiales_sin_el_test(sub, usuarios):
    """La vista train_val no contiene nada de test: para los usuarios de la submuestra, los índices de la bolsa A/B son
    exactamente sus películas de particion 0 y 1 (con la etiqueta de la bolsa)."""
    n_u = len(usuarios)
    ipA, ixA = np.load(D / "hist_train_val_A_indptr.npy"), np.load(D / "hist_train_val_A_indices.npy", mmap_mode="r")
    ipB, ixB = np.load(D / "hist_train_val_B_indptr.npy"), np.load(D / "hist_train_val_B_indices.npy", mmap_mode="r")
    assert len(ipA) == n_u + 1
    rng = np.random.default_rng(0)
    for u in rng.choice(sub.user_idx.unique(), 150, replace=False):
        r = sub[(sub.user_idx == u) & (sub.particion <= 1)]
        esperado = set(r.item_idx)
        visto = set(np.asarray(ixA[ipA[u]:ipA[u + 1]]).tolist()) | set(np.asarray(ixB[ipB[u]:ipB[u + 1]]).tolist())
        assert visto == esperado
        test = set(sub[(sub.user_idx == u) & (sub.particion == 2)].item_idx)
        assert not (visto & test)


def test_features_alineadas(n_items):
    f = pd.read_parquet(D / "peliculas_features.parquet")
    m = pd.read_parquet(D / "mapeo_peliculas.parquet").sort_values("item_idx")
    assert (f.item_idx.to_numpy() == np.arange(n_items)).all()
    assert (f.movieId.to_numpy() == m.movieId.to_numpy()).all()
    assert not f[["anio_z", "duracion_z"]].isna().any().any()
    assert json.loads((D / "meta_historial.json").read_text(encoding="utf-8"))["n_items"] == n_items
