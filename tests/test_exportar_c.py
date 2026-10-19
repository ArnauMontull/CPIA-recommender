"""Exportación al formato de la guía H10: tamaños de los binarios y coherencia numpy/PyTorch."""
import numpy as np
import pytest
import torch

from src.datos import cargar
from src.exportar_c import exportar
from src.modelo import DosTorres

from . import sintetico


@pytest.fixture()
def datos(tmp_path):
    return cargar(sintetico.crear(tmp_path))


def test_exporta_el_contrato(datos, tmp_path):
    torch.manual_seed(0)
    m = DosTorres.desde_config(datos.feats, sintetico.CFG)
    out = tmp_path / "c"
    meta = exportar(m, datos, out)
    N, d, h1 = datos.n_items, 128, 256
    assert (meta["N"], meta["d"], meta["d0"], meta["h1"]) == (N, d, 2 * d + 3, h1)
    tam = lambda f: (out / f).stat().st_size
    assert tam("V.f32") == N * d * 4 and tam("W1.f32") == h1 * (2 * d + 3) * 4 and tam("W2.f32") == d * h1 * 4
    assert tam("b1.f32") == h1 * 4 and tam("b2.f32") == d * 4 and tam("ref_scores.f32") == N * 4
    assert tam("usuario_A_idx.i32") == meta["n_A"] * 4 == tam("usuario_A_w.f32")
    V = np.fromfile(out / "V.f32", "<f4").reshape(N, d)
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-5)
    assert meta["dif_max_torch_numpy"] < 1e-4 and "orden_entrada = media_A,media_B,vacia_A,vacia_B,log1p_n" in (out / "meta.txt").read_text()


def test_la_referencia_no_recomienda_lo_excluido(datos, tmp_path):
    m = DosTorres.desde_config(datos.feats, sintetico.CFG)
    exportar(m, datos, tmp_path / "c")
    top = np.fromfile(tmp_path / "c" / "ref_top.i32", "<i4")
    excl = np.fromfile(tmp_path / "c" / "usuario_excl.i32", "<i4")
    validos = top[:datos.n_items - len(excl)]            # catálogo de 30: el top-200 acaba con los excluidos (-inf)
    assert not set(validos) & set(excl) and len(set(top)) == len(top)
