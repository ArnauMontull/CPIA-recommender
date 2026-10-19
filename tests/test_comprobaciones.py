"""Las comprobaciones sobre un modelo sin entrenar y datos sintéticos: todas deben pasar (son propiedades del código, no del entrenamiento)."""
import torch

from src.comprobaciones import ejecutar
from src.datos import cargar
from src.modelo import DosTorres

from . import sintetico


def test_todas_las_comprobaciones_pasan_en_el_dataset_sintetico(tmp_path):
    torch.manual_seed(0)
    datos = cargar(sintetico.crear(tmp_path, n_usuarios=80, n_items=40, por_usuario=24))
    r = ejecutar(datos, DosTorres.desde_config(datos.feats, sintetico.CFG))
    fallos = {k: v["detalle"] for k, v in r["comprobaciones"].items() if not v["ok"]}
    assert not fallos, fallos
    assert r["latencia"]["ms_por_usuario_solo"] > 0
