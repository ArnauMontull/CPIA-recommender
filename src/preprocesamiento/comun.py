"""Rutas y utilidades compartidas por las etapas de preprocesamiento."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
CRUDOS_ML = DATA / "ml-32m"
CRUDOS_TMDB = DATA / "the_movies_dataset_kaggle"
PROCESADOS = DATA / "processed"
INFORMES = RAIZ / "reports"

# Barra de progreso del LIFE OS de Arnau; si no existe (p. ej. en el PC de Uriel) no hace nada.
_PROGRESO = RAIZ.parents[2] / "scripts" / "progreso.py"
_marcar = None
if _PROGRESO.exists():
    _spec = importlib.util.spec_from_file_location("progreso", _PROGRESO)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _marcar = _mod.marcar


def marcar(tarea: str, hecho: int, total: int, texto: str = "") -> None:
    if _marcar is not None:
        _marcar(tarea, hecho, total, texto)


def leer_ratings() -> pd.DataFrame:
    """ratings.csv con tipos compactos (~400 MB en RAM en lugar de ~1 GB)."""
    return pd.read_csv(
        CRUDOS_ML / "ratings.csv",
        engine="pyarrow",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
