"""Une las piezas: export -> emparejado -> media_u -> bolsas -> ficheros en data/external/letterboxd/<alias>/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .emparejar import Catalogo, emparejar
from .export import leer_export
from .perfil import bolsas, media_encogida

METODOS = ("exacto", "api", "sin_emparejar")


def construir(peliculas: pd.DataFrame, catalogo: Catalogo, n_items: int, mu: float, cfg: dict, buscar=None,
              hilos: int = 4) -> tuple[pd.DataFrame, dict, dict]:
    """peliculas (salida de leer_export) -> (emparejado, bolsas, meta). Funcion pura (salvo `buscar`).

    media_u se calcula con TODAS las notas del export (es la tendencia del usuario, este o no la pelicula en el
    catalogo); las bolsas solo con las emparejadas. Si dos peliculas de Letterboxd caen en el mismo movieId, en las
    bolsas queda la de fecha mas reciente.
    """
    emp = emparejar(peliculas, catalogo, buscar, hilos)
    cruda, media_u = media_encogida(emp["rating"].to_numpy(), mu, cfg["letterboxd"])
    ok = emp[emp.metodo != "sin_emparejar"].sort_values("fecha", kind="stable", na_position="first")
    ok = ok.drop_duplicates(subset="item_idx", keep="last")
    item, nota = ok["item_idx"].to_numpy().astype("int64"), ok["rating"].to_numpy()
    b = bolsas(item, nota, media_u, n_items, cfg["senal"], cfg["historial"])
    n = len(emp)
    meta = {
        "n_notas": int(n), "n_emparejadas": int(len(emp) - (emp.metodo == "sin_emparejar").sum()),
        "n_duplicados_movieId": int((emp.metodo != "sin_emparejar").sum() - len(ok)),
        "pct_emparejado": {m: round(100.0 * float((emp.metodo == m).sum()) / max(n, 1), 2) for m in METODOS},
        "n_por_metodo": {m: int((emp.metodo == m).sum()) for m in METODOS},
        "media_u_cruda": None if np.isnan(cruda) else round(cruda, 4), "media_u": round(media_u, 4),
        "mu_global_train": round(mu, 4), "encogida": bool(n < cfg["letterboxd"]["min_notas_sin_encoger"]),
        "tamano_bolsa": {"A": int(len(b["A"][0])), "B": int(len(b["B"][0])), "valorados": int(len(ok))},
        "parametros": {"letterboxd": cfg["letterboxd"], "senal": cfg["senal"], "historial": cfg["historial"]},
    }
    return emp, {**{f"hist_{k}": v for k, v in b.items()}, "valorados": np.sort(item).astype("int32")}, meta


def guardar(carpeta: Path, emp: pd.DataFrame, bol: dict, meta: dict) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    emp.to_parquet(carpeta / "emparejado.parquet", index=False)
    for k in ("A", "B"):
        ix, ps = bol[f"hist_{k}"]
        np.save(carpeta / f"hist_{k}_indices.npy", ix)
        np.save(carpeta / f"hist_{k}_pesos.npy", ps)
    np.save(carpeta / "valorados_indices.npy", bol["valorados"])
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def ejecutar(export, carpeta: Path, catalogo: Catalogo, n_items: int, mu: float, cfg: dict, buscar=None, hilos=4) -> dict:
    emp, bol, meta = construir(leer_export(export), catalogo, n_items, mu, cfg, buscar, hilos)
    guardar(carpeta, emp, bol, meta)
    return meta
