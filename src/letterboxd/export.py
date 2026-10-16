"""Lectura del export de Letterboxd (zip o carpeta con ratings.csv y diary.csv) y re-visionados (C12, G3).

ratings.csv: Date, Name, Year, Letterboxd URI, Rating   (Date = dia en que se puso la nota)
diary.csv:   Date, Name, Year, Letterboxd URI, Rating, Rewatch, Tags, Watched Date
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

COLUMNAS = ["Name", "Year", "Letterboxd URI", "rating", "fecha"]


def _leer_csv(origen: Path, nombre: str) -> pd.DataFrame | None:
    """Lee `nombre` de un zip o de una carpeta; None si no existe."""
    if origen.is_dir():
        p = origen / nombre
        return pd.read_csv(p, dtype={"Name": "string"}) if p.exists() else None
    with zipfile.ZipFile(origen) as z:
        if nombre not in z.namelist():
            return None
        return pd.read_csv(io.BytesIO(z.read(nombre)), dtype={"Name": "string"})


def combinar(ratings: pd.DataFrame | None, diary: pd.DataFrame | None) -> pd.DataFrame:
    """Una fila por pelicula (Name, Year) con su ULTIMA nota (C12).

    La fecha de cada nota es la `Watched Date` del diary (o su `Date` si falta) y la `Date` de ratings.csv.
    Gana la fecha mayor; en empate, ratings.csv (que guarda la nota vigente). Las filas sin nota se descartan.
    Devuelve Name, Year (Int16), Letterboxd URI, rating (float32), fecha (datetime64), ordenado por fecha.
    """
    partes = []
    if ratings is not None and len(ratings):
        r = ratings.rename(columns={"Rating": "rating"}).copy()
        r["fecha"] = pd.to_datetime(r["Date"], errors="coerce")
        r["_orden"] = 1                       # gana en empate
        partes.append(r)
    if diary is not None and len(diary):
        d = diary.rename(columns={"Rating": "rating"}).copy()
        fecha = pd.to_datetime(d["Watched Date"], errors="coerce") if "Watched Date" in d else pd.NaT
        d["fecha"] = pd.Series(fecha, index=d.index).fillna(pd.to_datetime(d["Date"], errors="coerce"))
        d["_orden"] = 0
        partes.append(d)
    if not partes:
        return pd.DataFrame({"Name": pd.Series(dtype="string"), "Year": pd.Series(dtype="Int16"),
                             "Letterboxd URI": pd.Series(dtype="string"), "rating": pd.Series(dtype="float32"),
                             "fecha": pd.Series(dtype="datetime64[ns]")})
    t = pd.concat([p[["Name", "Year", "Letterboxd URI", "rating", "fecha", "_orden"]] for p in partes], ignore_index=True)
    t["rating"] = pd.to_numeric(t["rating"], errors="coerce")
    t = t[t["rating"].notna()].copy()
    t["Year"] = pd.to_numeric(t["Year"], errors="coerce").astype("Int16")
    # Las fechas ausentes pierden frente a las presentes; despues se quedan con la ultima por pelicula.
    t = t.sort_values(["fecha", "_orden"], kind="stable", na_position="first")
    t = t.drop_duplicates(subset=["Name", "Year"], keep="last")
    t["rating"] = t["rating"].astype("float32")
    return t.sort_values("fecha", kind="stable", na_position="first")[COLUMNAS].reset_index(drop=True)


def leer_export(origen: str | Path) -> pd.DataFrame:
    """zip o carpeta -> DataFrame de `combinar`. Hace falta al menos ratings.csv o diary.csv."""
    origen = Path(origen)
    ratings, diary = _leer_csv(origen, "ratings.csv"), _leer_csv(origen, "diary.csv")
    if ratings is None and diary is None:
        raise FileNotFoundError(f"{origen}: no hay ratings.csv ni diary.csv")
    return combinar(ratings, diary)
