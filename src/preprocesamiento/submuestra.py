"""Etapa 2 (A2b): submuestra de desarrollo estratificada por usuarios.

Se muestrean USUARIOS enteros (no filas) para no romper historiales. Cada usuario cae en un estrato
  género dominante (20) × nivel de actividad (quintil de nº de ratings) × época del primer rating (3)
y de cada estrato se toma la misma fracción, así la submuestra conserva la mezcla de géneros, de usuarios
muy y poco activos y de épocas. reports/submuestra.md compara submuestra y conjunto completo.

Uso: python -m src.preprocesamiento.submuestra [--frac 0.10] [--semilla 42]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .comun import INFORMES, PROCESADOS, marcar

TAREA = "CPIA submuestra"
EPOCAS = [0, 2005, 2015, 9999]  # primer rating en 1995-2004 / 2005-2014 / 2015-2023


def matriz_generos(m: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Fila por movieId: reparto 1/k entre sus k géneros (fila de ceros si no tiene)."""
    nombres = sorted({g for gs in m.generos for g in gs})
    col = {g: i for i, g in enumerate(nombres)}
    G = np.zeros((m.movieId.max() + 1, len(nombres)), dtype="float32")
    for mid, gs in zip(m.movieId, m.generos):
        for g in gs:
            G[mid, col[g]] = 1.0 / len(gs)
    return G, nombres


def perfil_generos(r: pd.DataFrame, G: np.ndarray, n_usuarios: int) -> np.ndarray:
    """Peso de cada género por usuario (usuarios × géneros) con un bincount por género."""
    u, mv = r.userId.to_numpy(), r.movieId.to_numpy()
    return np.stack([np.bincount(u, weights=G[mv, j], minlength=n_usuarios) for j in range(G.shape[1])], axis=1)


def estratos(r: pd.DataFrame, G: np.ndarray, nombres: list[str]) -> pd.DataFrame:
    n_u = r.userId.max() + 1
    P = perfil_generos(r, G, n_u)
    usuarios = np.unique(r.userId.to_numpy())
    P = P[usuarios]
    dominante = np.where(P.sum(1) > 0, np.array(nombres)[P.argmax(1)], "(sin géneros)")
    g = r.groupby("userId")
    actividad = g.size().reindex(usuarios).to_numpy()
    primer_anio = pd.to_datetime(g.timestamp.min().reindex(usuarios), unit="s").dt.year.to_numpy()
    return pd.DataFrame({
        "userId": usuarios,
        "genero_dominante": dominante,
        "actividad_q": pd.qcut(actividad, 5, labels=False, duplicates="drop"),
        "epoca": pd.cut(primer_anio, EPOCAS, right=False, labels=["1995-2004", "2005-2014", "2015-2023"]),
        "n_ratings": actividad,
    })


def reparto_generos(r: pd.DataFrame, G: np.ndarray) -> np.ndarray:
    v = G[r.movieId.to_numpy()].sum(0)
    return v / v.sum()


def comparar(r: pd.DataFrame, s: pd.DataFrame, G, nombres, u_all, u_s, frac, semilla) -> str:
    pp = lambda a, b: f"{100 * a:.2f} % | {100 * b:.2f} % | {100 * (b - a):+.2f}"
    filas = ["| Medida | Completo | Submuestra | Diferencia (pp) |", "|---|---:|---:|---:|"]

    gc, gs = reparto_generos(r, G), reparto_generos(s, G)
    for i in np.argsort(-gc):
        filas.append(f"| género: {nombres[i]} | {pp(gc[i], gs[i])} |")
    hc = r.rating.value_counts(normalize=True).sort_index()
    hs = s.rating.value_counts(normalize=True).reindex(hc.index, fill_value=0)
    for k in hc.index:
        filas.append(f"| nota {k:.1f} | {pp(hc[k], hs[k])} |")
    for e in ["1995-2004", "2005-2014", "2015-2023"]:
        a, b = (u_all.epoca == e).mean(), (u_s.epoca == e).mean()
        filas.append(f"| usuarios que empiezan en {e} | {pp(a, b)} |")

    def top10(df):
        c = df.movieId.value_counts().to_numpy()
        return c[: max(1, len(c) // 10)].sum() / c.sum()

    max_dif_gen = 100 * np.abs(gs - gc).max()
    return "\n".join([
        "# Submuestra de desarrollo (A2b)",
        "",
        f"Generada con `python -m src.preprocesamiento.submuestra --frac {frac} --semilla {semilla}`."
        " Se muestrean usuarios enteros, estratificados por género dominante × quintil de actividad × época"
        " del primer rating.",
        "",
        "## Tamaño",
        f"- Usuarios: {len(u_s):,} de {len(u_all):,} ({len(u_s) / len(u_all):.1%})".replace(",", "."),
        f"- Ratings: {len(s):,} de {len(r):,} ({len(s) / len(r):.1%})".replace(",", "."),
        f"- Películas con algún rating: {s.movieId.nunique():,} de {r.movieId.nunique():,}".replace(",", ".")
        + " (las que faltan son de la cola larga: solo las vio algún usuario no muestreado)",
        f"- Estratos: {u_all.estrato.nunique()} (de ellos {(u_all.groupby('estrato').size() < 1 / frac).sum()}"
        " con menos de 1/frac usuarios, donde el redondeo puede dar 0 o 1)",
        f"- Concentración en el top 10 % de películas: {top10(r):.2%} completo, {top10(s):.2%} submuestra",
        f"- Mayor diferencia en el reparto de géneros: **{max_dif_gen:.2f} pp**",
        f"- Ratings por usuario (mediana): {u_all.n_ratings.median():.0f} completo, {u_s.n_ratings.median():.0f} submuestra",
        "",
        "## Comparación",
        *filas,
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    marcar(TAREA, 0, 4, "leyendo parquet")
    r = pd.read_parquet(PROCESADOS / "ratings.parquet")
    m = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId", "generos"])
    G, nombres = matriz_generos(m)

    marcar(TAREA, 1, 4, "estratos de usuarios")
    u = estratos(r, G, nombres)
    u["estrato"] = u.genero_dominante + "|" + u.actividad_q.astype(str) + "|" + u.epoca.astype(str)

    marcar(TAREA, 2, 4, "muestreando usuarios")
    rng = np.random.default_rng(a.semilla)
    # Redondeo estocástico por estrato: en promedio se toma exactamente frac de cada estrato
    elegidos = []
    for _, grupo in u.groupby("estrato"):
        n = len(grupo) * a.frac
        k = int(n) + (rng.random() < n - int(n))
        elegidos.append(rng.choice(grupo.userId.to_numpy(), size=k, replace=False))
    sel = np.concatenate(elegidos)
    s = r[r.userId.isin(sel)].reset_index(drop=True)
    u_s = u[u.userId.isin(sel)]

    marcar(TAREA, 3, 4, "guardando e informe")
    sufijo = f"{round(a.frac * 100)}pct"
    s.to_parquet(PROCESADOS / f"ratings_dev_{sufijo}.parquet", index=False)
    u[["userId", "genero_dominante", "actividad_q", "epoca", "n_ratings"]].to_parquet(
        PROCESADOS / "usuarios_estratos.parquet", index=False)
    INFORMES.mkdir(exist_ok=True)
    (INFORMES / "submuestra.md").write_text(comparar(r, s, G, nombres, u, u_s, a.frac, a.semilla), encoding="utf-8")
    marcar(TAREA, 4, 4, "hecho")
    print(f"{len(sel)} usuarios, {len(s)} ratings -> ratings_dev_{sufijo}.parquet")


if __name__ == "__main__":
    main()
