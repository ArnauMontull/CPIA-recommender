"""Estadísticas descriptivas de la submuestra de desarrollo (para el explorador y el informe).

Lee data/processed/ratings_dev_10pct.parquet + movies.parquet y escribe CSV pequeños en reports/estadisticas/.

Uso: python -m src.analisis.estadisticas [--muestra ratings_dev_10pct.parquet] [--nodos 150]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.preprocesamiento.comun import INFORMES, PROCESADOS, marcar
from src.preprocesamiento.validar import a_generos_ml

SALIDA = INFORMES / "estadisticas"
TAREA = "CPIA estadísticas"


def guardar(df: pd.DataFrame, nombre: str) -> None:
    df.to_csv(SALIDA / f"{nombre}.csv", index=False, float_format="%.6g")


def genero_principal(m: pd.DataFrame) -> pd.Series:
    """El primer género de TMDB (TMDB los ordena por importancia) traducido a MovieLens.
    Sin TMDB, el primero de MovieLens que no sea IMAX."""
    def elegir(fila):
        for t in fila.generos_tmdb:
            ml = [g for g in a_generos_ml([t]) if g in fila.generos]
            if ml:
                return ml[0]
        gs = [g for g in fila.generos if g != "IMAX"]
        return gs[0] if gs else "(sin género)"
    return m.apply(elegir, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", default="ratings_dev_10pct.parquet")
    ap.add_argument("--nodos", type=int, default=150, help="películas en la red de co-visionado")
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    pasos = 12

    marcar(TAREA, 0, pasos, "leyendo")
    r = pd.read_parquet(PROCESADOS / a.muestra)
    m = pd.read_parquet(PROCESADOS / "movies.parquet")
    m = m[m.movieId.isin(r.movieId.unique())].copy()
    m["principal"] = genero_principal(m)
    r["anio_rating"] = pd.to_datetime(r.timestamp, unit="s").dt.year.astype("int16")
    por_peli = r.groupby("movieId").rating.agg(n="size", media="mean")
    por_user = r.groupby("userId").rating.agg(n="size", media="mean")
    m = m.join(por_peli, on="movieId")

    # 1. Resumen
    marcar(TAREA, 1, pasos, "resumen")
    n_sorted = np.sort(por_peli.n.to_numpy())[::-1]
    cum = np.cumsum(n_sorted) / n_sorted.sum()
    lor = np.concatenate([[0], np.cumsum(np.sort(por_peli.n.to_numpy())) / por_peli.n.sum()])
    gini = 1 - 2 * np.trapezoid(lor, dx=1 / (len(lor) - 1))
    top10 = cum[max(1, len(cum) // 10) - 1]
    guardar(pd.DataFrame([
        ("usuarios", r.userId.nunique()), ("ratings", len(r)), ("peliculas", len(m)),
        ("densidad_pct", 100 * len(r) / (r.userId.nunique() * len(m))),
        ("nota_media", r.rating.mean()), ("mediana_ratings_usuario", por_user.n.median()),
        ("mediana_ratings_pelicula", por_peli.n.median()), ("gini_peliculas", gini),
        ("pct_ratings_top10_peliculas", 100 * top10),
        ("primer_anio", r.anio_rating.min()), ("ultimo_anio", r.anio_rating.max()),
    ], columns=["metrica", "valor"]), "resumen")

    # 2. Notas
    marcar(TAREA, 2, pasos, "notas y tiempo")
    notas = r.rating.value_counts().sort_index()
    guardar(pd.DataFrame({"nota": notas.index, "ratings": notas.values, "pct": 100 * notas.values / len(r)}), "notas")

    # 3. Línea temporal
    t = r.groupby("anio_rating").agg(ratings=("rating", "size"), usuarios=("userId", "nunique"), nota_media=("rating", "mean"))
    guardar(t.reset_index().rename(columns={"anio_rating": "anio"}), "por_anio")

    # 4. Cola larga: curva de concentración (300 puntos) y rango-frecuencia (log)
    marcar(TAREA, 3, pasos, "cola larga")
    idx = np.unique(np.round(np.linspace(0, len(cum) - 1, 300)).astype(int))
    guardar(pd.DataFrame({"pct_peliculas": 100 * (idx + 1) / len(cum), "pct_ratings": 100 * cum[idx]}), "concentracion")
    rangos = np.unique(np.round(np.logspace(0, np.log10(len(n_sorted)), 200)).astype(int))
    guardar(pd.DataFrame({"rango": rangos, "ratings": n_sorted[rangos - 1]}), "rango_frecuencia")

    # 5. Actividad de usuarios (bins log2) y nota media por usuario
    marcar(TAREA, 4, pasos, "usuarios")
    bordes = 2 ** np.arange(np.floor(np.log2(por_user.n.min())), np.ceil(np.log2(por_user.n.max())) + 1)
    h, _ = np.histogram(por_user.n, bins=bordes)
    guardar(pd.DataFrame({"desde": bordes[:-1].astype(int), "hasta": bordes[1:].astype(int) - 1, "usuarios": h}), "actividad_usuarios")
    b = np.arange(0.5, 5.01, 0.25)
    h, _ = np.histogram(por_user.media, bins=b)
    guardar(pd.DataFrame({"desde": b[:-1], "hasta": b[1:], "usuarios": h}), "nota_media_usuarios")

    # 6. Géneros: películas, cuota de ratings (1/k por género), nota media, usuarios con ese género dominante
    marcar(TAREA, 5, pasos, "géneros")
    ex = m[["movieId", "generos", "n", "media"]].explode("generos").dropna(subset=["generos"])
    ex["k"] = ex.movieId.map(m.set_index("movieId").generos.map(len))
    ex["peso"] = ex.n / ex.k
    rg = r.merge(ex[["movieId", "generos", "k"]], on="movieId")
    g = pd.DataFrame({
        "peliculas": ex.groupby("generos").size(),
        "pct_ratings": 100 * ex.groupby("generos").peso.sum() / len(r),
        "nota_media": rg.groupby("generos").rating.mean(),
    })
    estr = pd.read_parquet(PROCESADOS / "usuarios_estratos.parquet")
    estr = estr[estr.userId.isin(r.userId.unique())]
    g["usuarios_dominante"] = estr.genero_dominante.value_counts().reindex(g.index).fillna(0).astype(int)
    guardar(g.reset_index(names="genero").sort_values("pct_ratings", ascending=False), "generos")

    # 7. Co-ocurrencia de géneros en el catálogo (lift = P(a,b) / P(a)P(b))
    marcar(TAREA, 6, pasos, "red de géneros")
    nombres = sorted(g.index)
    M = np.array([[gg in gs for gg in nombres] for gs in m.generos], dtype=float)
    co = M.T @ M
    p = M.mean(0)
    filas = []
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            if co[i, j] >= 20:
                filas.append((nombres[i], nombres[j], int(co[i, j]), co[i, j] / len(M) / (p[i] * p[j])))
    guardar(pd.DataFrame(filas, columns=["a", "b", "peliculas", "lift"]), "generos_coocurrencia")

    # 8. Género × década de estreno (cuota de ratings dentro de cada década)
    marcar(TAREA, 7, pasos, "décadas")
    rg["decada"] = (rg.movieId.map(m.set_index("movieId").anio).astype("float") // 10 * 10)
    rg = rg.dropna(subset=["decada"])
    rg = rg[rg.decada >= 1920]
    gd = rg.assign(w=1 / rg.k).groupby(["decada", "generos"]).w.sum()
    gd = (100 * gd / gd.groupby(level=0).transform("sum")).rename("pct").reset_index()
    gd["decada"] = gd.decada.astype(int)
    guardar(gd.rename(columns={"generos": "genero"}), "genero_decada")

    # 9. Desfase entre estreno y rating
    marcar(TAREA, 8, pasos, "desfase")
    lag = (r.anio_rating - r.movieId.map(m.set_index("movieId").anio).astype("float")).dropna()
    lag = lag.clip(-1, 60).astype(int).value_counts().sort_index()
    guardar(pd.DataFrame({"anios": lag.index, "ratings": lag.values}), "desfase_estreno")

    # 10. Popularidad frente a nota (películas con ≥ 30 ratings en la muestra) y ranking
    marcar(TAREA, 9, pasos, "popularidad y top")
    C, mu = 50, r.rating.mean()
    m["bayes"] = (m.n * m.media + C * mu) / (m.n + C)
    pv = m[m.n >= 30][["movieId", "titulo", "anio", "principal", "n", "media", "bayes"]]
    guardar(pv.rename(columns={"n": "ratings", "media": "nota_media", "principal": "genero"}), "popularidad_nota")
    top = pd.concat([
        m.nlargest(15, "n").assign(lista="mas_vistas"),
        m[m.n >= 100].nlargest(15, "bayes").assign(lista="mejor_valoradas"),
        m[m.n >= 100].nsmallest(15, "bayes").assign(lista="peor_valoradas"),
    ])
    top["puesto"] = top.groupby("lista").cumcount() + 1
    top["clave"] = top.lista + "-" + top.puesto.astype(str)
    guardar(top[["clave", "lista", "puesto", "titulo", "anio", "principal", "n", "media", "bayes"]]
            .rename(columns={"n": "ratings", "media": "nota_media", "principal": "genero"}), "top_peliculas")

    # 11. Red de co-visionado: las N más vistas, similitud coseno entre sus conjuntos de usuarios, 4 vecinos por nodo
    marcar(TAREA, 10, pasos, "red de películas")
    nodos = m.nlargest(a.nodos, "n").reset_index(drop=True)
    col = {mid: i for i, mid in enumerate(nodos.movieId)}
    sub = r[r.movieId.isin(col)]
    users = {u: i for i, u in enumerate(sub.userId.unique())}
    X = np.zeros((len(users), len(nodos)), dtype=np.float32)
    X[sub.userId.map(users).to_numpy(), sub.movieId.map(col).to_numpy()] = 1
    co = X.T @ X
    n = np.diag(co).copy()
    cos = co / np.sqrt(np.outer(n, n))
    np.fill_diagonal(cos, 0)
    aristas = set()
    for i in range(len(nodos)):
        for j in np.argsort(-cos[i])[:4]:
            aristas.add((min(i, j), max(i, j)))
    guardar(pd.DataFrame([(int(nodos.movieId[i]), int(nodos.movieId[j]), float(cos[i, j])) for i, j in sorted(aristas)],
                         columns=["origen", "destino", "similitud"]), "red_aristas")
    guardar(nodos[["movieId", "titulo", "anio", "principal", "n", "media"]]
            .rename(columns={"n": "ratings", "media": "nota_media", "principal": "genero"}), "red_nodos")

    # 12. Procedencia de las sinopsis y estratos de usuarios
    marcar(TAREA, 11, pasos, "sinopsis y estratos")
    mm = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["fuente_sinopsis"])
    fs = mm.fuente_sinopsis.replace("", "sin sinopsis").value_counts()
    guardar(pd.DataFrame({"fuente": fs.index, "peliculas": fs.values}), "fuentes_sinopsis")
    e = estr.groupby(["epoca", "actividad_q"], observed=True).size().rename("usuarios").reset_index()
    e["actividad_q"] = e.actividad_q + 1
    guardar(e, "estratos")
    marcar(TAREA, pasos, pasos, "hecho")
    print("CSV en", SALIDA)


if __name__ == "__main__":
    main()
