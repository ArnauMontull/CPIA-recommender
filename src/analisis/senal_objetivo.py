"""Bloque C (señal objetivo): mide en la submuestra qué pasa con cada definición de «le gusta».

Escribe reports/senal_objetivo.md (tablas para el planning del bloque C) y CSV en reports/senal_objetivo/.
Las medidas que dependen del split (bloque D) usan un corte temporal provisional: el último 20 % de los
ratings de cada usuario hace de «test».

Uso: python -m src.analisis.senal_objetivo [--muestra ratings_dev_10pct.parquet] [--frac-test 0.2]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.preprocesamiento.comun import INFORMES, PROCESADOS, marcar

SALIDA = INFORMES / "senal_objetivo"
TAREA = "CPIA señal objetivo"


def num(v, d: int = 0) -> str:
    """Formato español: 31 725 154 y 0,86."""
    if pd.isna(v):
        return "—"
    s = f"{v:,.{d}f}"
    return s.replace(",", " ").replace(".", ",")


def md(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|---|" + "---:|" * (len(df.columns) - 1)  # primera columna = etiquetas, el resto = números
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", default="ratings_dev_10pct.parquet")
    ap.add_argument("--frac-test", type=float, default=0.2)
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    pasos = 10
    out: list[str] = []

    # 0. Lectura y columnas auxiliares ------------------------------------------------------------
    marcar(TAREA, 0, pasos, "leyendo")
    r = pd.read_parquet(PROCESADOS / a.muestra).sort_values(["userId", "timestamp", "movieId"], ignore_index=True)
    m = pd.read_parquet(PROCESADOS / "movies.parquet", columns=["movieId", "titulo", "anio"]).set_index("movieId")
    N = len(r)
    g = r.groupby("userId").rating
    r["n_u"] = g.transform("size")
    r["media_u"] = g.transform("mean")
    r["std_u"] = g.transform("std", ddof=0)
    r["n_i"] = r.groupby("movieId").rating.transform("size")
    usuarios = r.userId.unique()
    U = len(usuarios)
    # Popularidad: el 10 % de películas con más ratings en la muestra
    pop = r.movieId.value_counts()
    top10_items = set(pop.index[: max(1, len(pop) // 10)])
    r["top10"] = r.movieId.isin(top10_items)
    # Usuarios más activos (top 10 % por número de ratings)
    nu = r.groupby("userId").size()
    top10_users = set(nu.sort_values(ascending=False).index[: max(1, U // 10)])
    r["user_top10"] = r.userId.isin(top10_users)
    # Corte temporal provisional (bloque D): último frac_test de cada usuario
    rango = r.groupby("userId").cumcount()
    r["en_test"] = rango >= r.n_u - np.ceil(a.frac_test * r.n_u)
    r["ultimo"] = rango == r.n_u - 1
    # Sesgos con encogimiento (baseline de Koren): r ≈ μ + b_u + b_i
    mu = r.rating.mean()
    s = (r.rating - mu).groupby(r.userId).agg(["sum", "count"])
    r["b_u"] = r.userId.map(s["sum"] / (s["count"] + 10))
    s = (r.rating - mu - r.b_u).groupby(r.movieId).agg(["sum", "count"])
    r["b_i"] = r.movieId.map(s["sum"] / (s["count"] + 25))
    r["resid"] = r.rating - mu - r.b_u - r.b_i
    z = (r.rating - r.media_u) / r.std_u.replace(0, np.nan)

    out += [
        "# Señal objetivo: qué significa «le gusta» (bloque C)", "",
        f"Generado por `python -m src.analisis.senal_objetivo --muestra {a.muestra} --frac-test {a.frac_test}`. "
        f"Submuestra: {num(N)} ratings, {num(U)} usuarios, {num(r.movieId.nunique())} películas, nota media {num(mu, 3)}.",
        f"«Test» = último {num(100 * a.frac_test)} % de los ratings de cada usuario por fecha (corte provisional: lo decide el bloque D). "
        "«Top 10 %» = el 10 % de películas con más ratings en la muestra.", "",
    ]

    # 1. Sesgo de cada usuario --------------------------------------------------------------------
    marcar(TAREA, 1, pasos, "sesgo por usuario")
    pu = r.groupby("userId").agg(n=("rating", "size"), media=("rating", "mean"), std=("rating", lambda x: x.std(ddof=0)),
                                 medias_estrellas=("rating", lambda x: (x % 1 != 0).any()))
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    tq = pd.DataFrame({"Medida": ["media del usuario", "desviación típica del usuario"],
                       **{f"p{int(100 * k)}": [num(pu.media.quantile(k), 2), num(pu["std"].quantile(k), 2)] for k in q}})
    b = np.arange(0.5, 5.01, 0.25)
    h, _ = np.histogram(pu.media, bins=b)
    pd.DataFrame({"desde": b[:-1], "hasta": b[1:], "usuarios": h}).to_csv(SALIDA / "media_usuario.csv", index=False)
    corr_n_media = np.corrcoef(np.log(pu.n), pu.media)[0, 1]
    out += [
        "## 1. Sesgo de cada usuario", "", md(tq), "",
        f"- Usuarios con media < 3,0: **{num(100 * (pu.media < 3).mean(), 1)} %**; con media ≥ 4,0: **{num(100 * (pu.media >= 4).mean(), 1)} %**.",
        f"- Usuarios que ponen siempre la misma nota (desviación 0): {num((pu['std'] == 0).sum())}; con desviación < 0,5: "
        f"{num((pu['std'] < 0.5).sum())} ({num(100 * (pu['std'] < 0.5).mean(), 1)} %).",
        f"- Usuarios que nunca usan medias estrellas: {num((~pu.medias_estrellas).sum())} ({num(100 * (~pu.medias_estrellas).mean(), 1)} %).",
        f"- Correlación entre log(nº de ratings) y la media del usuario: {num(corr_n_media, 2)}.",
        "- Histograma: `reports/senal_objetivo/media_usuario.csv`.", "",
    ]

    # Medias estrellas por año (antes de 2003 MovieLens solo tenía estrellas enteras)
    anio = pd.to_datetime(r.timestamp, unit="s").dt.year
    pa = r.assign(anio=anio, media_estrella=r.rating % 1 != 0, p35=r.rating >= 3.5, p40=r.rating >= 4) \
          .groupby("anio").agg(ratings=("rating", "size"), nota_media=("rating", "mean"),
                               pct_medias=("media_estrella", "mean"), pct_ge35=("p35", "mean"), pct_ge40=("p40", "mean"))
    pa[["pct_medias", "pct_ge35", "pct_ge40"]] *= 100
    pa.reset_index().to_csv(SALIDA / "por_anio.csv", index=False, float_format="%.4g")
    primer_medio = int(pa[pa.pct_medias > 0].index.min())
    antes = r[anio < primer_medio]
    out += [
        f"- Primer año con medias estrellas: **{primer_medio}**. Ratings anteriores: {num(len(antes))} "
        f"({num(100 * len(antes) / N, 1)} %), de los que son 3 o 4 «enteros» el "
        f"{num(100 * antes.rating.isin([3, 4]).mean(), 1)} %. Por año: `reports/senal_objetivo/por_anio.csv`.", "",
    ]

    # 2. Umbrales de positivo ---------------------------------------------------------------------
    marcar(TAREA, 2, pasos, "umbrales")
    esquemas = {
        "r ≥ 3,0": r.rating >= 3.0,
        "r ≥ 3,5": r.rating >= 3.5,
        "r ≥ 4,0": r.rating >= 4.0,
        "r ≥ 4,5": r.rating >= 4.5,
        "r = 5,0": r.rating == 5.0,
        "r > media_u": r.rating > r.media_u,
        "r ≥ media_u": r.rating >= r.media_u,
        "r ≥ media_u + 0,5": r.rating >= r.media_u + 0.5,
        "z_u ≥ 0,5": (z >= 0.5).fillna(False),
        "r ≥ max(3,5; media_u)": r.rating >= np.maximum(3.5, r.media_u),
        "r ≥ min(4,0; media_u)": r.rating >= np.minimum(4.0, r.media_u),
        "residuo r − (μ+b_u+b_i) > 0": r.resid > 0,
    }
    filas, filas_csv = [], []
    for nombre, pos in esquemas.items():
        rp = r[pos.to_numpy()]
        pu_pos = rp.groupby("userId").size().reindex(usuarios, fill_value=0)
        pu_test = rp[rp.en_test].groupby("userId").size().reindex(usuarios, fill_value=0)
        pi = rp.groupby("movieId").size()
        d = dict(
            esquema=nombre, pct_positivos=100 * pos.mean(), positivos=int(pos.sum()),
            mediana_pos_usuario=pu_pos.median(), p10_pos_usuario=pu_pos.quantile(0.1),
            usuarios_0_pos=int((pu_pos == 0).sum()), usuarios_menos5_pos=int((pu_pos < 5).sum()),
            usuarios_0_pos_test=int((pu_test == 0).sum()), pct_ultimo_positivo=100 * pos[r.ultimo].mean(),
            peliculas_con_pos=len(pi), peliculas_20pos=int((pi >= 20).sum()),
            pct_pos_en_top10=100 * rp.top10.mean(), pct_pos_usuarios_top10=100 * rp.user_top10.mean(),
        )
        filas_csv.append(d)
        filas.append([nombre, num(d["pct_positivos"], 1), num(d["mediana_pos_usuario"]), num(d["p10_pos_usuario"]),
                      num(d["usuarios_0_pos"]), num(d["usuarios_menos5_pos"]), num(d["usuarios_0_pos_test"]),
                      num(d["pct_ultimo_positivo"], 1), num(d["peliculas_con_pos"]), num(d["peliculas_20pos"]),
                      num(d["pct_pos_en_top10"], 1)])
    pd.DataFrame(filas_csv).to_csv(SALIDA / "umbrales.csv", index=False, float_format="%.4g")
    out += [
        "## 2. Positivos con cada definición", "",
        md(pd.DataFrame(filas, columns=["Esquema", "% ratings", "Pos./usuario (mediana)", "p10", "Usuarios sin pos.",
                                        "Usuarios < 5 pos.", "Usuarios sin pos. en test", "% último rating positivo",
                                        "Películas con ≥ 1 pos.", "Películas con ≥ 20 pos.", "% pos. en top 10 %"])), "",
        f"Referencia: en todos los ratings, el top 10 % de películas reúne el {num(100 * r.top10.mean(), 1)} %; "
        f"el top 10 % de usuarios más activos aporta el {num(100 * r.user_top10.mean(), 1)} % de los ratings "
        "(columna `pct_pos_usuarios_top10` del CSV para cada esquema). Todo en `reports/senal_objetivo/umbrales.csv`.", "",
    ]

    # 3. Tasa de positivos según la popularidad de la película -------------------------------------
    marcar(TAREA, 3, pasos, "positivos por popularidad")
    orden = pop.index  # de más a menos ratings
    decil = pd.Series(np.minimum(9, np.arange(len(orden)) * 10 // len(orden)), index=orden)
    r["decil_pop"] = r.movieId.map(decil) + 1
    tp = r.assign(ge35=r.rating >= 3.5, ge40=r.rating >= 4, rel=r.rating > r.media_u) \
          .groupby("decil_pop").agg(ratings=("rating", "size"), nota_media=("rating", "mean"),
                                    pct_ge35=("ge35", "mean"), pct_ge40=("ge40", "mean"), pct_rel=("rel", "mean"))
    tp[["pct_ge35", "pct_ge40", "pct_rel"]] *= 100
    tp.reset_index().to_csv(SALIDA / "positivos_por_popularidad.csv", index=False, float_format="%.4g")
    m50 = r[r.n_i >= 50].groupby("movieId").agg(n=("rating", "size"), media=("rating", "mean"))
    out += [
        "## 3. Tasa de positivos según la popularidad de la película", "",
        "Decil 1 = el 10 % de películas más vistas; decil 10 = el 10 % menos vistas (por número de películas, no de ratings).", "",
        md(pd.DataFrame({"Decil": tp.index, "Ratings": tp.ratings.map(num), "Nota media": tp.nota_media.map(lambda v: num(v, 2)),
                         "% ≥ 3,5": tp.pct_ge35.map(lambda v: num(v, 1)), "% ≥ 4,0": tp.pct_ge40.map(lambda v: num(v, 1)),
                         "% > media_u": tp.pct_rel.map(lambda v: num(v, 1))})), "",
        f"Correlación entre log(nº de ratings) y nota media de la película (películas con ≥ 50 ratings): "
        f"**{num(np.corrcoef(np.log(m50.n), m50.media)[0, 1], 2)}**.", "",
    ]

    # 4. Notas bajas (candidatas a negativos explícitos) -----------------------------------------
    marcar(TAREA, 4, pasos, "notas bajas")
    negs = {
        "r ≤ 2,0": r.rating <= 2.0,
        "r ≤ 2,5": r.rating <= 2.5,
        "r ≤ 3,0": r.rating <= 3.0,
        "r ≤ media_u − 1": r.rating <= r.media_u - 1,
        "residuo < −1": r.resid < -1,
    }
    filas = []
    for nombre, neg in negs.items():
        rn = r[neg.to_numpy()]
        pn = rn.groupby("userId").size().reindex(usuarios, fill_value=0)
        filas.append([nombre, num(100 * neg.mean(), 1), num(len(rn)), num(100 * (pn >= 1).mean(), 1),
                      num(pn.median()), num(100 * (pn >= 5).mean(), 1), num(100 * rn.top10.mean(), 1)])
    out += [
        "## 4. Notas bajas", "",
        md(pd.DataFrame(filas, columns=["Definición", "% ratings", "Ratings", "% usuarios con ≥ 1", "Mediana por usuario",
                                        "% usuarios con ≥ 5", "% en top 10 %"])), "",
    ]

    # Negativos in-batch: con qué probabilidad la película muestreada (∝ frecuencia de positivos, como en un
    # lote) resulta ser un positivo del propio usuario (falso negativo), una nota baja suya, o algo que ya vio.
    pos40 = r.rating >= 4.0
    qpos = r[pos40].movieId.value_counts(normalize=True)
    r["q"] = r.movieId.map(qpos).fillna(0)
    prob = pd.DataFrame({
        "falso_negativo": r[pos40].groupby("userId").q.sum(),
        "nota_baja": r[r.rating <= 2.5].groupby("userId").q.sum(),
        "ya_visto": r.groupby("userId").q.sum(),
    }).reindex(usuarios).fillna(0)
    out += [
        "### Negativos del lote (*in-batch*) frente a negativos explícitos", "",
        "Si los negativos se muestrean como en un lote (proporcional a la frecuencia de positivos con r ≥ 4,0), "
        "probabilidad de que la película sacada para un usuario sea…", "",
        md(pd.DataFrame({"…": ["un positivo suyo (falso negativo)", "una nota baja suya (r ≤ 2,5)", "algo que ya valoró"],
                         "media por usuario (%)": [num(100 * prob[c].mean(), 2) for c in prob],
                         "mediana (%)": [num(100 * prob[c].median(), 2) for c in prob],
                         "p90 (%)": [num(100 * prob[c].quantile(0.9), 2) for c in prob]})), "",
    ]

    # 5. Ponderación de los positivos por la nota ----------------------------------------------------
    marcar(TAREA, 5, pasos, "pesos")
    rp = r[r.rating >= 3.5]
    pesos = {
        "w = 1": np.ones(len(rp)),
        "w = r − 2,5 (3,5→1 … 5→2,5)": rp.rating - 2.5,
        "w = 2^(r − 3,5) (1 … 2,83)": 2.0 ** (rp.rating - 3.5),
        "w = 1 + max(0, r − media_u)": 1 + np.maximum(0, rp.rating - rp.media_u),
        "w = 1/√n_u (iguala usuarios)": 1 / np.sqrt(rp.n_u),
        "w = (r − 2,5)/√n_u": (rp.rating - 2.5) / np.sqrt(rp.n_u),
    }
    filas = []
    for nombre, w in pesos.items():
        w = np.asarray(w, dtype=float)
        ess = w.sum() ** 2 / (len(w) * (w ** 2).sum())
        filas.append([nombre, num(100 * ess, 1), num(100 * w[rp.rating.to_numpy() == 5].sum() / w.sum(), 1),
                      num(100 * w[rp.top10.to_numpy()].sum() / w.sum(), 1), num(100 * w[rp.user_top10.to_numpy()].sum() / w.sum(), 1)])
    out += [
        "## 5. Ponderar los positivos (sobre positivos r ≥ 3,5)", "",
        "ESS = tamaño efectivo de muestra (Σw)²/(N·Σw²): 100 % = todos pesan igual; cuanto más bajo, más ruido en el gradiente.", "",
        md(pd.DataFrame(filas, columns=["Pesos", "ESS (%)", "% del peso en notas 5,0", "% del peso en top 10 % películas",
                                        "% del peso en top 10 % usuarios"])), "",
    ]

    # 6. logQ: popularidad contada con todos los ratings o solo con positivos -----------------------
    marcar(TAREA, 6, pasos, "logQ")
    filas = []
    n_all = r.movieId.value_counts()
    for nombre, pos in [("r ≥ 3,5", r.rating >= 3.5), ("r ≥ 4,0", r.rating >= 4.0), ("r > media_u", r.rating > r.media_u)]:
        n_pos = r[pos.to_numpy()].movieId.value_counts().reindex(n_all.index, fill_value=0)
        con = n_pos > 0
        rho = pd.Series(np.log(n_all[con])).rank().corr(pd.Series(np.log(n_pos[con])).rank())
        ratio = (n_pos / n_pos.sum()) / (n_all / n_all.sum())
        filas.append([nombre, num(rho, 3), num((~con).sum()), num(ratio[n_all >= 100].quantile(0.05), 2),
                      num(ratio[n_all >= 100].quantile(0.95), 2)])
    out += [
        "## 6. logQ: P(i) con todos los ratings o solo con los positivos", "",
        "Ratio = P_pos(i) / P_todos(i) para películas con ≥ 100 ratings: si fuera 1 siempre, daría igual cómo se cuente.", "",
        md(pd.DataFrame(filas, columns=["Positivos", "Spearman log P", "Películas con 0 positivos (P = 0)", "Ratio p5", "Ratio p95"])), "",
    ]
    # Películas populares cuyo rango cae más al contar solo positivos (r ≥ 4,0)
    n_pos = r[pos40].movieId.value_counts().reindex(n_all.index, fill_value=0)
    rk = pd.DataFrame({"rango_todos": n_all.rank(ascending=False, method="min"),
                       "rango_pos": n_pos.rank(ascending=False, method="min"), "ratings": n_all,
                       "pct_pos": 100 * n_pos / n_all}).head(300)
    rk["cambio"] = rk.rango_pos - rk.rango_todos
    ej = pd.concat([rk.nlargest(5, "cambio"), rk.nsmallest(5, "cambio")]).join(m)
    ej.reset_index().to_csv(SALIDA / "logq_ejemplos.csv", index=False, float_format="%.4g")
    out += [
        "Entre las 300 más vistas, las que más bajan (y suben) de rango al contar solo positivos r ≥ 4,0:", "",
        md(pd.DataFrame({"Película": ej.titulo + " (" + ej.anio.astype("string").fillna("?") + ")",
                         "Rango (todos)": ej.rango_todos.map(num), "Rango (positivos)": ej.rango_pos.map(num),
                         "% ≥ 4,0": ej.pct_pos.map(lambda v: num(v, 1))})), "",
    ]

    # 7. Tiempo: sesiones de valoración en bloque, empates de timestamp y ratings antes del estreno --------
    marcar(TAREA, 7, pasos, "tiempo")
    t0 = r.groupby("userId").timestamp.transform("min")
    dia1 = (r.timestamp - t0) < 86400
    frac_dia1 = dia1.groupby(r.userId).mean()
    empate = r.duplicated(["userId", "timestamp"], keep=False)
    ts_distintos = r.groupby("userId").timestamp.nunique()
    ae = r[r.antes_estreno]
    out += [
        "## 7. Tiempo", "",
        f"- Ratings hechos en las primeras 24 h del usuario: **{num(100 * dia1.mean(), 1)} %**. Usuarios con ≥ 50 % de sus ratings "
        f"el primer día: {num(100 * (frac_dia1 >= 0.5).mean(), 1)} %; con todos el primer día: {num(100 * (frac_dia1 == 1).mean(), 1)} %.",
        f"- Nota media el primer día: {num(r[dia1].rating.mean(), 3)}; después: {num(r[~dia1].rating.mean(), 3)}. "
        f"% ≥ 4,0: {num(100 * (r[dia1].rating >= 4).mean(), 1)} % frente a {num(100 * (r[~dia1].rating >= 4).mean(), 1)} %.",
        f"- Ratings que comparten segundo exacto con otro del mismo usuario: {num(100 * empate.mean(), 1)} % "
        f"(el orden temporal dentro del empate no existe). Instantes distintos por usuario (mediana): {num(ts_distintos.median())}.",
        f"- Ratings más de 1 año antes del estreno (flag `antes_estreno`): {num(len(ae))} en la muestra, nota media "
        f"{num(ae.rating.mean(), 2)}, {num(100 * (ae.rating >= 4).mean(), 1)} % ≥ 4,0.",
        "- Re-ratings: MovieLens solo guarda la última nota de cada par (usuario, película) (0 pares repetidos en B4); "
        "los únicos «re-ratings» fueron los 836 de duplicados fusionados (B1b), donde ya se quedó la más reciente.", "",
    ]

    # 8. Sesgo usuario vs película: positivos absolutos que el residuo considera negativos y viceversa -------
    marcar(TAREA, 8, pasos, "residuos")
    a40, res = r.rating >= 4, r.resid > 0
    out += [
        "## 8. Coincidencia entre definiciones", "",
        md(pd.DataFrame({
            "Comparación": ["r ≥ 4,0 pero residuo ≤ 0 (le gustó menos de lo esperable)",
                            "r < 4,0 pero residuo > 0 (le gustó más de lo esperable)",
                            "r ≥ 4,0 pero r ≤ media_u", "r < 4,0 pero r > media_u"],
            "% ratings": [num(100 * (a40 & ~res).mean(), 1), num(100 * (~a40 & res).mean(), 1),
                          num(100 * (a40 & (r.rating <= r.media_u)).mean(), 1), num(100 * (~a40 & (r.rating > r.media_u)).mean(), 1)],
        })), "",
        f"Sesgos con encogimiento (λ_u = 10, λ_i = 25): b_u entre {num(r.groupby('userId').b_u.first().quantile(0.05), 2)} y "
        f"{num(r.groupby('userId').b_u.first().quantile(0.95), 2)} (p5–p95); b_i entre "
        f"{num(r.groupby('movieId').b_i.first().quantile(0.05), 2)} y {num(r.groupby('movieId').b_i.first().quantile(0.95), 2)}.", "",
    ]

    marcar(TAREA, 9, pasos, "escribiendo")
    (INFORMES / "senal_objetivo.md").write_text("\n".join(out), encoding="utf-8")
    marcar(TAREA, pasos, pasos, "hecho")
    print("Informe en", INFORMES / "senal_objetivo.md")


if __name__ == "__main__":
    main()
