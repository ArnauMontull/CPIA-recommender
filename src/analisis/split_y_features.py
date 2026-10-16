"""Bloques D (split) y E (features de la torre de ítem): mide en la submuestra cómo se comporta cada opción.

Escribe reports/split_y_features.md y CSV en reports/split_y_features/.
Splits comparados (80/10/10): aleatorio por rating, último 10 %+10 % de cada usuario, corte temporal global
y por usuarios (contexto 80 % de la historia, objetivo el último 20 %). La definición de positivo es la
provisional de C2 (r >= min(4,0; media_u) con media_u solo de la historia visible); todo «por usuario» o
«por película» se calcula solo con train.

Uso: python -m src.analisis.split_y_features [--muestra ratings_dev_10pct.parquet] [--semilla 42]
"""
from __future__ import annotations

import argparse
import ast
import json

import numpy as np
import pandas as pd

from src.analisis.senal_objetivo import md, num
from src.preprocesamiento.comun import CRUDOS_TMDB, DATA, INFORMES, PROCESADOS, marcar

SALIDA = INFORMES / "split_y_features"
TAREA = "CPIA split y features"
DIA = 86400
KS = (10, 20, 50, 200)


def split_aleatorio(r: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(len(r))
    return np.where(u < 0.8, "train", np.where(u < 0.9, "val", "test"))


def split_ultimo(r: pd.DataFrame, frac: float = 0.1) -> tuple[np.ndarray, pd.DataFrame]:
    """r ordenado por usuario, tiempo, movieId. Último frac -> test, el anterior frac -> val."""
    n = r.groupby("userId").userId.transform("size").to_numpy()
    rango = r.groupby("userId").cumcount().to_numpy()
    n_te = np.ceil(frac * n).astype(int)
    lab = np.where(rango >= n - n_te, "test", np.where(rango >= n - 2 * n_te, "val", "train"))
    # ¿el corte cae dentro de un empate de segundo?
    ts = r.timestamp.to_numpy()
    prev = np.r_[0, ts[:-1]]
    mismo_seg = (ts == prev) & (rango > 0)  # igual que el rating anterior del mismo usuario
    corte_tv = rango == n - 2 * n_te       # primer val
    corte_vt = rango == n - n_te           # primer test
    emp = pd.DataFrame({"userId": r.userId.to_numpy(), "tv": corte_tv & mismo_seg, "vt": corte_vt & mismo_seg})
    return lab, emp.groupby("userId").any()


def split_global(r: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    c1, c2 = np.quantile(r.timestamp, [0.8, 0.9])
    ts = r.timestamp.to_numpy()
    return np.where(ts <= c1, "train", np.where(ts <= c2, "val", "test")), int(c1), int(c2)


def split_usuarios(r: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """80 % de usuarios a train; 10 % val; 10 % test. De los de val/test: el primer 80 % de la historia es
    contexto (entrada de la torre de usuario) y el último 20 % es lo que hay que predecir."""
    usuarios = r.userId.unique()
    g = pd.Series(rng.choice(["train", "val", "test"], size=len(usuarios), p=[0.8, 0.1, 0.1]), index=usuarios)
    grupo = r.userId.map(g).to_numpy()
    n = r.groupby("userId").userId.transform("size").to_numpy()
    rango = r.groupby("userId").cumcount().to_numpy()
    objetivo = rango >= n - np.ceil(0.2 * n).astype(int)
    return np.where(grupo == "train", "train", np.where(objetivo, grupo, "ctx"))


def evaluar(r: pd.DataFrame, lab: np.ndarray, t0: pd.Series) -> dict:
    """Mide un split. Devuelve un diccionario con las cifras de la tabla."""
    r = r.assign(split=lab)
    N = len(r)
    hist = r[r.split.isin(["train", "ctx"])]
    mu_hist = hist.groupby("userId").rating.mean()
    thr = np.minimum(4.0, r.userId.map(mu_hist)).fillna(4.0)
    r["pos"] = (r.rating >= thr).to_numpy()
    r["pos4"] = r.rating >= 4.0
    tr = r[r.split == "train"]
    n_tr_item = tr.movieId.value_counts()
    r["n_tr_item"] = r.movieId.map(n_tr_item).fillna(0)
    last_hist = hist.groupby("userId").timestamp.max()
    r["d"] = r.timestamp - r.userId.map(last_hist)
    r["dia1"] = (r.timestamp - r.userId.map(t0)) < DIA
    users_tr = set(tr.userId)
    d: dict = {"N": N}
    for s in ("train", "val", "test"):
        x = r[r.split == s]
        d[f"n_{s}"] = len(x)
        d[f"u_{s}"] = x.userId.nunique()
        d[f"dia1_{s}"] = 100 * x.dia1.mean() if len(x) else np.nan
        d[f"pos_{s}"] = 100 * x.pos.mean() if len(x) else np.nan
    for s in ("val", "test"):
        x = r[r.split == s]
        sin_train = ~x.userId.isin(users_tr)
        d[f"usr_sin_train_{s}"] = x[sin_train].userId.nunique()
        d[f"rat_sin_train_{s}"] = 100 * sin_train.mean()
        d[f"item_0_{s}"] = 100 * (x.n_tr_item == 0).mean()
        d[f"item_0n_{s}"] = x[x.n_tr_item == 0].movieId.nunique()
        d[f"item_lt5_{s}"] = 100 * (x.n_tr_item < 5).mean()
        d[f"item_lt20_{s}"] = 100 * (x.n_tr_item < 20).mean()
        con = x[~sin_train | (x.userId.isin(hist.userId))]
        dd = con.d
        d[f"antes_{s}"] = 100 * (dd < 0).mean()
        d[f"mismodia_{s}"] = 100 * ((dd >= 0) & (dd < DIA)).mean()
        d[f"despues_{s}"] = 100 * (dd >= DIA).mean()
        # usuarios elegibles: >=1 positivo en s y >=1 positivo en la historia visible
        hp = hist[r.loc[hist.index, "pos"]].groupby("userId").size()
        for nombre, col in (("c2", "pos"), ("r4", "pos4")):
            pu = x[x[col]].groupby("userId").size()
            if nombre == "c2":
                hpos = hp
            else:
                hpos = hist[hist.rating >= 4].groupby("userId").size()
            elig = pu[pu.index.isin(hpos.index)]
            d[f"elig_{nombre}_{s}"] = len(elig)
            d[f"elig_{nombre}_pct_{s}"] = 100 * len(elig) / max(1, x.userId.nunique())
            d[f"sinpos_{nombre}_{s}"] = x.userId.nunique() - len(elig)
            if nombre == "c2":
                d[f"med_pos_{s}"] = elig.median() if len(elig) else np.nan
                d[f"lt3_{s}"] = 100 * (elig < 3).mean() if len(elig) else np.nan
                d[f"pos_tot_{s}"] = int(elig.sum())
                for K in KS:
                    d[f"techo{K}_{s}"] = 100 * (np.minimum(K, elig) / elig).mean() if len(elig) else np.nan
    # P(i) con train: películas con positivos en test y cero en train
    pos_tr = r[(r.split == "train") & r.pos].movieId.value_counts()
    xt = r[(r.split == "test") & r.pos]
    d["pos_test_item_p0"] = 100 * (~xt.movieId.isin(pos_tr.index)).mean() if len(xt) else np.nan
    d["items_p0_train"] = int(r.movieId.nunique() - len(pos_tr))
    return d


def seccion_features(out: list[str], r: pd.DataFrame, lab_ult: np.ndarray) -> None:
    m = pd.read_parquet(PROCESADOS / "movies.parquet")
    M = len(m)
    en_muestra = m.movieId.isin(r.movieId.unique())
    nr = r.movieId.value_counts()
    peso = m.movieId.map(nr).fillna(0)  # ratings de la muestra por película
    TOT = peso.sum()

    def cob(mask: pd.Series) -> tuple[str, str]:
        return num(100 * mask.mean(), 1), num(100 * peso[mask].sum() / TOT, 1)

    g_ml = m.generos.map(len) > 0
    g_tm = m.generos_tmdb.map(lambda v: 0 if v is None else len(v)) > 0
    sinop = m.tiene_sinopsis
    anio_ok = m.anio.notna()

    # --- créditos y keywords ---
    marcar(TAREA, 6, 8, "leyendo credits.csv")
    ids = set(m.tmdbId.dropna().astype(int))
    cr = pd.read_csv(CRUDOS_TMDB / "credits.csv")
    cr["id"] = pd.to_numeric(cr.id, errors="coerce")
    cr = cr[cr.id.isin(ids)].drop_duplicates("id")
    dire: dict[int, str] = {}
    cast3: dict[int, list[str]] = {}
    for tid, c, w in zip(cr.id.astype(int), cr.cast, cr.crew):
        try:
            cl, wl = ast.literal_eval(c), ast.literal_eval(w)
        except (ValueError, SyntaxError):
            continue
        ds = [p["name"] for p in wl if p.get("job") == "Director"]
        if ds:
            dire[tid] = ds[0]
        cs = [p["name"] for p in sorted(cl, key=lambda p: p.get("order", 99))[:3]]
        if cs:
            cast3[tid] = cs
    kw = pd.read_csv(CRUDOS_TMDB / "keywords.csv")
    kw["id"] = pd.to_numeric(kw.id, errors="coerce")
    kw = kw[kw.id.isin(ids)].drop_duplicates("id")
    kw_ids = {int(i) for i, k in zip(kw.id, kw.keywords) if isinstance(k, str) and len(k) > 5}
    api = DATA / "external" / "tmdb_api.jsonl"
    if api.exists():
        for linea in api.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if j.get("estado") == 200 and j.get("keywords"):
                kw_ids.add(int(j["tmdbId"]))
    tid = m.tmdbId
    tiene_dir = tid.map(lambda v: pd.notna(v) and int(v) in dire)
    tiene_cast = tid.map(lambda v: pd.notna(v) and int(v) in cast3)
    tiene_kw = tid.map(lambda v: pd.notna(v) and int(v) in kw_ids)

    filas = [
        ["Géneros de MovieLens (no vacíos)", *cob(g_ml)],
        ["Géneros de TMDB (no vacíos)", *cob(g_tm)],
        ["Géneros de alguna de las dos", *cob(g_ml | g_tm)],
        ["Sinopsis", *cob(sinop)],
        ["Director (credits.csv)", *cob(tiene_dir)],
        ["Reparto (credits.csv)", *cob(tiene_cast)],
        ["Palabras clave (keywords.csv o API)", *cob(tiene_kw)],
        ["Año de estreno", *cob(anio_ok)],
    ]
    out += [
        "## E1. Cobertura de cada feature de la torre de ítem", "",
        f"Catálogo: {num(M)} películas ({num(en_muestra.sum())} con algún rating en la submuestra). "
        "«% ratings» = ponderado por los ratings de la submuestra.", "",
        md(pd.DataFrame(filas, columns=["Feature", "% películas", "% ratings"])), "",
    ]

    # Cobertura por época: lo que pasa tras 2017 (Kaggle)
    ep = pd.cut(m.anio.astype("float"), [0, 1989, 1999, 2009, 2016, 2100],
                labels=["≤1989", "1990-99", "2000-09", "2010-16", "≥2017"])
    fil = []
    for e in ep.cat.categories:
        s = ep == e
        fil.append([e, num(s.sum()), num(100 * tiene_dir[s].mean(), 1), num(100 * tiene_cast[s].mean(), 1),
                    num(100 * tiene_kw[s].mean(), 1), num(100 * g_tm[s].mean(), 1), num(100 * sinop[s].mean(), 1),
                    num(m.n_ratings[s].median(), 0)])
    out += [
        "### Cobertura por época de estreno", "",
        md(pd.DataFrame(fil, columns=["Época", "Películas", "% con director", "% con reparto", "% con keywords",
                                      "% con géneros TMDB", "% con sinopsis", "Mediana de ratings (completo)"])), "",
        f"Fuente de la sinopsis: {', '.join(f'{k} = {num(v)}' for k, v in m.fuente_sinopsis.value_counts().items())}.", "",
    ]

    # --- B3: géneros ML vs TMDB ---
    marcar(TAREA, 7, 8, "géneros")
    eq = {"Science Fiction": "Sci-Fi", "Family": "Children"}
    comunes = {"Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Horror",
               "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Children"}
    gm = m.generos.map(lambda v: set(v))
    gt = m.generos_tmdb.map(lambda v: {eq.get(x, x) for x in (v if v is not None else [])})
    ambos = g_ml & g_tm
    jac = [len(a & b & comunes) / max(1, len((a | b) & comunes)) for a, b in zip(gm[ambos], gt[ambos])]
    iguales = np.mean([(a & comunes) == (b & comunes) for a, b in zip(gm[ambos], gt[ambos])])
    vocab_ml = sorted({x for s in gm for x in s})
    vocab_tm = sorted({x for s in gt for x in s})
    vocab_tm_raw = sorted({x for v in m.generos_tmdb for x in (v if v is not None else [])})
    solo_ml_g = m[g_ml & ~g_tm]
    solo_tm_g = m[~g_ml & g_tm]
    out += [
        "## E2. Géneros de MovieLens frente a los de TMDB (decisión B3)", "",
        f"- Géneros distintos en el catálogo filtrado: MovieLens {len(vocab_ml)}, TMDB {len(vocab_tm_raw)}.",
        f"- Géneros por película (media): MovieLens {num(m.generos.map(len)[g_ml].mean(), 2)}, "
        f"TMDB {num(m.generos_tmdb.map(lambda v: len(v) if v is not None else 0)[g_tm].mean(), 2)}.",
        f"- Películas con géneros en MovieLens y no en TMDB: {num(len(solo_ml_g))}; en TMDB y no en MovieLens: {num(len(solo_tm_g))}; "
        f"en ninguna: {num((~g_ml & ~g_tm).sum())}.",
        f"- En las {num(ambos.sum())} películas con las dos listas, sobre los 16 géneros comunes (Sci-Fi = Science Fiction, "
        f"Children ≈ Family): Jaccard medio **{num(np.mean(jac), 2)}**; listas idénticas en el **{num(100 * iguales, 1)} %**.",
        f"- Películas con IMAX, Film-Noir o Musical (solo ML): {num(m.generos.map(lambda v: bool({'IMAX', 'Film-Noir', 'Musical'} & set(v))).sum())}; "
        f"con Music, History, Foreign o TV Movie (solo TMDB): "
        f"{num(m.generos_tmdb.map(lambda v: bool({'Music', 'History', 'Foreign', 'TV Movie'} & set(v if v is not None else []))).sum())}.", "",
    ]
    return m, tiene_dir, tiene_cast, dire, cast3, tid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", default="ratings_dev_10pct.parquet")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    pasos = 8
    marcar(TAREA, 0, pasos, "leyendo")
    r = pd.read_parquet(PROCESADOS / a.muestra).sort_values(["userId", "timestamp", "movieId"], ignore_index=True)
    N, U = len(r), r.userId.nunique()
    t0 = r.groupby("userId").timestamp.min()
    out = [
        "# Split (bloque D) y features de la torre de ítem (bloque E)", "",
        f"Generado por `python -m src.analisis.split_y_features --muestra {a.muestra} --semilla {a.semilla}`. Submuestra: "
        f"{num(N)} ratings, {num(U)} usuarios, {num(r.movieId.nunique())} películas.",
        "Todos los splits son 80/10/10. «Positivo» = definición provisional de C2: r ≥ min(4,0; media_u), con media_u de la historia "
        "visible (train, y contexto en el split por usuarios). «Elegible» = usuario con ≥ 1 positivo en la partición evaluada y ≥ 1 en su historia visible.", "",
    ]

    rng = np.random.default_rng(a.semilla)
    marcar(TAREA, 1, pasos, "splits")
    lab_ale = split_aleatorio(r, rng)
    lab_ult, empates = split_ultimo(r)
    lab_glob, c1, c2 = split_global(r)
    lab_usr = split_usuarios(r, rng)
    splits = {"Aleatorio por rating": lab_ale, "Último 10 % + 10 % por usuario": lab_ult,
              "Temporal global": lab_glob, "Por usuarios (contexto 80 % / objetivo 20 %)": lab_usr}

    marcar(TAREA, 2, pasos, "midiendo splits")
    res = {k: evaluar(r, v, t0) for k, v in splits.items()}
    pd.DataFrame(res).T.to_csv(SALIDA / "splits.csv", float_format="%.4g")
    nombres = list(res)

    def fila(etq, clave, d=0, pct=False):
        return [etq] + [num(res[k][clave], d) for k in nombres]

    cols = ["Medida", "Aleatorio", "Último por usuario", "Temporal global", "Por usuarios"]
    out += ["## D1. Tamaño de las particiones", "",
            md(pd.DataFrame([
                fila("Ratings en train", "n_train"), fila("Ratings en val", "n_val"), fila("Ratings en test", "n_test"),
                [("Ratings de contexto (no entran en la pérdida)")] + [num((lab == "ctx").sum()) for lab in splits.values()],
                fila("Usuarios con ratings en train", "u_train"), fila("Usuarios con ratings en val", "u_val"),
                fila("Usuarios con ratings en test", "u_test"),
            ], columns=cols)), "",
            f"Corte temporal global: train hasta {pd.to_datetime(c1, unit='s').date()}, val hasta {pd.to_datetime(c2, unit='s').date()}, "
            f"test después.", ""]

    out += ["## D2. Arranque en frío en cada split", "",
            md(pd.DataFrame([
                fila("Usuarios de test sin ningún rating en train", "usr_sin_train_test"),
                fila("% de ratings de test de usuarios sin train", "rat_sin_train_test", 1),
                fila("% de ratings de test en películas sin ningún rating en train", "item_0_test", 2),
                fila("Películas distintas sin train que salen en test", "item_0n_test"),
                fila("% de ratings de test en películas con < 5 en train", "item_lt5_test", 2),
                fila("% de ratings de test en películas con < 20 en train", "item_lt20_test", 2),
                fila("Películas sin ningún positivo en train (P(i) = 0)", "items_p0_train"),
                fila("% de positivos de test en películas sin positivo en train", "pos_test_item_p0", 2),
            ], columns=cols)), ""]

    out += ["## D3. ¿Test está en el futuro del usuario? (tiempo)", "",
            "Para los ratings de test (y val) con historia visible: posición respecto al último rating de la historia visible del mismo usuario.", "",
            md(pd.DataFrame([
                fila("% de ratings de test ANTES del último rating visible (ve el futuro)", "antes_test", 1),
                fila("% de ratings de test el mismo día (< 24 h después)", "mismodia_test", 1),
                fila("% de ratings de test más de 24 h después", "despues_test", 1),
                fila("% de ratings de train hechos en las primeras 24 h del usuario", "dia1_train", 1),
                fila("% de ratings de test hechos en las primeras 24 h del usuario", "dia1_test", 1),
            ], columns=cols)), ""]
    n_us = len(empates)
    out += [
        "### Último por usuario: empates en el corte", "",
        f"- Usuarios cuyo corte train/val cae dentro de un empate de segundo: {num(empates.tv.sum())} ({num(100 * empates.tv.mean(), 1)} %); "
        f"corte val/test: {num(empates.vt.sum())} ({num(100 * empates.vt.mean(), 1)} %); alguno de los dos: "
        f"{num((empates.tv | empates.vt).sum())} ({num(100 * (empates.tv | empates.vt).mean(), 1)} %). En ellos «el último» depende del desempate (aquí, movieId).", "",
    ]
    # usuarios totalmente en una partición (global)
    gl = pd.DataFrame({"u": r.userId, "s": lab_glob}).groupby("u").s.agg(lambda s: tuple(sorted(set(s))))
    vc = gl.value_counts()
    out += ["### Temporal global: en qué particiones aparece cada usuario", "",
            md(pd.DataFrame({"Particiones": [" + ".join(k) for k in vc.index], "Usuarios": [num(v) for v in vc.values],
                             "%": [num(100 * v / n_us, 1) for v in vc.values]})), ""]

    out += ["## D4. Usuarios elegibles y techo de las métricas (test)", "",
            md(pd.DataFrame([
                fila("Usuarios de test con ≥ 1 positivo en test y en su historia (C2)", "elig_c2_test"),
                fila("  % de los usuarios de test", "elig_c2_pct_test", 1),
                fila("Usuarios de test que salen de las métricas (C8, C2)", "sinpos_c2_test"),
                fila("Usuarios elegibles con r ≥ 4,0 fijo (C7)", "elig_r4_test"),
                fila("Positivos totales en test de los elegibles", "pos_tot_test"),
                fila("Positivos por usuario elegible (mediana)", "med_pos_test"),
                fila("% de elegibles con < 3 positivos en test", "lt3_test", 1),
                fila("Techo de Recall@10 (%)", "techo10_test", 1),
                fila("Techo de Recall@20 (%)", "techo20_test", 1),
                fila("Techo de Recall@50 (%)", "techo50_test", 1),
                fila("Techo de Recall@200 (%)", "techo200_test", 1),
            ], columns=cols)), "",
            "Techo de Recall@K = media de min(K, |R_u|)/|R_u| sobre los usuarios elegibles: el Recall@K máximo alcanzable aunque el modelo fuera perfecto.", ""]

    # Fuga: media_u con todo vs solo train (split último por usuario)
    marcar(TAREA, 3, pasos, "fuga de información")
    x = r.assign(split=lab_ult)
    mu_all = x.groupby("userId").rating.transform("mean")
    mu_tr = x[x.split == "train"].groupby("userId").rating.mean()
    mu_tr_u = x.userId.map(mu_tr)
    thr_all, thr_tr = np.minimum(4.0, mu_all), np.minimum(4.0, mu_tr_u).fillna(4.0)
    trn = x.split == "train"
    flip = ((x.rating >= thr_all) != (x.rating >= thr_tr))
    dif = (mu_all - mu_tr_u).abs()
    pos_tr_i = x[trn & (x.rating >= thr_tr)].movieId.value_counts()
    pos_all_i = x[(x.rating >= thr_all)].movieId.value_counts()
    com = pos_all_i.index.intersection(pos_tr_i.index)
    rho = pd.Series(np.log(pos_all_i[com])).rank().corr(pd.Series(np.log(pos_tr_i[com])).rank())
    out += [
        "## D5. Cuánta fuga habría (split «último por usuario»)", "",
        f"- Media de usuario con todo el historial frente a solo train: diferencia absoluta media {num(dif[trn].mean(), 3)} "
        f"(p95 {num(dif[trn].quantile(0.95), 3)}, máx. {num(dif[trn].max(), 2)}).",
        f"- Ratings de train cuya etiqueta de positivo cambia si media_u usa todo el historial: **{num(100 * flip[trn].mean(), 2)} %** "
        f"({num(flip[trn].sum())} ratings); en test: {num(100 * flip[x.split == 'test'].mean(), 2)} %.",
        f"- Películas con algún positivo en la muestra completa pero ninguno en train: {num(len(pos_all_i.index.difference(pos_tr_i.index)))} "
        f"(con P(i) calculada con todo se les daría un P(i) que solo existe gracias a test). Spearman de log P(i) entre ambos conteos: {num(rho, 3)}.", "",
    ]

    # Semillas
    marcar(TAREA, 4, pasos, "semillas")
    filas, rows = [], []
    for sd in range(5):
        rg = np.random.default_rng(1000 + sd)
        for nombre, lab in (("Aleatorio por rating", split_aleatorio(r, rg)), ("Por usuarios", split_usuarios(r, rg))):
            d = evaluar(r, lab, t0)
            rows.append((nombre, sd, d["n_test"], d["u_test"], d["elig_c2_test"], d["pos_tot_test"], d["item_lt5_test"], d["item_0_test"]))
    ds = pd.DataFrame(rows, columns=["split", "semilla", "n_test", "u_test", "elig", "pos_tot", "item_lt5", "item_0"])
    ds.to_csv(SALIDA / "semillas.csv", index=False)
    for nombre, g in ds.groupby("split", sort=False):
        filas.append([nombre] + [f"{num(g[c].mean(), 1 if c.startswith('item') else 0)} ± {num(g[c].std(), 1 if c.startswith('item') else 0)}"
                                 for c in ("n_test", "u_test", "elig", "pos_tot", "item_lt5", "item_0")])
    out += [
        "## D6. Variación entre semillas (5 semillas, media ± desviación típica)", "",
        md(pd.DataFrame(filas, columns=["Split", "Ratings de test", "Usuarios de test", "Elegibles", "Positivos en test",
                                        "% test en películas con < 5 en train", "% test en películas sin train"])), "",
    ]

    # Submuestra frente a completo: cortes temporales
    marcar(TAREA, 5, pasos, "completo")
    full = pd.read_parquet(PROCESADOS / "ratings.parquet", columns=["userId", "timestamp"])
    f1, f2 = np.quantile(full.timestamp, [0.8, 0.9])
    nfu = full.userId.nunique()
    out += [
        "## D7. Submuestra frente a conjunto completo", "",
        f"- Corte temporal global (cuantiles 80/90 de los timestamps): submuestra {pd.to_datetime(c1, unit='s').date()} / "
        f"{pd.to_datetime(c2, unit='s').date()}; completo {pd.to_datetime(f1, unit='s').date()} / {pd.to_datetime(f2, unit='s').date()}.",
        f"- Completo: {num(len(full))} ratings, {num(nfu)} usuarios. La submuestra tiene el {num(100 * N / len(full), 1)} % de los ratings.", "",
    ]
    del full

    seccion = seccion_features(out, r, lab_ult)
    m, tiene_dir, tiene_cast, dire, cast3, tid = seccion

    # E3: cardinalidades y películas con pocos positivos en train (split último por usuario)
    marcar(TAREA, 8, pasos + 1, "cardinalidades")
    dir_s = tid.map(lambda v: dire.get(int(v)) if pd.notna(v) else None)
    cnt_d = dir_s.value_counts()
    n_dir_5 = dir_s.map(cnt_d).fillna(0) >= 5
    actores = pd.Series([a for v in tid if pd.notna(v) for a in cast3.get(int(v), [])])
    cnt_a = actores.value_counts()
    tr = x[(x.split == "train")]
    pos_tr = tr[tr.rating >= np.minimum(4.0, tr.userId.map(mu_tr)).fillna(4.0)].movieId.value_counts()
    npos = m.movieId.map(pos_tr).fillna(0)
    ntr = m.movieId.map(tr.movieId.value_counts()).fillna(0)
    sinop_len = m.sinopsis.fillna("").str.split().map(len)
    frio = npos < 5
    fil = [["Todas", num(len(m)), num(100 * m.tiene_sinopsis.mean(), 1), num(100 * tiene_dir.mean(), 1),
            num(100 * (m.generos.map(len) > 0).mean(), 1)],
           ["< 20 ratings en train", num((ntr < 20).sum()), num(100 * m.tiene_sinopsis[ntr < 20].mean(), 1),
            num(100 * tiene_dir[ntr < 20].mean(), 1), num(100 * (m.generos.map(len) > 0)[ntr < 20].mean(), 1)],
           ["< 5 positivos en train", num(frio.sum()), num(100 * m.tiene_sinopsis[frio].mean(), 1),
            num(100 * tiene_dir[frio].mean(), 1), num(100 * (m.generos.map(len) > 0)[frio].mean(), 1)],
           ["0 positivos en train", num((npos == 0).sum()), num(100 * m.tiene_sinopsis[npos == 0].mean(), 1),
            num(100 * tiene_dir[npos == 0].mean(), 1), num(100 * (m.generos.map(len) > 0)[npos == 0].mean(), 1)]]
    tr_pos_total = int(pos_tr.sum())
    out += [
        "## E3. Películas poco vistas (split «último por usuario»; submuestra)", "",
        md(pd.DataFrame(fil, columns=["Grupo", "Películas", "% con sinopsis", "% con director", "% con géneros ML"])), "",
        "## E4. Cardinalidades y tamaño de los parámetros", "",
        f"- Directores distintos: {num(cnt_d.size)}; el {num(100 * n_dir_5[tid.notna()].mean(), 1)} % de las películas tiene un director con ≥ 5 películas en el catálogo.",
        f"- Actores distintos en el top-3 del reparto: {num(cnt_a.size)}; con ≥ 10 películas: {num((cnt_a >= 10).sum())} "
        f"({num(100 * cnt_a[cnt_a >= 10].sum() / cnt_a.sum(), 1)} % de las apariciones).",
        f"- Longitud de la sinopsis (palabras): p5 {num(sinop_len[m.tiene_sinopsis].quantile(0.05))}, mediana {num(sinop_len[m.tiene_sinopsis].median())}, "
        f"p95 {num(sinop_len[m.tiene_sinopsis].quantile(0.95))}; con < 10 palabras: {num(100 * (sinop_len[m.tiene_sinopsis] < 10).mean(), 1)} %. "
        "MiniLM (all-MiniLM-L6-v2) trunca a 256 *word pieces* (≈ 190 palabras).",
        f"- Tabla de embeddings de movieId (23 335 × 128): {num(len(m) * 128)} parámetros, frente a {num(tr_pos_total)} positivos de train en la submuestra "
        f"(≈ {num(tr_pos_total / (len(m) * 128), 2)} positivos por parámetro).",
        f"- Correlación de Spearman entre nº de ratings (completo) y año de estreno: {num(m.n_ratings.rank().corr(m.anio.astype('float').rank()), 2)}.", "",
    ]
    # Duración (runtime de movies_metadata.csv de Kaggle; la API de TMDB aporta `duracion`)
    mm = pd.read_csv(CRUDOS_TMDB / "movies_metadata.csv", usecols=["id", "runtime"], low_memory=False)
    mm["id"] = pd.to_numeric(mm.id, errors="coerce")
    rt = mm.dropna(subset=["id"]).drop_duplicates("id").set_index("id").runtime
    rt = pd.to_numeric(rt, errors="coerce")
    dur = tid.map(lambda v: rt.get(int(v)) if pd.notna(v) else np.nan)
    dur_ok = dur.notna() & (dur > 0)
    out += [
        f"- Duración (runtime de Kaggle): disponible en el {num(100 * dur_ok.mean(), 1)} % de las películas "
        f"(mediana {num(dur[dur_ok].median())} min, p5–p95 {num(dur[dur_ok].quantile(0.05))}–{num(dur[dur_ok].quantile(0.95))}); "
        f"las ≥ 2017 con duración: {num(100 * dur_ok[m.anio.astype('float') >= 2017].mean(), 1)} %.", "",
    ]
    dec = pd.cut(m.anio.astype("float"), [0, 1969, 1979, 1989, 1999, 2009, 2019, 2100],
                 labels=["≤1969", "70s", "80s", "90s", "00s", "10s", "≥2020"])
    td = m.assign(dec=dec).groupby("dec", observed=True).agg(peliculas=("movieId", "size"), mediana_ratings=("n_ratings", "median"))
    out += ["### Películas y mediana de ratings por década de estreno (completo)", "",
            md(pd.DataFrame({"Década": td.index.astype(str), "Películas": td.peliculas.map(num),
                             "Mediana de ratings": td.mediana_ratings.map(num)})), ""]

    marcar(TAREA, pasos + 1, pasos + 1, "escribiendo")
    (INFORMES / "split_y_features.md").write_text("\n".join(out), encoding="utf-8")
    print("Informe en", INFORMES / "split_y_features.md")


if __name__ == "__main__":
    main()
