"""Etapa 3 (E1-E9): features de la torre de películas.

Entrada: movies.parquet, credits/keywords/movies_metadata de Kaggle, data/external/tmdb_api.jsonl (relleno, E9)
y split.parquet (solo para saber qué películas están en train: todo lo que se aprende sale de train, D3).
Salida (data/processed/<muestra>/): peliculas_features.parquet, vocab_creditos.json, vocab_generos.json,
meta_features.json. Los embeddings no dependen del split: data/processed/emb_texto.npy (float16 [N, 384]),
en el orden de item_idx, compartidos por dev y completo.

Uso: python -m src.preprocesamiento.features_peliculas --muestra {dev,completo} [--forzar] [--sin-embeddings]
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .comun import CRUDOS_TMDB, DATA, PROCESADOS, RAIZ, marcar
from .validar import TMDB_A_ML

TAREA = "CPIA features"
API = DATA / "external" / "tmdb_api.jsonl"

GENEROS = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Horror",
           "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Children"]
PAD, DESCONOCIDO, OTRO = 0, 1, 2
RESERVADOS = {"<PAD>": PAD, "<DESCONOCIDO>": DESCONOCIDO, "<OTRO>": OTRO}


# ---------------------------------------------------------------- lectura (E/S fina, separada de la lógica)
def _lista(x) -> list:
    """Lista de Python a partir de list/ndarray/None/NaN."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return []


def _eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []


def leer_api(ruta: Path = API) -> pd.DataFrame:
    """tmdb_api.jsonl -> DataFrame indexado por tmdbId (solo estado 200, la última fila de cada id)."""
    if not Path(ruta).exists():
        return pd.DataFrame(columns=["duracion", "keywords", "director", "reparto"])
    api = pd.read_json(ruta, lines=True)
    api = api[api.estado == 200].drop_duplicates("tmdbId", keep="last")
    return api.set_index("tmdbId")


def leer_kaggle(ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(credits, keywords, runtime) de Kaggle, restringidos a los tmdbId pedidos y con la basura de ids fuera."""
    cr = pd.read_csv(CRUDOS_TMDB / "credits.csv")
    cr["id"] = pd.to_numeric(cr.id, errors="coerce")
    cr = cr[cr.id.isin(ids)].drop_duplicates("id").set_index("id")
    kw = pd.read_csv(CRUDOS_TMDB / "keywords.csv")
    kw["id"] = pd.to_numeric(kw.id, errors="coerce")
    kw = kw[kw.id.isin(ids)].drop_duplicates("id").set_index("id")
    mm = pd.read_csv(CRUDOS_TMDB / "movies_metadata.csv", usecols=["id", "runtime"], low_memory=False)
    mm["id"] = pd.to_numeric(mm.id, errors="coerce")
    mm["runtime"] = pd.to_numeric(mm.runtime, errors="coerce")
    mm = mm.dropna(subset=["id"])
    rt = mm.groupby("id").runtime.first()
    return cr, kw, rt[rt.index.isin(ids)]


# ---------------------------------------------------------------- funciones puras
def generos_comunes(generos_ml, generos_tmdb) -> list[int]:
    """Unión de los géneros de MovieLens y de TMDB, traducida y filtrada al vocabulario de 16 (E5).

    Devuelve los índices (posiciones en GENEROS), ordenados. Family -> Children, Science Fiction -> Sci-Fi;
    IMAX, Foreign, Musical, Film-Noir, etc. quedan fuera.
    """
    nombres = set(_lista(generos_ml))
    for g in _lista(generos_tmdb):
        nombres.update(TMDB_A_ML.get(g, [g]))
    return sorted(GENEROS.index(g) for g in nombres if g in GENEROS)


def _nombres(filas) -> list[str]:
    return [p["name"] for p in filas if isinstance(p, dict) and p.get("name")]


def _dedup(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


def cargar_creditos(movies: pd.DataFrame, credits: pd.DataFrame, api: pd.DataFrame, n_actores: int = 5) -> pd.DataFrame:
    """Directores y actores por película, Kaggle primero y la API solo donde Kaggle no tiene nada (E9).

    movies: con `tmdbId`. credits: columnas cast/crew (strings de Python) indexado por tmdbId.
    api: indexado por tmdbId con `director` y `reparto` (listas).
    Devuelve, alineado con movies: directores, actores (listas de nombres), fuente_dir, fuente_act ('kaggle'|'api'|'falta').
    """
    dirs, acts, fd, fa = [], [], [], []
    for tid in movies.tmdbId:
        d, a, sd, sa = [], [], "falta", "falta"
        if pd.notna(tid):
            tid = int(tid)
            if tid in credits.index:
                fila = credits.loc[tid]
                crew = _eval(fila["crew"]) if isinstance(fila["crew"], str) else []
                cast = _eval(fila["cast"]) if isinstance(fila["cast"], str) else []
                d = _dedup(_nombres([p for p in crew if isinstance(p, dict) and p.get("job") == "Director"]))
                cast = sorted((p for p in cast if isinstance(p, dict)), key=lambda p: p.get("order", 999))
                a = _nombres(cast)[:n_actores]
                sd, sa = ("kaggle" if d else "falta"), ("kaggle" if a else "falta")
            if tid in api.index:
                if not d:
                    d = _dedup([x for x in _lista(api.at[tid, "director"]) if x])
                    sd = "api" if d else "falta"
                if not a:
                    a = [x for x in _lista(api.at[tid, "reparto"]) if x][:n_actores]
                    sa = "api" if a else "falta"
        dirs.append(d)
        acts.append(a)
        fd.append(sd)
        fa.append(sa)
    return pd.DataFrame({"directores": dirs, "actores": acts, "fuente_dir": fd, "fuente_act": fa}, index=movies.index)


def vocabulario_creditos(listas: list[list[str]], en_train: np.ndarray, minimo: int) -> dict[str, int]:
    """Persona -> índice (desde 3) para quien sale en >= `minimo` películas de train.

    0 <PAD>, 1 <DESCONOCIDO> (no hay dato), 2 <OTRO> (fuera del vocabulario) quedan reservados. Orden: más frecuentes
    primero, empates alfabéticos (determinista).
    """
    cuenta: dict[str, int] = {}
    for lista, tr in zip(listas, en_train):
        if tr:
            for p in set(lista):
                cuenta[p] = cuenta.get(p, 0) + 1
    ok = sorted((p for p, c in cuenta.items() if c >= minimo), key=lambda p: (-cuenta[p], p))
    return {**RESERVADOS, **{p: i + 3 for i, p in enumerate(ok)}}


def indices_creditos(listas: list[list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    """Lista vacía -> [<DESCONOCIDO>]; persona fuera del vocabulario -> <OTRO>."""
    return [np.array([vocab.get(p, OTRO) for p in l] if l else [DESCONOCIDO], dtype="int32") for l in listas]


def duraciones(movies: pd.DataFrame, runtime: pd.Series, api: pd.DataFrame) -> pd.DataFrame:
    """Duración en minutos: runtime de Kaggle (0 = falta) y, si falta, `duracion` de la API.

    Devuelve duracion (float32, NaN si no hay) y fuente_dur ('kaggle'|'api'|'falta').
    """
    rt = movies.tmdbId.map(lambda t: runtime.get(int(t), np.nan) if pd.notna(t) else np.nan)
    rt = pd.to_numeric(rt, errors="coerce").astype("float64")
    rt = rt.where(rt > 0)
    ap = movies.tmdbId.map(lambda t: api.at[int(t), "duracion"] if pd.notna(t) and int(t) in api.index else np.nan)
    ap = pd.to_numeric(ap, errors="coerce").astype("float64")
    ap = ap.where(ap > 0)
    dur = rt.fillna(ap)
    fuente = np.where(rt.notna(), "kaggle", np.where(ap.notna(), "api", "falta"))
    return pd.DataFrame({"duracion": dur.astype("float32"), "fuente_dur": fuente}, index=movies.index)


def cargar_keywords(movies: pd.DataFrame, keywords: pd.DataFrame, api: pd.DataFrame) -> pd.DataFrame:
    """keywords (lista de str) por película: Kaggle y, si no hay, la API. Con fuente_kw."""
    out, fu = [], []
    for tid in movies.tmdbId:
        k, f = [], "falta"
        if pd.notna(tid):
            tid = int(tid)
            if tid in keywords.index and isinstance(keywords.at[tid, "keywords"], str):
                k = _nombres(_eval(keywords.at[tid, "keywords"]))
                f = "kaggle" if k else "falta"
            if not k and tid in api.index:
                k = [x for x in _lista(api.at[tid, "keywords"]) if x]
                f = "api" if k else "falta"
        out.append(k)
        fu.append(f)
    return pd.DataFrame({"keywords": out, "fuente_kw": fu}, index=movies.index)


def numericas(valores, en_train) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estandariza con la media y la desviación (poblacional) de las películas de train que tienen dato (E4).

    Devuelve (z float32, falta bool, media, desv). Si falta: z = 0 y falta = True (E9).
    """
    v = pd.to_numeric(pd.Series(valores), errors="coerce").to_numpy(dtype="float64")
    falta = np.isnan(v)
    ref = v[np.asarray(en_train, dtype=bool) & ~falta]
    media = float(ref.mean()) if len(ref) else 0.0
    desv = float(ref.std()) if len(ref) else 1.0
    if desv == 0:
        desv = 1.0
    z = np.where(falta, 0.0, (np.where(falta, media, v) - media) / desv)
    return z.astype("float32"), falta, media, desv


def textos(sinopsis, keywords, keywords_en_texto: bool = True) -> list[str]:
    """sinopsis + " Keywords: " + ", ".join(keywords); sin keywords, solo la sinopsis (E3)."""
    out = []
    for s, k in zip(sinopsis, keywords):
        s = s.strip() if isinstance(s, str) else ""
        k = _lista(k) if keywords_en_texto else []
        out.append(f"{s} Keywords: {', '.join(k)}".strip() if k else s)
    return out


def embeddings(textos_: list[str], encoder: Callable[[list[str]], np.ndarray], lote: int = 256, dim: int = 384,
               dtype="float16", progreso: Callable[[int, int], None] | None = None) -> np.ndarray:
    """[N, dim]: encoder(lista de str) -> array [n, dim]. Los textos vacíos no se codifican: vector cero (E1)."""
    out = np.zeros((len(textos_), dim), dtype=dtype)
    pos = [i for i, t in enumerate(textos_) if t.strip()]
    for a in range(0, len(pos), lote):
        idx = pos[a:a + lote]
        out[idx] = np.asarray(encoder([textos_[i] for i in idx])).astype(dtype)
        if progreso:
            progreso(min(a + lote, len(pos)), len(pos))
    return out


def construir(movies: pd.DataFrame, credits: pd.DataFrame, keywords: pd.DataFrame, runtime: pd.Series, api: pd.DataFrame,
              en_train, cfg: dict) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    """Todo salvo los embeddings. movies ya ordenado por movieId (posición = item_idx).

    Devuelve (features, vocab_creditos, meta, textos).
    """
    f = cfg["features"]
    mv = movies.reset_index(drop=True)
    en_train = np.asarray(en_train, dtype=bool)

    gen = [generos_comunes(a, b) for a, b in zip(mv.generos, mv.generos_tmdb)]
    cred = cargar_creditos(mv, credits, api, f["n_actores"])
    dur = duraciones(mv, runtime, api)
    kw = cargar_keywords(mv, keywords, api)
    txt = textos(mv.sinopsis, kw.keywords, f["keywords_en_texto"])

    voc_d = vocabulario_creditos(list(cred.directores), en_train, f["min_peliculas_credito"])
    voc_a = vocabulario_creditos(list(cred.actores), en_train, f["min_peliculas_credito"])
    anio_z, sin_anio, m_a, s_a = numericas(mv.anio.astype("float64"), en_train)
    dur_z, sin_dur, m_d, s_d = numericas(dur.duracion, en_train)

    feats = pd.DataFrame({"item_idx": np.arange(len(mv), dtype="int32"), "movieId": mv.movieId.astype("int32")})
    feats["generos"] = [np.array(g, dtype="int8") for g in gen]
    feats["directores"] = indices_creditos(list(cred.directores), voc_d)
    feats["actores"] = indices_creditos(list(cred.actores), voc_a)
    feats["anio_z"] = anio_z
    feats["duracion_z"] = dur_z
    feats["sin_generos"] = [len(g) == 0 for g in gen]
    feats["sin_director"] = (cred.directores.map(len) == 0).to_numpy()
    feats["sin_reparto"] = (cred.actores.map(len) == 0).to_numpy()
    feats["sin_anio"] = sin_anio
    feats["sin_duracion"] = sin_dur
    feats["sin_texto"] = [not t.strip() for t in txt]

    def cob(mask) -> float:
        return round(float(np.mean(mask)), 4)

    def fuentes(s) -> dict:
        vc = pd.Series(list(s)).value_counts()
        return {k: int(vc.get(k, 0)) for k in ("kaggle", "api", "falta")}

    rec = (mv.anio >= 2017).fillna(False).to_numpy(dtype=bool)

    def reciente(fuente) -> dict:
        s = np.asarray(fuente)[rec]
        return {"n": int(rec.sum()),
                "solo_kaggle": cob(s == "kaggle") if rec.any() else 0.0,
                "con_api": cob(s != "falta") if rec.any() else 0.0}

    solo_ml = [len(generos_comunes(g, [])) > 0 for g in mv.generos]
    gen_fuente = np.where(solo_ml, "movielens", np.where([len(g) > 0 for g in gen], "tmdb", "falta"))
    meta = {
        "n_peliculas": int(len(mv)), "n_train": int(en_train.sum()),
        "cobertura": {
            "generos": cob(~feats.sin_generos), "director": cob(~feats.sin_director), "reparto": cob(~feats.sin_reparto),
            "anio": cob(~feats.sin_anio), "duracion": cob(~feats.sin_duracion),
            "keywords": cob(kw.fuente_kw != "falta"), "sinopsis": cob(mv.sinopsis.str.len() > 0),
            "texto": cob(~feats.sin_texto),
        },
        "fuente": {
            "generos": {k: int(v) for k, v in pd.Series(gen_fuente).value_counts().items()},
            "director": fuentes(cred.fuente_dir), "reparto": fuentes(cred.fuente_act),
            "duracion": fuentes(dur.fuente_dur), "keywords": fuentes(kw.fuente_kw),
        },
        "vocabulario": {"directores": len(voc_d), "actores": len(voc_a), "generos": len(GENEROS),
                        "reservados": len(RESERVADOS), "min_peliculas_credito": f["min_peliculas_credito"]},
        "anio_ge_2017": {"director": reciente(cred.fuente_dir), "reparto": reciente(cred.fuente_act),
                         "duracion": reciente(dur.fuente_dur)},
        "estandarizacion": {"anio": {"media": m_a, "desv": s_a}, "duracion": {"media": m_d, "desv": s_d}},
    }
    return feats, {"directores": voc_d, "actores": voc_a}, meta, txt


# ---------------------------------------------------------------- E/S
def _guardar_parquet(feats: pd.DataFrame, ruta: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {}
    for c in feats.columns:
        if c == "generos":
            cols[c] = pa.array([x.tolist() for x in feats[c]], type=pa.list_(pa.int8()))
        elif c in ("directores", "actores"):
            cols[c] = pa.array([x.tolist() for x in feats[c]], type=pa.list_(pa.int32()))
        else:
            cols[c] = pa.array(feats[c].to_numpy())
    pq.write_table(pa.table(cols), ruta)


def _encoder_minilm(nombre: str) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(nombre, device="cpu")
    return lambda lote: modelo.encode(lote, batch_size=len(lote), normalize_embeddings=False, show_progress_bar=False)


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", choices=["dev", "completo"], required=True)
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--sin-embeddings", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    f = cfg["features"]
    sal = PROCESADOS / a.muestra
    emb_ruta = PROCESADOS / "emb_texto.npy"
    if (sal / "peliculas_features.parquet").exists() and not a.forzar and (a.sin_embeddings or emb_ruta.exists()):
        print(f"{sal / 'peliculas_features.parquet'} ya existe (usa --forzar para rehacerlo)")
        return

    marcar(TAREA, 0, 4, "leyendo movies y split")
    movies = pd.read_parquet(PROCESADOS / "movies.parquet").sort_values("movieId").reset_index(drop=True)
    split = pd.read_parquet(sal / "split.parquet", columns=["item_idx", "particion"])
    mapeo_ruta = sal / "mapeo_peliculas.parquet"
    if mapeo_ruta.exists():
        mp = pd.read_parquet(mapeo_ruta).sort_values("item_idx")
        assert (mp.movieId.to_numpy() == movies.movieId.to_numpy()).all(), "item_idx no coincide con movies por movieId"
    en_train = np.zeros(len(movies), dtype=bool)
    en_train[split.item_idx[split.particion == 0].unique()] = True

    marcar(TAREA, 1, 4, "leyendo Kaggle y la API")
    cr, kw, rt = leer_kaggle({int(t) for t in movies.tmdbId.dropna()})
    api = leer_api()
    marcar(TAREA, 2, 4, "géneros, créditos, numéricas y texto")
    feats, vocab, meta, txt = construir(movies, cr, kw, rt, api, en_train, cfg)

    sal.mkdir(parents=True, exist_ok=True)
    _guardar_parquet(feats, sal / "peliculas_features.parquet")
    (sal / "vocab_generos.json").write_text(json.dumps({g: i for i, g in enumerate(GENEROS)}, indent=1), encoding="utf-8")
    (sal / "vocab_creditos.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    if not a.sin_embeddings and (a.forzar or not emb_ruta.exists()):
        t0 = time.time()
        emb = embeddings(txt, _encoder_minilm(f["modelo_texto"]), f["lote_texto"], dtype=f["dtype_embeddings"],
                         progreso=lambda h, t: marcar(TAREA, h, t, "embeddings MiniLM"))
        np.save(emb_ruta, emb)
        meta["embeddings"] = {"forma": list(emb.shape), "segundos": round(time.time() - t0, 1)}
        print(f"embeddings {emb.shape} en {time.time() - t0:.0f} s")
    elif emb_ruta.exists():
        meta["embeddings"] = {"forma": list(np.load(emb_ruta, mmap_mode="r").shape), "reutilizados": True}
    (sal / "meta_features.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    marcar(TAREA, 4, 4, "hecho")
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
