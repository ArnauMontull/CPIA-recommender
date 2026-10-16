"""Tests de la etapa 3 con datos sintéticos hechos a mano."""
import numpy as np
import pandas as pd

from src.preprocesamiento import features_peliculas as fp


def test_generos_family_scifi_imax_foreign():
    # ML: IMAX, Sci-Fi, Musical (fuera); TMDB: Family -> Children, Science Fiction -> Sci-Fi, Foreign fuera
    idx = fp.generos_comunes(["IMAX", "Sci-Fi", "Musical"], ["Family", "Science Fiction", "Foreign"])
    assert [fp.GENEROS[i] for i in idx] == ["Sci-Fi", "Children"]
    assert fp.generos_comunes([], []) == []
    assert fp.generos_comunes(["Action"], None) == [0]
    assert len(fp.GENEROS) == 16


def test_vocabulario_frecuencia_minima():
    listas = [["Ana", "Bob"], ["Ana"], ["Ana", "Bob"], ["Bob", "Cy"], ["Cy", "Cy"]]
    en_train = np.array([True, True, True, False, True])
    # train: Ana 3, Bob 2, Cy 1 (la película 4 no es de train; Cy repetido en una película cuenta 1)
    voc = fp.vocabulario_creditos(listas, en_train, minimo=2)
    assert voc == {"<PAD>": 0, "<DESCONOCIDO>": 1, "<OTRO>": 2, "Ana": 3, "Bob": 4}
    ind = fp.indices_creditos([["Ana", "Cy"], [], ["Bob"]], voc)
    assert ind[0].tolist() == [3, 2]   # Cy -> <OTRO> = 2
    assert ind[1].tolist() == [1]      # sin dato -> <DESCONOCIDO> = 1
    assert ind[2].tolist() == [4]


def _movies():
    return pd.DataFrame({"movieId": [1, 2, 3], "tmdbId": [10, 20, 30]})


def _credits():
    crew = "[{'job': 'Director', 'name': 'Dir K'}, {'job': 'Producer', 'name': 'Prod'}]"
    cast = "[{'name': 'C', 'order': 2}, {'name': 'A', 'order': 0}, {'name': 'B', 'order': 1}]"
    return pd.DataFrame({"cast": [cast, "[]"], "crew": [crew, "[]"]}, index=pd.Index([10, 20], name="id"))


def _api():
    return pd.DataFrame({
        "director": [["Dir API"], ["Dir API 2"], ["Dir 30"]],
        "reparto": [["X", "Y"], ["P", "Q", "R"], ["Z"]],
        "duracion": [99, 77, 0], "keywords": [["k1"], ["k2", "k3"], []],
    }, index=pd.Index([10, 20, 30], name="tmdbId"))


def test_creditos_kaggle_y_relleno_api():
    c = fp.cargar_creditos(_movies(), _credits(), _api(), n_actores=2)
    # película 10: Kaggle manda (solo el director; los 2 primeros por order); la API no pisa
    assert c.directores[0] == ["Dir K"] and c.actores[0] == ["A", "B"]
    assert c.fuente_dir[0] == "kaggle" and c.fuente_act[0] == "kaggle"
    # película 20: Kaggle vacío -> API (reparto recortado a 2)
    assert c.directores[1] == ["Dir API 2"] and c.actores[1] == ["P", "Q"]
    assert c.fuente_dir[1] == "api"
    # película 30: no está en Kaggle -> API
    assert c.directores[2] == ["Dir 30"] and c.fuente_act[2] == "api"
    # sin API: queda vacío y 'falta'
    c2 = fp.cargar_creditos(_movies(), _credits(), _api().iloc[0:0], n_actores=2)
    assert c2.directores[1] == [] and c2.fuente_dir[2] == "falta"


def test_duracion_runtime_cero_es_falta_y_api_rellena():
    runtime = pd.Series({10: 120.0, 20: 0.0})   # 20: runtime 0 = falta; 30 ausente
    d = fp.duraciones(_movies(), runtime, _api())
    assert d.duracion[0] == 120 and d.fuente_dur[0] == "kaggle"   # la API (99) no pisa a Kaggle
    assert d.duracion[1] == 77 and d.fuente_dur[1] == "api"
    assert np.isnan(d.duracion[2]) and d.fuente_dur[2] == "falta"   # API 0 también es falta


def test_keywords_y_texto():
    kw = pd.DataFrame({"keywords": ["[{'id': 1, 'name': 'robot'}, {'id': 2, 'name': 'space'}]", "[]"]},
                      index=pd.Index([10, 20], name="id"))
    k = fp.cargar_keywords(_movies(), kw, _api())
    assert k.keywords[0] == ["robot", "space"] and k.fuente_kw[0] == "kaggle"
    assert k.keywords[1] == ["k2", "k3"] and k.fuente_kw[1] == "api"
    assert k.keywords[2] == [] and k.fuente_kw[2] == "falta"
    t = fp.textos(["Una peli.", "Otra.", "Sola."], k.keywords)
    assert t == ["Una peli. Keywords: robot, space", "Otra. Keywords: k2, k3", "Sola."]
    assert fp.textos(["Una peli."], [["a"]], keywords_en_texto=False) == ["Una peli."]
    assert fp.textos([""], [[]]) == [""]


def test_estandarizacion_solo_train():
    v = pd.Series([100.0, 120.0, 1000.0, np.nan])
    en_train = np.array([True, True, False, True])
    z, falta, media, desv = fp.numericas(v, en_train)
    # train con dato: 100 y 120 -> media 110, desv poblacional 10
    assert media == 110 and desv == 10
    assert z[0] == -1 and z[1] == 1 and z[2] == 89   # 1000 (no train) se transforma con los datos de train
    assert z[3] == 0 and falta.tolist() == [False, False, False, True]
    assert z.dtype == np.float32


def test_embeddings_encoder_falso():
    llamadas = []

    def falso(lote):
        llamadas.append(len(lote))
        return np.tile(np.arange(4, dtype="float32") + 1, (len(lote), 1))

    e = fp.embeddings(["a", "", "b", "c", "  "], falso, lote=2, dim=4)
    assert e.shape == (5, 4) and e.dtype == np.float16
    assert llamadas == [2, 1]                      # 3 textos no vacíos en lotes de 2
    assert e[0].tolist() == [1, 2, 3, 4]
    assert not e[1].any() and not e[4].any()       # vacío -> vector cero
    assert fp.embeddings([], falso, dim=4).shape == (0, 4)


def test_construir_extremo_a_extremo():
    mv = pd.DataFrame({
        "movieId": [1, 2, 3], "tmdbId": [10, 20, 30], "anio": pd.array([2000, 2018, pd.NA], dtype="Int16"),
        "generos": [["Action", "IMAX"], [], ["Comedy"]], "generos_tmdb": [["Family"], ["Science Fiction"], []],
        "sinopsis": ["Hola.", "", ""],
    })
    kw = pd.DataFrame({"keywords": ["[]"]}, index=pd.Index([10], name="id"))
    cfg = {"features": {"n_actores": 5, "min_peliculas_credito": 1, "keywords_en_texto": True}}
    feats, vocab, meta, txt = fp.construir(mv, _credits(), kw, pd.Series({10: 100.0}), _api(),
                                           np.array([True, True, False]), cfg)
    assert feats.generos[0].tolist() == [0, 15] and feats.generos[1].tolist() == [11]
    assert feats.generos[2].tolist() == [3]
    assert feats.sin_anio.tolist() == [False, False, True]
    assert txt == ["Hola. Keywords: k1", "Keywords: k2, k3", ""]
    assert feats.sin_texto.tolist() == [False, False, True]
    assert meta["anio_ge_2017"]["director"]["n"] == 1
    assert meta["cobertura"]["director"] == 1.0
    assert meta["fuente"]["director"] == {"kaggle": 1, "api": 2, "falta": 0}
