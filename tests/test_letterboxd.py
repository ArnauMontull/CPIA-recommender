"""Tests del modulo Letterboxd (bloque G) con un export simulado y un catalogo hecho a mano. Sin red ni credenciales."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.letterboxd import __main__ as cli
from src.letterboxd import emparejar as E
from src.letterboxd import perfil as P
from src.letterboxd.construir import construir
from src.letterboxd.export import leer_export

SIM = Path(__file__).parent / "datos" / "letterboxd_simulado"
CFG = {"letterboxd": {"min_notas_sin_encoger": 20, "lambda_encogimiento": 10, "carpeta": "external/letterboxd"},
       "senal": {"tope_umbral": 4.0, "margen_nota_baja": 1.0, "peso_primer_dia": 0.75, "segundos_primer_dia": 86400,
                 "umbral_fijo_eval": 4.0, "suavizado_logq": 1.0},
       "historial": {"peso_neutra": 0.25}}
MU = 3.5
N_ITEMS = 8

# movieId, titulo (formato MovieLens), anio, tmdbId, n_ratings
MOVIES = [
    (1, "Matrix, The", 1999, 603, 1000),
    (2, "Amelie (Fabuleux destin d'Amélie Poulain, Le)", 2001, 194, 800),
    (3, "Matrix Reloaded, The", 2003, 604, 600),
    (4, "Toy Story", 1995, 862, 900),
    (5, "Crash", 1999, 1, 100),
    (6, "Crash", 2001, 2, 500),
    (7, "Manbiki kazoku", 2018, 505192, 50),
    (8, "Heat", 1995, 949, 700),
]


def catalogo():
    movies = pd.DataFrame(MOVIES, columns=["movieId", "titulo", "anio", "tmdbId", "n_ratings"]).astype(
        {"movieId": "int32", "anio": "Int16", "tmdbId": "Int64", "n_ratings": "int32"})
    mapeo = pd.DataFrame({"movieId": movies.movieId, "item_idx": (8 - movies.movieId).astype("int32")})  # idx = 8 - id
    return E.Catalogo(movies, mapeo)


def idx(movie_id):
    return 8 - movie_id


def api_simulada(consultas=None):
    respuestas = {"Shoplifters": [{"id": 505192, "title": "Shoplifters", "release_date": "2018-06-02"}],
                  "Pelicula Imaginaria": []}

    def buscar(nombre, anio):
        if consultas is not None:
            consultas.append((nombre, anio))
        return respuestas.get(nombre, [])
    return buscar


# --- normalizacion y emparejamiento exacto ---------------------------------------------------------------

def test_normalizacion_articulos_acentos_y_puntuacion():
    assert E.normalizar("Matrix, The") == E.normalizar("The Matrix") == "the matrix"
    assert E.normalizar("Amélie") == "amelie"
    assert E.normalizar("Dr. Strangelove: Or, How I Learned...") == "dr strangelove or how i learned"
    assert E.normalizar("Beautiful Mind, A") == "a beautiful mind"
    assert E.titulos_catalogo("Amelie (Fabuleux destin d'Amélie Poulain, Le)") == [
        "amelie", "le fabuleux destin d amelie poulain"]


def test_exacto_the_matrix_y_acentos():
    c = catalogo()
    assert c.exacto("The Matrix", 1999) == 1
    assert c.exacto("Amélie", 2001) == 2
    assert c.exacto("Le Fabuleux Destin d'Amélie Poulain", 2001) == 2     # titulo alternativo


def test_anio_mas_menos_uno_y_cero_primero():
    c = catalogo()
    assert c.exacto("The Matrix Reloaded", 2004) == 3        # catalogo 2003
    assert c.exacto("The Matrix Reloaded", 2005) is None     # +-2 no vale
    assert c.exacto("Crash", 1999) == 5                    
    assert c.exacto("Crash", 2001) == 6


def test_empate_de_candidatos_gana_n_ratings():
    c = catalogo()
    assert c.exacto("Crash", 2000) == 6                      # 1999 y 2001 a +-1: gana n_ratings 500 > 100


def test_anio_cero_gana_aunque_el_pm1_tenga_mas_ratings():
    c = catalogo()
    assert c.exacto("Crash", 1999) == 5                      # +-0 primero: 5 (n=100); 6 (2001) esta a 2 anios
    assert c.exacto("Crash", 1998) == 5


# --- export: re-visionados -----------------------------------------------------------------------------

def test_revisionado_ultima_nota():
    df = leer_export(SIM).set_index("Name")
    assert len(df) == 8                                   # sin duplicados; la fila del diary sin nota no cuenta
    assert df.loc["Toy Story", "rating"] == 2.0           # diary 2024-06-01 (2.0) mas reciente que ratings 2024-01-13 (3.5)
    assert df.loc["The Matrix", "rating"] == 5.0          # ratings 2024-01-10 mas reciente que diary 2022-03-03 (4.0)
    assert df.loc["Heat", "rating"] == 3.0                # el diary sin nota se ignora
    assert df.loc["Toy Story", "fecha"] == pd.Timestamp("2024-06-01")


def test_export_zip_igual_que_carpeta(tmp_path):
    import zipfile
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n in ("ratings.csv", "diary.csv"):
            zf.write(SIM / n, n)
    pd.testing.assert_frame_equal(leer_export(z), leer_export(SIM))


# --- API simulada y catalogo ---------------------------------------------------------------------------

def test_fallback_api_y_fuera_de_catalogo():
    consultas = []
    emp = E.emparejar(leer_export(SIM), catalogo(), api_simulada(consultas)).set_index("Name")
    assert emp.loc["Shoplifters", "metodo"] == "api" and emp.loc["Shoplifters", "movieId"] == 7
    assert emp.loc["Shoplifters", "item_idx"] == idx(7)
    assert emp.loc["Pelicula Imaginaria", "metodo"] == "sin_emparejar" and pd.isna(emp.loc["Pelicula Imaginaria", "movieId"])
    assert emp.loc["The Matrix", "metodo"] == "exacto" and emp.loc["The Matrix", "item_idx"] == idx(1)
    assert sorted(consultas) == [("Pelicula Imaginaria", 2020), ("Shoplifters", 2018)]    # la API solo para lo que el exacto no encontro


def test_api_devuelve_pelicula_que_no_esta_en_el_catalogo():
    emp = E.emparejar(leer_export(SIM), catalogo(), lambda n, a: [{"id": 999999, "release_date": "2018-01-01"}])
    assert (emp.metodo == "api").sum() == 0
    assert emp.set_index("Name").loc["Shoplifters", "metodo"] == "sin_emparejar"


def test_sin_api_solo_exacto():
    emp = E.emparejar(leer_export(SIM), catalogo(), None)
    assert dict(emp.metodo.value_counts()) == {"exacto": 6, "sin_emparejar": 2}


def test_elegir_resultado_prefiere_el_anio():
    res = [{"id": 1, "release_date": "1990-01-01"}, {"id": 2, "release_date": "2018-03-03"}]
    assert E.elegir_resultado(res, 2018) == 2
    assert E.elegir_resultado(res, None) == 1
    assert E.elegir_resultado([], 2018) is None


def test_cache_en_disco_no_repite_llamadas(tmp_path):
    class Cliente:
        n = 0

        def get_ruta(self, ruta):
            Cliente.n += 1
            assert ruta.startswith("/search/movie?") and "query=Shoplifters" in ruta and "year=2018" in ruta
            return {"estado": 200, "results": [{"id": 505192, "title": "Shoplifters", "release_date": "2018-06-02"}]}

    cache = tmp_path / "cache.jsonl"
    assert E.BuscadorTMDB(Cliente(), cache)("Shoplifters", 2018)[0]["id"] == 505192
    assert E.BuscadorTMDB(Cliente(), cache)("Shoplifters", 2018)[0]["id"] == 505192     # lee del disco
    assert Cliente.n == 1


# --- media_u y bolsas -----------------------------------------------------------------------------------

def test_media_encogida_a_mano():
    r = np.array([5, 4.5, 3, 2, 1, 4, 2, 3])               # suma 24.5, n = 8 < 20
    cruda, m = P.media_encogida(r, MU, CFG["letterboxd"])
    assert cruda == pytest.approx(24.5 / 8)
    assert m == pytest.approx((24.5 + 10 * 3.5) / 18)       # 3.30556
    # n >= 20: sin encoger
    r20 = np.full(20, 5.0)
    assert P.media_encogida(r20, MU, CFG["letterboxd"]) == (5.0, 5.0)


def test_bolsas_pesos_como_f4_con_c9_igual_1():
    emp, bol, meta = construir(leer_export(SIM), catalogo(), N_ITEMS, MU, CFG, api_simulada())
    m = (24.5 + 35) / 18                                    # media_u encogida; incluye la nota de la pelicula fuera del catalogo
    assert meta["media_u"] == pytest.approx(m, abs=1e-4) and meta["encogida"]
    pesos_A = {idx(1): 1 + (5 - m), idx(2): 1 + (4.5 - m), idx(7): 1 + (4 - m),    # positivos: 1 + max(0, r - media_u)
               idx(3): 0.25, idx(8): 0.25}                                        # neutras: peso_neutra
    ix, ps = bol["hist_A"]
    assert list(ix) == sorted(pesos_A)
    np.testing.assert_allclose(ps, [pesos_A[i] for i in sorted(pesos_A)], rtol=1e-6)
    pesos_B = {idx(4): 1.0, idx(6): 1.0}                                          # notas bajas: 1 (Toy Story 2.0, Crash 1.0)
    ix, ps = bol["hist_B"]
    assert list(ix) == sorted(pesos_B) and list(ps) == [1.0, 1.0]


def test_bolsa_b_y_valorados_con_el_empate_resuelto():
    emp, bol, meta = construir(leer_export(SIM), catalogo(), N_ITEMS, MU, CFG, api_simulada())
    # Crash (2000) -> movieId 6; Toy Story -> 4
    assert list(bol["hist_B"][0]) == sorted([idx(6), idx(4)])
    assert list(bol["valorados"]) == sorted(idx(m) for m in (1, 2, 3, 4, 6, 7, 8))
    assert meta["n_por_metodo"] == {"exacto": 6, "api": 1, "sin_emparejar": 1}
    assert meta["pct_emparejado"]["exacto"] == pytest.approx(75.0)
    assert meta["tamano_bolsa"] == {"A": 5, "B": 2, "valorados": 7}


def test_pesos_iguales_a_historial_pesos_historial():
    """Las bolsas salen de la misma funcion que la pipeline (c9 = 1: primer_dia False)."""
    from src.preprocesamiento.historial import pesos_historial
    from src.preprocesamiento.senal import etiquetar
    r = np.array([5.0, 4.0, 3.0, 1.0], dtype="float32")
    media = np.array([3.25])
    df = pd.DataFrame({"user_idx": np.zeros(4, dtype="int32"), "rating": r, "primer_dia": False})
    w, es_b = pesos_historial(etiquetar(df, media, CFG["senal"]), media, CFG["senal"], CFG["historial"])
    b = P.bolsas(np.array([0, 1, 2, 3]), r, 3.25, 4, CFG["senal"], CFG["historial"])
    np.testing.assert_allclose(b["A"][1], w[~es_b])
    np.testing.assert_allclose(b["B"][1], w[es_b])
    assert list(w) == pytest.approx([2.75, 1.75, 0.25, 1.0])


# --- CLI completo, sin red ------------------------------------------------------------------------------

def test_cli_escribe_los_ficheros(tmp_path, monkeypatch):
    movies = pd.DataFrame(MOVIES, columns=["movieId", "titulo", "anio", "tmdbId", "n_ratings"])
    mapeo = pd.DataFrame({"movieId": movies.movieId.astype("int32"), "item_idx": (8 - movies.movieId).astype("int32")})
    proc = tmp_path / "processed"
    (proc / "dev").mkdir(parents=True)
    movies.to_parquet(proc / "movies.parquet")
    mapeo.to_parquet(proc / "dev" / "mapeo_peliculas.parquet")
    monkeypatch.setattr(cli, "PROCESADOS", proc)
    monkeypatch.setattr(cli, "DATA", tmp_path)
    monkeypatch.setattr(cli, "crear_buscador", lambda cache: E.BuscadorTMDB(
        type("C", (), {"get_ruta": lambda self, ruta: {"estado": 404}})(), cache))
    cli.main(["--export", str(SIM), "--nombre", "t", "--mu", "3.5"])
    out = tmp_path / "external" / "letterboxd" / "t"
    for f in ("emparejado.parquet", "hist_A_indices.npy", "hist_A_pesos.npy", "hist_B_indices.npy", "hist_B_pesos.npy",
              "valorados_indices.npy", "meta.json"):
        assert (out / f).exists(), f
    emp = pd.read_parquet(out / "emparejado.parquet")
    assert list(emp.columns) == ["Name", "Year", "Letterboxd URI", "rating", "fecha", "movieId", "item_idx", "metodo"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_por_metodo"] == {"exacto": 6, "api": 0, "sin_emparejar": 2}
