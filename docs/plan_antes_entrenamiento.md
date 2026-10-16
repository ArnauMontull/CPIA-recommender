# Plan: lo que queda antes del entrenamiento

Estado: **en curso** (16/10/2026). Sigue a `docs/plan_implementacion_CDE.md` (hecho en dev). Mismas reglas de código que allí.
Después de esto viene el entrenamiento (modelo Two-Tower en PyTorch) y, tras él, la inferencia en C (H10).

| # | Quién | Qué | Archivos (solo esos) |
|---|---|---|---|
| 1 | Haiku | Pipeline en el conjunto **completo** (split → senal → historial → features sin rehacer embeddings) y medir tiempo y RAM | ninguno (solo `data/processed/completo/`) |
| 2 | Sonnet | **Letterboxd** (G1, G3, G4, G5): leer el export (zip o carpeta: `ratings.csv` + `diary.csv`), emparejar, media_u encogida, bolsas A/B como `historial.py` | `src/letterboxd/`, `tests/test_letterboxd.py`, `tests/datos/letterboxd_simulado/` |
| 3 | Sonnet | **Experimento de frío** (G6): split por usuarios (10 % fuera; 80 % de historia dada, 20 % a predecir) con las mismas salidas que la pipeline normal | `src/preprocesamiento/frio.py`, `tests/test_frio.py` |
| 4 | Sonnet | **Colab** (H8): notebook, empaquetado de datos para Drive y guía paso a paso | `notebooks/entrenar_colab.ipynb`, `scripts/empaquetar_datos.py`, `docs/colab.md` |
| 5 | Opus | Revisión, invariantes, informe, commit cuando Arnau lo diga | — |

## Contratos

**2. Letterboxd.** `python -m src.letterboxd --export <zip|carpeta> --nombre <alias>` → `data/external/letterboxd/<alias>/`:
`emparejado.parquet` (Name, Year, Letterboxd URI, rating, fecha, `movieId`, `item_idx`, `metodo` ∈ {exacto, api, sin_emparejar}),
`hist_A_{indices,pesos}.npy`, `hist_B_{indices,pesos}.npy`, `valorados_indices.npy` (para la máscara C14) y `meta.json` (% emparejado por método,
media_u cruda y encogida, tamaños de bolsa). Re-visionados: la última nota (C12). Las notas de Letterboxd van de 0,5 a 5 (como MovieLens).
Pesos como F4 (c9 = 1: no hay «primer día» en un usuario nuevo), con la media_u de G4. La API: `descargar_tmdb.leer_credencial()` y `/search/movie`;
los tests la simulan (nunca llaman a la API ni leen `secrets/`).

**3. Frío.** `python -m src.preprocesamiento.frio --muestra dev` → `data/processed/<muestra>_frio/` con los mismos archivos que la pipeline normal
(`split.parquet`, `senal.parquet`, `usuarios.parquet`, `logq.npy`, máscaras, historiales, `ejemplos_train.parquet`, metas). Usuarios fríos: el 10 %
elegido con hash estable de `userId` + semilla (D10). Sus ratings no entran en train: el 80 % primero (por fecha) es la historia dada (`particion` 3)
y el 20 % último es el test (`particion` 2). Los demás usuarios: train y val como la pipeline normal, sin test. media_u y P(i) siguen saliendo solo de
train (D3); la media_u de un usuario frío sale de su historia dada (como Letterboxd, G4). Reutiliza las funciones de `split.py`, `senal.py` e
`historial.py` sin modificarlas.

**4. Colab.** El notebook monta Drive, clona `https://github.com/ArnauMontull/CPIA-recommender` (si el repo es privado, con un token guardado en los
secretos de Colab), instala `requirements.txt` sin reinstalar torch, descomprime los datos de Drive, comprueba la GPU, pasa `pytest` y deja preparada la
celda de entrenamiento (`python -m src.entrenar --muestra completo --device cuda`, que aún no existe). `empaquetar_datos.py` crea un zip con
`data/processed/<muestra>/`, `emb_texto.npy` y `movies.parquet`.
