# Planning de los bloques G (Letterboxd) y H (ingeniería)

Estado: **abierto** (16/10/2026). G1–G2 y H1–H7 son las preguntas originales del 15/10 (`docs/planning_bloques_D_H.md`), actualizadas;
el resto son nuevas. Las ⭐ **no son decisiones**.

Lo que prometimos en E1: Letterboxd con `watched.csv`, `ratings.csv` y `diary.csv` para «un usuario nuevo que introduce su diario» y medir
latencia y calidad con perfiles que no están en el entrenamiento; vectores precalculados, HNSW con 200 candidatos en < 5 ms; motor en C/C++ con
OpenMP y SIMD; benchmarks en CPU en la E2. Ojo: E1 habla de 84 000 películas y 43 MB; tras B1 el catálogo es de 23 335.

## G: Letterboxd

| # | Pregunta | Opciones | ⭐ | Implicación |
|---|---|---|---|---|
| G1 | Cómo se emparejan las películas (el export trae `Name`, `Year` y una URI de Letterboxd, sin tmdbId ni imdbId) | a) título + año exactos (normalizados) · b) difuso (`rapidfuzz`) con umbral · c) API de TMDB (`/search/movie` con título y año) · d) a, y lo que falle por c | d | a) pierde títulos traducidos o con «The» al final; b) falsos positivos (remakes del mismo año). Con d se registra qué falla (% emparejado en el informe). |
| G2 | De dónde sale el export de prueba | a) el de Arnau · b) el de Uriel · c) los dos · d) uno simulado | c (+ d para los tests) | Es un dato personal: fuera de git (`data/external/letterboxd/`, ya ignorado con `data/`). Los tests, con uno simulado. |
| G3 | Qué archivos del export se usan | a) solo `ratings.csv` · b) `ratings.csv` + `diary.csv` (fechas y re-visionados, C12) · c) b + `watched.csv` (vistas sin nota) | b | Las vistas sin nota (c) no tienen positivo/nota baja: entrarían como neutras (F4, α) o se ignoran. |
| G4 | media_u de un usuario nuevo | a) la de sus notas (C6, cruda) · b) si tiene < 20 notas, encogida hacia la media global (como C6 b) · c) umbral fijo 4,0 | b | Con pocas notas la media cruda es ruidosa (C6 lo dejó para la E3). Solo cambia la inferencia, no el entrenamiento. |
| G5 | Películas de Letterboxd fuera del catálogo | a) ignorarlas · b) calcular su v_i con la torre de ítem (contenido de TMDB) y usarlas en el historial | b si da tiempo, a en v1 | b es lo que permite F6 a; necesita bajar sinopsis/créditos de TMDB para esas películas. |
| G6 | Cómo se mide el arranque en frío de usuarios (D1 c no lo mide) | a) solo cualitativo con los exports reales · b) experimento aparte en MovieLens: 10 % de usuarios fuera del entrenamiento, se les da el 80 % de su historial y se predice el resto (= D1 d) · c) a + b | c | b es barato con F1 b (sin userId: no hay que reentrenar nada por usuario) y da cifras para la E3; a es la demo. |

## H: ingeniería

Ya decidido el 16/10/2026: **H1** pandas · **H2** tipos compactos (int32/float32/int8) · **H3** índices contiguos `item_idx`/`user_idx` con
tabla de mapeo (hecho en `split.py`) · **H4/H5** Parquet + `.npy` + `config.yaml`, etapas con caché · **H7** `config.yaml` con todos los
parámetros y semillas (D10). Quedan:

| # | Pregunta | Opciones | ⭐ | Implicación |
|---|---|---|---|---|
| H6 | Preprocesamiento en Python o en C++ | a) Python · b) C++ | a | Se ejecuta una vez; la eficiencia que se evalúa es la inferencia (ANN + MMR en C++/OpenMP). Es el argumento para el informe. |
| H8 | Dónde se entrena | a) CPU local (torch CPU, ya instalado) · b) GPU en Colab/Kaggle · c) a para dev y b para el completo | c | En dev: 1,5 M ejemplos por época, viable en CPU. El completo (~15 M) mejor con GPU. Colab implica subir los `.npy`/Parquet (unos cientos de MB). |
| H9 | HNSW | a) `hnswlib` (Python, para el baseline) · b) implementación propia en C++ con OpenMP · c) a como referencia + b como entrega | c | E1 promete C++/OpenMP/SIMD; con a se valida que la propia da los mismos vecinos (recall del ANN frente a búsqueda exacta). |
| H10 | Cómo pasa el modelo a C++ | a) exportar solo los vectores v_i y los pesos del MLP de usuario a `.npy` + un MLP en C++ · b) TorchScript/ONNX con libtorch/onnxruntime · c) la torre de usuario en Python y solo ANN + MMR en C++ | a | La torre de usuario es una media ponderada + un MLP de 2 capas: se reescribe en C++ en pocas líneas, sin dependencias. b añade librerías pesadas. |
| H11 | Pruebas | a) pytest por etapa (ya hecho) · b) a + un test de extremo a extremo con una muestra mínima · c) b + comprobación automática de fugas | c | La comprobación de fugas (media_u/P(i) solo de train) ya existe en `test_senal.py`; c la extiende a toda la pipeline. |
| H12 | Qué va a git | a) solo código, docs y `reports/` · b) a + artefactos pequeños (`meta_*.json`, vocabularios) · c) todo con Git LFS | b | `data/` sigue fuera; los `meta_*.json` documentan cada ejecución y pesan poco. |
