# Plan de implementación: preprocesamiento de los bloques C, D y E

Estado: **hecho en dev el 16/10/2026** (bloques C, D, E y F; informe en `reports/pipeline_dev.md`). Decisiones en `docs/decisiones_preprocesamiento.md`; parámetros en `config.yaml`.
Queda fuera hasta cerrar F, G y H: el generador de ejemplos (historiales), el modelo, el entrenamiento y Letterboxd.
H1 = pandas, H4/H5 = Parquet + `.npy` + `config.yaml` con etapas en caché (decidido el 16/10/2026).

## Reparto

| Fase | Quién | Qué |
|---|---|---|
| 0 | Opus (gestión) | Contratos (este documento) y `config.yaml` |
| 1 | Haiku | Dependencias en el venv del proyecto (torch CPU, sentence-transformers, pytest, pyyaml), bug de la ruta de la barra de progreso, inventario de los CSV de Kaggle |
| 2 | 4 × Sonnet en paralelo, cada uno en sus archivos | `split.py`, `senal.py`, `features_peliculas.py`, `metricas.py`, cada uno con sus tests |
| 3 | Haiku | Ejecutar la pipeline en `dev` + `pytest`, recoger cifras |
| 4 | Opus | Revisión de código, invariantes de fuga, informe `reports/pipeline_dev.md`, commit cuando Arnau lo diga |

## Reglas para todo el código

- pandas + numpy; tipos compactos (int32/float32/int8). Rutas y `marcar()` de `src/preprocesamiento/comun.py`.
- Cada etapa: funciones puras (DataFrame/arrays → DataFrame/arrays, sin E/S) + un `main()` con
  `python -m src.preprocesamiento.<etapa> --muestra {dev,completo}` que lee `config.yaml` y escribe en
  `data/processed/<muestra>/`. Si la salida existe, se salta salvo `--forzar` (H5).
- Todo lo que se aprende de los datos (medias, P(i), vocabularios, normalizaciones) sale **solo de train** (D3).
- Tests en `tests/test_<etapa>.py` con datos sintéticos pequeños hechos a mano (nunca los datos reales), ejecutables con
  `..\venv\Scripts\python.exe -m pytest tests`. Cada test comprueba un número calculado a mano.
- Las etapas largas llaman a `marcar(TAREA, hecho, total, texto)`.

## Índices (H3)

- `item_idx`: 0..N−1 sobre **todas** las películas de `movies.parquet`, ordenadas por `movieId`. Así el índice es el mismo en dev y en completo.
  → `mapeo_peliculas.parquet` (`movieId` int32, `item_idx` int32).
- `user_idx`: 0..U−1 sobre los usuarios de la muestra, ordenados por `userId` → `mapeo_usuarios.parquet`.

## Etapa 1: `src/preprocesamiento/split.py` (D1, D2, D4, D7, D8, C11)

Entrada: `data/processed/<muestras[m]>` y `movies.parquet`.
1. Quitar las filas con `antes_estreno` (C11).
2. Para cada usuario: ordenar por (`timestamp`, `movieId`). Con n ratings: n_test = max(1, round(0,1·n)) y n_val = max(1, round(0,1·n)),
   tomados del final (test = los últimos). Si n < 3, el usuario queda entero en train.
3. `primer_dia` = timestamp < primer timestamp del usuario + 86 400 (C9).
4. D8: en val/test se quitan las filas de películas sin ningún rating en train (se cuentan en `meta`).

Salida `split.parquet`: `user_idx` int32, `item_idx` int32, `rating` float32, `timestamp` int64, `particion` int8
(0 train, 1 val, 2 test), `primer_dia` bool. Más `meta_split.json` con recuentos (filas por partición, quitadas por C11 y D8,
usuarios con corte en empate de segundo).

Tests mínimos: usuario con 10 ratings → 8/1/1; empates resueltos por `movieId`; ningún val/test anterior a un train del mismo
usuario; película solo en test → fuera; usuario con 2 ratings → todo train.

## Etapa 2: `src/preprocesamiento/senal.py` (C2–C9, C13, C14, C7, C8)

Entrada: `split.parquet` + `config.yaml`. Con train (particion 0):
- `media_u` = media cruda de los ratings de train del usuario (C6).
- `positivo` = rating ≥ min(tope_umbral, media_u) (C2); `nota_baja` = rating ≤ media_u − margen_nota_baja (C3). Se calculan en las tres
  particiones, pero siempre con la media_u de train.
- `peso` (solo positivos de train; 0 en el resto) = (1 + max(0, rating − media_u)) × (peso_primer_dia si `primer_dia`, si no 1) (C4, C9).
- `relevante_4` = rating ≥ umbral_fijo_eval (C7; en val/test).
- `logq.npy` float32 [N] = log((pos_i + s) / (total_pos + s·N)) con los positivos de train (C13).
- Máscara C14 en CSR: `valorados_train_indptr.npy` int64 [U+1] y `valorados_train_indices.npy` int32 (item_idx ordenados por usuario).
- `usuarios.parquet`: `user_idx`, `media_u`, `n_train`, `n_pos_train`, `elegible_val`, `elegible_test` (C8: ≥ 1 positivo en esa
  partición y ≥ 1 positivo en train), y lo mismo con `relevante_4`.

Salida `senal.parquet` = columnas de `split.parquet` + `positivo`, `nota_baja`, `peso` float32, `relevante_4`. Más `meta_senal.json`
(% de positivos por partición, usuarios elegibles…).

Tests: los usuarios Ana, Bruno y Carla de `docs/planning_bloque_C.md` (positivos con min(4,0; media_u)); un rating de test no cambia
media_u; logQ suma 1 en probabilidad; peso de un 5,0 del primer día = (1 + 5 − media_u)·0,75.

## Etapa 3: `src/preprocesamiento/features_peliculas.py` (E1–E9)

Entrada: `movies.parquet`, `data/the_movies_dataset_kaggle/{credits,keywords,movies_metadata}.csv`, `data/external/tmdb_api.jsonl`
(API = relleno de lo que falta en Kaggle: E9) y `split.parquet` (solo para saber qué películas están en train).
- Géneros (E5): vocabulario de 16 = Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance,
  Sci-Fi (= Science Fiction), Thriller, War, Western, Children (= Family). Unión de los de MovieLens y TMDB, filtrada al vocabulario.
- Créditos (E2): todos los directores + los `n_actores` primeros del reparto (por `order`). Vocabulario con las personas que salen en
  ≥ `min_peliculas_credito` películas con rating en train. Índices reservados: 0 `<PAD>`, 1 `<DESCONOCIDO>` (no hay dato), 2 `<OTRO>`.
- Numéricas (E4): año (de `movies.anio`) y duración (`runtime`), estandarizadas con media y desviación de las películas de train;
  si falta: 0 y bandera `sin_<campo>` (E9).
- Texto (E1, E3, E8): `sinopsis + " Keywords: " + ", ".join(keywords)` → MiniLM congelado, sin normalizar → `emb_texto.npy` float16 [N, 384].
  Sin texto: vector cero y bandera `sin_texto`.

Salida `peliculas_features.parquet` (una fila por `item_idx`, en orden): `item_idx`, `movieId`, `generos` list<int8>,
`directores` list<int32>, `actores` list<int32>, `anio_z`, `duracion_z` float32, banderas bool. Más `vocab_creditos.json`,
`vocab_generos.json` y `meta_features.json` (coberturas, fuente de cada dato: Kaggle / API / falta).
Los embeddings no dependen del split: se guardan en `data/processed/emb_texto.npy` (compartidos por dev y completo).

Tests: mapeo de géneros (Family → Children, IMAX fuera), corte de frecuencia mínima, relleno con la API cuando falta en Kaggle,
estandarización solo con train; para el texto, un encoder falso inyectable (los tests no descargan MiniLM).

## Etapa 4: `src/evaluacion/metricas.py` (D6, C14)

Funciones numpy puras:
- `top_k(scores, k, excluir_indptr, excluir_indices)`: pone −∞ en lo valorado en train (C14) y devuelve los índices top-k.
- `recall_at_k`, `ndcg_at_k` (ganancia binaria; IDCG con min(k, |R_u|)) por usuario, con relevantes en CSR.
- `cobertura(topk, N)` = fracción del catálogo que sale en algún top-k; `novedad(topk, pop)` = media de −log2 p(i) con la popularidad de train.
- `evaluar(...)` → dict {metrica@k: media sobre usuarios elegibles}, para las dos definiciones de relevante (C7).

Tests con casos de 2–3 usuarios calculados a mano (incluido NDCG con un solo relevante en la posición 2 = 1/log2(3)).
