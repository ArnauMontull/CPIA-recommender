# Decisiones de preprocesamiento

Registro de las decisiones del planning de preprocesamiento (bloques A–H). Cada una se convertirá en un
parámetro del pipeline. Las que siguen abiertas están marcadas como **pendiente**.

## Cómo reproducir

```
python -m venv venv && venv\Scripts\pip install -r requirements.txt
python -m src.preprocesamiento.validar         # B4 -> data/processed/{ratings,movies,fusiones}.parquet + reports/validacion.md
python -m src.preprocesamiento.descargar_tmdb  # sinopsis que faltan (necesita token de TMDB, ver el docstring)
python -m src.preprocesamiento.validar         # otra vez, para incorporar lo descargado
python -m src.preprocesamiento.submuestra    # A2 -> data/processed/ratings_dev_10pct.parquet + reports/submuestra.md
```

`data/` no está en git: cada uno pone los CSV originales en `data/ml-32m/` y `data/the_movies_dataset_kaggle/`.

## Bloque A: qué datos entran (15/10/2026)

| # | Decisión | Por qué |
|---|---|---|
| A1 | Solo `ml-32m/ratings.csv` | El `ratings.csv` de TMDB Kaggle es MovieLens antiguo (2017) con otros IDs: duplicaría interacciones sin poder detectarlo. De Kaggle solo usamos metadatos. |
| A2 | Submuestra de desarrollo del 10 % **por usuarios**, estratificada (género dominante × quintil de actividad × época del primer rating); el conjunto completo, solo para resultados finales | Iterar rápido sin romper historiales. El reparto de géneros difiere como mucho 0,35 pp (ver `reports/submuestra.md`). |
| A3 | Todo el histórico (1995–2023) | Sin recorte temporal por ahora. |
| A4 | Dataset 2024–2026 descartado de momento | No tiene IDs (solo título) y tiene cifras sospechosas (p. ej. 280 952 votos en un título desconocido). Revisarlo más adelante. |

## Bloque B: limpieza (15/10/2026)

| # | Decisión | Por qué |
|---|---|---|
| B1 | ~~Sin *k-core*~~ → **revisada: fuera las películas con < 20 ratings** (y sus ratings) | Usuarios: MovieLens ya trae ≥ 20 por usuario. Películas: pasan de 87 585 a 23 335, pero solo se pierde el 0,86 % de los ratings. Contrapartida: la cola larga extrema desaparece, y el argumento de arranque en frío de E1 habrá que defenderlo con películas de 20–100 ratings o con un experimento aparte. 185 usuarios quedan con < 20 ratings tras el filtro; se mantienen. |
| B1b | Duplicados fusionados: 36 `movieId` con el mismo `tmdbId` se integran en el que tiene más ratings | Son la misma película (p. ej. *Confessions of a Dangerous Mind* dos veces). Si un usuario valoró las dos copias (836 casos), se queda la nota más reciente. Lista en `reports/validacion.md`. |
| B2 | Las películas sin sinopsis se mantienen (sinopsis vacía + flag) y los huecos se rellenan con la API de TMDB | Kaggle es de 2017. Tras el filtro faltaban 3 508 sinopsis; la API recupera 3 412 (117 404). Ahora el 99,4 % de las películas y el 99,9 % de los ratings tienen sinopsis. 15 películas no tenían ni géneros ni sinopsis: casi todas eran series o episodios con un tmdbId de película roto. 13 se han enlazado a mano por su imdbId (`src/preprocesamiento/correcciones_tmdb.csv`) con sinopsis y géneros de TMDB traducidos al vocabulario de MovieLens (flags `es_serie`, `generos_imputados`). Quedan 2 sin identificar: *The Code* (tt1194105) y *The Misadventures of Rick and Morty* (tt5195738). |
| B3 | **Pendiente**: géneros de MovieLens o de TMDB | Ver la comparación más abajo. |
| B4 | Validación de integridad con correcciones y flags | `reports/validacion.md`. Los ratings vienen limpios (0 nulos, 0 duplicados, 0 notas inválidas); los problemas están en los metadatos. |

### B3: comparación de géneros

Ambas fuentes tienen 20 géneros, pero no son los mismos:

- **Comunes (16)**: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy, Horror, Mystery, Romance,
  Sci-Fi = Science Fiction, Thriller, War, Western, y Children ≈ Family (parecidos, no iguales).
- **Solo MovieLens**: Musical, Film-Noir, IMAX (IMAX es un formato de proyección, no un género) y Children.
- **Solo TMDB**: Family, Music, History, Foreign (indica que la película no es estadounidense, no es un género) y TV Movie.
- **Cobertura**: MovieLens tiene géneros en 80 505 de 87 585 películas (7 080 «(no genres listed)»); TMDB solo cubre las
  ~42 000 que están en Kaggle. De las 7 080 sin género en ML, TMDB rellena 1 188.

## Bloque C: señal objetivo («le gusta»)

Decidido el 16/10/2026. Opciones y cifras en `docs/planning_bloque_C.md` (medidas con
`python -m src.analisis.senal_objetivo`, `reports/senal_objetivo.md`). media_u = media del usuario **solo con train**
(ver D). Pendiente de confirmar con D1 y F1 (el split y si la torre de usuario usa historial).

| # | Decisión | Por qué |
|---|---|---|
| C1 | Implícito binario: contrastiva con negativos del lote + logQ | Lo prometido en E1; ordena candidatos para HNSW + MMR, que es lo que se evalúa (Recall@K, NDCG@K). La regresión de r_ui queda como línea base. |
| C2 | Positivo si r ≥ min(4,0; media_u) | Nadie se queda sin positivos y solo 151 usuarios sin positivos en test (59,4 % de positivos). Rescata a los exigentes sin rebajar el «4» de los generosos. Con tope 3,4 (= 3,5, las notas van de 0,5 en 0,5) 1 646 ratings de 451 usuarios serían a la vez positivo y nota baja (C3); con 4,0, ninguno. |
| C3 | Notas bajas (r ≤ media_u − 1) como anti-preferencias en la torre de usuario; no entran en la pérdida | Los negativos del lote casi nunca son notas bajas del usuario (1,1 %): la señal hay que meterla a propósito. Se implementa en F. |
| C4 | Peso del positivo w = 1 + max(0, r − media_u), multiplicando la pérdida | «Ponderada por r_ui» (E1), relativo al usuario como C2. ESS 90,2 %. |
| C5 | Cada rating positivo es un ejemplo, sin tope por usuario | Simplicidad y logQ exacto. Contrapartida: el 10 % más activo aporta el 42,6 % de los positivos. |
| C6 | media_u = media cruda (sin encogimiento) | MovieLens tiene ≥ 20 ratings por usuario; revisar para usuarios nuevos de Letterboxd (E3). |
| C7 | Relevante en test: la definición de train (C2) y, en otra columna, r ≥ 4,0 | Comparar modelos y con la literatura. |
| C8 | Usuarios sin positivos (en train o test): se quedan en train y salen de las métricas | Su historial ayuda a las películas; con C2 son 0 sin positivos y 151 sin positivos en test. |
| C9 | Peso ×0,75 a los ratings de las primeras 24 h de cada usuario (se multiplica por C4) | El 62,5 % de los ratings son del primer día: recuerdos introducidos de golpe, con nota más alta (3,60 vs 3,44). |
| C10 | Sin ajuste para usuarios sin medias estrellas | El umbral relativo (media_u) ya absorbe la escala. |
| C11 | Fuera los ratings anteriores al estreno (flag `antes_estreno`) | 42 en la submuestra; no sabemos si la vio. |
| C12 | Re-ratings (Letterboxd): la última nota | Coherente con B1b. |
| C13 | logQ con P(i) = (positivos de i en train + 1) / total | Es la distribución con la que salen de verdad los negativos del lote (C5: un ejemplo por positivo; C4 y C9 pesan la pérdida, no el muestreo). |
| C14 | En la softmax se enmascara todo lo que el usuario ha valorado en train | Sin máscara, el 7,2 % de los negativos del lote es un positivo del propio usuario. Coste: una máscara por lote. |

## Bloques D–H

Preguntas originales en `docs/planning_bloques_D_H.md` (D split, E features de películas, F torre de usuario,
G Letterboxd, H ingeniería). Guía con cifras de D y E: `docs/guia_bloque_D_E/guia_bloque_D_E.pdf` y
`docs/planning_bloque_D_E.md` (medidas con `python -m src.analisis.split_y_features`). Lo que no sale aquí, **pendiente**.

### Bloque D: split (16/10/2026)

| # | Decisión | Por qué |
|---|---|---|
| D1 | Split por usuario y por tiempo: de cada usuario, el último 10 % de sus ratings (por fecha) a test, el 10 % anterior a validación y el resto a train | Estándar (NCF, SASRec) y sin fuga temporal dentro del usuario. Contrapartida: todos los usuarios de test están en train, así que el caso Letterboxd (usuario nuevo) no se mide aquí: hará falta un experimento aparte. |
| D2 | Empates de segundo: se desempata por `movieId` | Determinista. En el 24,4 % de los usuarios el corte cae en un empate. |
| D3 | Todo lo «por usuario» o «por película» se ajusta solo con train y se aplica al resto (`fit/transform`): media_u, P(i), máscara de C14, vocabularios, normalizaciones | Si media_u y P(i) usaran todo, cambiaría la etiqueta del 1,46 % de los ratings de train y 1 730 películas tendrían P(i) solo gracias a test. |
| D4 | Validación con el mismo formato que test (el 10 % penúltimo de cada usuario, D1) | Hiperparámetros y early stopping con la misma vara que el test. |
| D5 | Primer día: nada más que C9 (×0,75) y el desempate de D2 | El 50,2 % del test es del mismo día que el último rating de train; se documenta en la E2. |
| D6 | Recall@K y NDCG@K con K ∈ {10, 20, 50, 200} sobre el catálogo completo, más cobertura y novedad | @200 mide la recuperación de candidatos (HNSW) y @10–20 la lista final (MMR). Sin negativos muestreados. |
| D7 | 80/10/10 dentro de cada usuario | — |
| D8 | Películas sin ratings en train: fuera del test | Son el 0,43 % del test con este split. Contrapartida: el arranque en frío de películas que prometimos en E1 no se mide aquí. |
| D9 | Desarrollo en la submuestra; 2–3 configuraciones finales en el conjunto completo | Iterar rápido. Ojo: en la submuestra hay más películas «frías» que en el completo (15 422 con < 20 ratings en train). |
| D10 | 5 semillas (inicialización y orden de los lotes); asignaciones aleatorias con hash estable de `userId` + semilla | Con D1 el split es determinista; las semillas miden la variabilidad del entrenamiento. Se reporta media ± desviación. |

### Bloque E: features de la torre de películas (16/10/2026)

| # | Decisión | Por qué |
|---|---|---|
| E1 | Sinopsis con MiniLM (`all-MiniLM-L6-v2`, d = 384), congelado y precalculado en caché `.npy` | Lo prometido en E1; se calcula una vez (17,9 MB en float16). |
| E2 | Director + los 5 primeros actores; los que salen en < 5 películas (en train) van a `<OTRO>` | 7 827 directores y 22 100 actores solo en el top 3: hace falta recortar el vocabulario. |
| E3 | Keywords añadidas al texto de la sinopsis antes de MiniLM (opción b de `planning_bloque_D_E.md`) | Sin features ni parámetros nuevos; la sinopsis (mediana 46 palabras) + keywords caben en MiniLM. Su efecto solo se mide comparando con y sin. |
| E4 | Año (tal cual, como número continuo y estandarizado con train, no por décadas) y duración; sin popularidad ni `vote_average` | La popularidad como feature refuerza el sesgo que el MMR quiere combatir. |
| E5 | Géneros: los 16 comunes a MovieLens y TMDB (Children ≈ Family; Sci-Fi = Science Fiction) | Cierra B3. Mismo vocabulario en MovieLens, TMDB y Letterboxd (que se enlazará por TMDB). Se pierden Musical, Film-Noir, Music, History y TV Movie. |
| E6 | Contenido + embedding de `movieId` con dropout del ID | El ID memoriza lo que el contenido no explica; el dropout obliga a funcionar también solo con contenido (películas nuevas, Letterboxd). |
| E7 | Normalización L2 de u y v_i + temperatura τ | Producto escalar = coseno: estable, y coherente con el MMR. τ es un hiperparámetro (validación, D4). |
| E8 | Encoder de texto congelado + capa densa entrenable encima | Barato y sin sobreajuste; es lo que permite el caché de E1. |
| E9 | Lo que falta se completa con la API de TMDB | Ya descargado en `data/external/tmdb_api.jsonl` (3 528 películas: 3 404 con director, 3 387 con reparto, 3 407 con duración, 3 147 con keywords); hay que integrarlo en las features (hoy solo se usa para sinopsis y géneros). Lo que siga faltando: token «desconocido» + bandera. |

### Bloque F: torre de usuario (16/10/2026)

Opciones y cifras en `docs/planning_bloque_F.md`.

| # | Decisión | Por qué |
|---|---|---|
| F1 | El usuario = agregación de su historial (sin embedding de `userId`) | Sirve para usuarios nuevos (Letterboxd) y para las anti-preferencias (C3 c). |
| F2 | Historial completo, de longitud variable (sin tope) | No se pierde nada de los usuarios grandes (p99 = 563 positivos). Se guarda como CSR (offsets + índices), que también lee C++. |
| F3 | Cada ejemplo (un positivo de train, C5 a) lleva como historial todos los demás ratings de train del usuario, sin el objetivo | Pooling sin orden; coherente con la evaluación (en test el historial es todo lo anterior) y con el 59 % de usuarios de un solo día. |
| F4 | En el historial entra todo lo valorado, en dos medias: (1) positivos con peso C4 × C9 + zona neutra con peso α × C9 (α = 0,25); (2) notas bajas (C3 c) con peso 1 × C9 | Usa todo sin que lo «regular» diluya lo que le gusta; con medias separadas un peso uniforme no tendría efecto. α = 0 (sin neutras) queda como ablación. |
| F5 | Medias ponderadas con los pesos de F4 | Reutiliza los pesos de `senal.py`. |
| F6 | Las películas del historial usan el mismo v_i de la torre de ítem (pesos compartidos) | Una película nueva o de Letterboxd ya tiene vector por su contenido; menos parámetros. |
| F7 | [media positivos+neutras, media notas bajas, banderas (sin notas bajas…), log n] → MLP de 2 capas → d = 128, con L2 (E7) | La torre de ítem sale con la misma d. |
| F8 | Historial al evaluar: en val, train; en test, train + val. La máscara C14 en test, lo valorado en train + val | Realista; el historial es entrada, no un estadístico ajustado (D3 sigue valiendo para media_u y P(i)). |

### Bloque H: ingeniería (16/10/2026)

Opciones en `docs/planning_bloques_G_H.md`. G (Letterboxd) y H8–H10 siguen **pendientes**.

| # | Decisión | Por qué |
|---|---|---|
| H1 | pandas | Lo que ya usa el repo; Polars, como mucho, benchmark para la E2. |
| H2 | Tipos compactos (int32 / float32 / int8 / bool) | ~400 MB en lugar de ~1 GB para los 32 M de ratings. |
| H3 | Índices contiguos `item_idx` (todas las películas, por `movieId`) y `user_idx`, con tabla de mapeo | Tablas de embeddings y arrays en C; el mismo mapeo para dev, completo y Letterboxd. |
| H4/H5 | Parquet para tablas, `.npy` para matrices y CSR, `config.yaml`; etapas con caché (`--forzar`) | C lee los `.npy` sin parser complicado; cada etapa se repite sola. |
| H6 | Preprocesamiento en Python | Se ejecuta una vez; la eficiencia evaluada es la inferencia. |
| H7 | Todos los parámetros y semillas en `config.yaml` | Reproducibilidad (D10). |
| H11 | pytest por etapa + test de extremo a extremo con una muestra mínima + comprobación automática de fugas en toda la pipeline | La de fugas ya existe en `test_senal.py`; se extiende. |
| H12 | A git: código, docs, `reports/`, `config.yaml`, `meta_*.json` y vocabularios. `data/` sigue fuera | Los meta documentan cada ejecución y pesan poco. |

### Bloque G: Letterboxd (16/10/2026)

| # | Decisión | Por qué |
|---|---|---|
| G1 | Emparejar por título + año exactos (normalizados) y lo que falle, con la búsqueda de la API de TMDB | Exacto primero (sin falsos positivos); la API rescata títulos traducidos o con «The» al final. Se registra el % emparejado. |
| G2 | Prueba con un dataset scrapeado de Letterboxd (**pendiente**: ubicación) y con los exports de Arnau y Uriel; los tests, con un export simulado | Datos personales: fuera de git (`data/external/letterboxd/`). |
| G3 | `ratings.csv` + `diary.csv` | Fechas y re-visionados (C12: la última nota). |
| G4 | media_u de un usuario nuevo: encogida hacia la media global si tiene < 20 notas, (Σr + λ·μ)/(n + λ) con λ = 10 | Con pocas notas la media cruda es ruidosa. Solo en inferencia. |
| G5 | Películas fuera del catálogo: se ignoran (v1) | Calcular su v_i con la torre de ítem queda para la E3. |
| G6 | Arranque en frío de usuarios: experimento aparte en MovieLens (10 % de usuarios fuera del entrenamiento; se les da el 80 % de su historial y se predice el 20 %) + demo cualitativa con exports reales | D1 c no lo mide; con F1 b no hay que reentrenar por usuario. |

### Bloque H: entrenamiento e inferencia (16/10/2026)

| # | Decisión | Por qué |
|---|---|---|
| H8 | Entrenar en remoto (Colab, GPU) con un notebook sencillo; dev también en local (CPU). El código no depende del dispositivo | El portátil no tiene GPU NVIDIA. |
| H9 | Búsqueda exacta en C como motor; HNSW con `hnswlib` (Python) como comparación en los benchmarks (lo hace Uriel) | Con 23 335 películas la búsqueda exacta ya cumple < 5 ms; HNSW se mide (velocidad y recall del ANN). |
| H10 | Inferencia en C (no C++) con OpenMP: lee v_i y los pesos del MLP de usuario en binario (+ `meta.json`); torre de usuario, top-200 exacto y MMR en C | Arnau y Uriel saben C; gcc 13.1 con `-fopenmp`. Benchmarks desde Python con `ctypes`. |
