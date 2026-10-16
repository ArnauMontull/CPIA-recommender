# CPIA-recommender

Recomendador de películas **Two-Tower** entrenado con **MovieLens 32M**, metadatos de **TMDB** (Kaggle + API) y perfiles de **Letterboxd**.
Una torre codifica al usuario a partir de su historial ponderado y la otra a la película (género, director, reparto, año, duración y
embedding MiniLM de la sinopsis). Se entrena de forma contrastiva con corrección **logQ** del sesgo de muestreo, y la inferencia hace
**búsqueda exacta** más re-ranking **MMR** (diversidad) en **C con OpenMP**.

Proyecto de CPIA (UPC, Ingeniería Física) de **Arnau Montull** y **Uriel Martínez**.

## Estado

| Etapa | Entrega | Estado |
|---|---|---|
| E1 | Propuesta (`ENTREGAS/E1_PROPUESTA/main.pdf`) | ✅ entregada |
| E2 | Diseño | ⏳ 29/10/2026 |
| E3 | Implementación | ⏳ 08/12/2026 |
| E4 | Informe y vídeo | ⏳ 22/12/2026 |

**Hecho**

- ✅ Preprocesamiento bloques C–G: validación, submuestra de desarrollo, split por usuario y tiempo, señal objetivo, historiales,
  features de películas, experimento de arranque en frío y perfiles de Letterboxd. Ejecutado en `dev` y en el conjunto `completo`.
- ✅ Métricas de evaluación (`src/evaluacion/metricas.py`), con tests.
- ✅ Notebook y guía de Colab (GPU T4) y empaquetado de datos para Drive.

**Pendiente**

- ⏳ Entrenamiento del modelo Two-Tower (`src/entrenar.py` aún no existe).
- ⏳ Inferencia en C (búsqueda exacta + MMR con OpenMP).
- ⏳ Benchmarks (tiempo, escalado con hilos, calidad de las recomendaciones).

## Estructura

```
CPIA-recommender/
├── config.yaml                  parámetros de la pipeline (cada decisión de docs/ es un parámetro)
├── requirements.txt
├── src/
│   ├── preprocesamiento/        validar, descargar_tmdb, submuestra, split, senal, historial, features_peliculas, frio
│   ├── letterboxd/              export de Letterboxd -> emparejado con el catálogo -> media_u y bolsas A/B
│   ├── evaluacion/              métricas (Recall@K, NDCG@K, cobertura, novedad) sobre CSR, por lotes
│   └── analisis/                estadísticas y análisis previos a las decisiones (submuestra, señal, split)
├── tests/                       pytest con datos sintéticos hechos a mano + invariantes de dev
├── notebooks/                   entrenar_colab.ipynb
├── scripts/                     empaquetar_datos.py (zip para Drive)
├── reports/                     informes y meta_*.json de cada ejecución (dev, completo, frío)
├── docs/                        decisiones, planes, guías por bloque y papers
├── ENTREGAS/E1_PROPUESTA/       propuesta en LaTeX y PDF
└── data/                        (fuera de git) crudos, procesados y externos
```

## Instalación

```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Los datos crudos **no están en git** (`data/` está en `.gitignore`). Colócalos así:

- `data/ml-32m/` : MovieLens 32M (`ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`).
- `data/the_movies_dataset_kaggle/` : *The Movies Dataset* de Kaggle (`movies_metadata.csv`, `credits.csv`, `keywords.csv`, ...).

## Reproducir la pipeline

Todos los comandos se lanzan desde la raíz del repo. Cada etapa guarda su salida en `data/processed/` y se salta si ya existe
(salvo `--forzar`). Los tiempos son los de los informes en `reports/`; donde no hay medida, se indica.

| # | Comando | Tiempo aproximado |
|---|---|---|
| 1 | `python -m src.preprocesamiento.validar` | sin medir en los informes |
| 2 | `python -m src.preprocesamiento.descargar_tmdb` (opcional; necesita token de TMDB, ver su docstring) | sin medir; se puede interrumpir y relanzar |
| 3 | `python -m src.preprocesamiento.submuestra` | sin medir (solo `dev`: crea el 10 % por usuarios) |
| 4 | `python -m src.preprocesamiento.split --muestra dev\|completo` | dev ~4 s, completo ~37 s |
| 5 | `python -m src.preprocesamiento.senal --muestra dev\|completo` | dev ~2 s, completo ~12 s |
| 6 | `python -m src.preprocesamiento.historial --muestra dev\|completo` | dev ~1 s, completo ~8 s |
| 7 | `python -m src.preprocesamiento.features_peliculas --muestra dev\|completo` | dev ~14 min (MiniLM en CPU, una sola vez); los embeddings se reutilizan en completo (~21 s) |
| 8 | `python -m src.preprocesamiento.frio --muestra dev\|completo` | dev ~12 s |
| 9 | `python -m src.letterboxd --export <zip\|carpeta> --nombre <alias> [--sin-api]` | unos segundos (la API, solo para lo que no empareja exacto) |

Notas:

- Si hiciste el paso 2, repite el 1 para incorporar lo descargado.
- El paso 9 usa la API de TMDB para emparejar títulos que no coinciden de forma exacta; con `--sin-api` solo empareja por título y año.
- Para entrenar en Colab: `python scripts/empaquetar_datos.py --muestra completo` y sigue [`docs/colab.md`](docs/colab.md).

## Tests

```
python -m pytest tests
```

Cubren, con datos sintéticos cuyos resultados se calculan a mano: split, señal objetivo, historiales, features de películas,
experimento de frío, módulo de Letterboxd (export simulado, sin red ni credenciales) y métricas de evaluación.
`tests/test_invariantes_dev.py` comprueba sobre las salidas reales de `dev` que no hay fugas (orden temporal, media_u y logQ solo con
train, val/test sin películas frías, alineación de features); **se salta si no existe `data/processed/dev/`**. En dev son 53 tests.

## Cifras clave

Fuente: `reports/pipeline_dev.md` y `reports/pipeline_completo/*.json`.

| | dev (10 % de usuarios) | completo |
|---|---:|---:|
| Usuarios | 20 085 | 200 948 |
| Películas (catálogo) | 23 335 | 23 335 |
| Ratings train / val / test | 2 523 929 / 314 318 / 313 721 | 25 385 248 / 3 169 838 / 3 169 694 |
| % positivos train / val / test | 59,6 / 56,0 / 57,0 | 59,7 / 55,8 / 56,8 |
| % notas bajas train | 13,4 | 13,3 |
| % r ≥ 4,0 train | 50,8 | 50,8 |
| Usuarios elegibles en test | 19 190 | 192 721 |
| Ejemplos de entrenamiento (un positivo de train) | 1 503 681 | 15 159 684 |
| Bolsa A por usuario (p50 / p99) | 51 / 893 | 50 / 877 |
| Cobertura director / reparto | 99,0 % / 97,9 % | 99,0 % / 97,9 % |
| Embeddings MiniLM | 23 335 × 384 (float16) | los mismos |

Gracias a la API de TMDB, el director de las películas de 2017 en adelante pasa del 9,2 % al 99,1 % de cobertura.

## Documentación

Índice de `docs/` y `reports/`:

| Documento | Contenido |
|---|---|
| [`docs/decisiones_preprocesamiento.md`](docs/decisiones_preprocesamiento.md) | Registro de decisiones A–H y por qué; cómo reproducir |
| [`docs/plan_implementacion_CDE.md`](docs/plan_implementacion_CDE.md) | Contratos y reparto de la implementación de los bloques C, D, E y F |
| [`docs/plan_antes_entrenamiento.md`](docs/plan_antes_entrenamiento.md) | Lo que queda antes de entrenar: pipeline completa, Letterboxd, frío y Colab |
| [`docs/plan_entrenamiento.md`](docs/plan_entrenamiento.md) | Plan del entrenamiento del modelo Two-Tower |
| [`docs/colab.md`](docs/colab.md) | Guía paso a paso para entrenar en Google Colab (GPU T4) |
| [`docs/guia_H10_inferencia_C/`](docs/guia_H10_inferencia_C/) | Guía de la inferencia en C (búsqueda exacta, MMR, OpenMP) |
| [`docs/guia_bloque_C/`](docs/guia_bloque_C/) | Guía del bloque C (señal objetivo) |
| [`docs/guia_bloque_D_E/`](docs/guia_bloque_D_E/) | Guía de los bloques D y E (split y features) |
| [`docs/planning_bloque_C.md`](docs/planning_bloque_C.md), [`docs/planning_bloque_D_E.md`](docs/planning_bloque_D_E.md), [`docs/planning_bloque_F.md`](docs/planning_bloque_F.md), [`docs/planning_bloques_D_H.md`](docs/planning_bloques_D_H.md), [`docs/planning_bloques_G_H.md`](docs/planning_bloques_G_H.md) | Planning detallado de cada bloque |
| [`docs/papers/`](docs/papers/) | Papers de referencia (YouTube DNN, logQ, SASRec, MMR, EASE, LightGCN...) |
| [`reports/pipeline_dev.md`](reports/pipeline_dev.md) | Informe de la pipeline en dev |
| [`reports/validacion.md`](reports/validacion.md), [`reports/submuestra.md`](reports/submuestra.md), [`reports/senal_objetivo.md`](reports/senal_objetivo.md), [`reports/split_y_features.md`](reports/split_y_features.md) | Informes de cada análisis |
| [`ENTREGAS/E1_PROPUESTA/main.pdf`](ENTREGAS/E1_PROPUESTA/main.pdf) | Propuesta (E1) |

## Privacidad

Los exports de Letterboxd son **datos personales**. Viven en `data/external/letterboxd/` (fuera de git) y en el Drive compartido del
equipo, **nunca en este repo, que es público**. Tampoco se suben tokens ni credenciales (el token de TMDB se lee de una variable de
entorno o de un fichero fuera del repo). Los tests usan solo datos sintéticos.
