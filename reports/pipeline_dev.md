# Pipeline de preprocesamiento en dev (bloques C, D, E y F)

Ejecutado el 16/10/2026 sobre la submuestra de desarrollo (`ratings_dev_10pct.parquet`). Los `meta_*.json` y vocabularios
están copiados en `reports/pipeline_dev/`. Contratos en `docs/plan_implementacion_CDE.md`; decisiones en
`docs/decisiones_preprocesamiento.md`.

```
python -m src.preprocesamiento.split --muestra dev               # ~4 s
python -m src.preprocesamiento.senal --muestra dev               # ~2 s
python -m src.preprocesamiento.historial --muestra dev           # ~2 s
python -m src.preprocesamiento.features_peliculas --muestra dev  # ~14 min (MiniLM en CPU, una sola vez)
python -m pytest tests                                           # 53 tests, incluidos los invariantes de dev
```

## Split (D)

| | Ratings |
|---|---:|
| train | 2 523 929 |
| val | 314 318 |
| test | 313 721 |
| quitados: antes del estreno (C11) | 42 |
| quitados de val/test: película sin train (D8) | 774 + 1 371 |

20 085 usuarios, 23 335 películas. En 4 822 usuarios (24,0 %) un corte cae en un empate de segundo (desempate por movieId).

## Señal (C)

| | train | val | test |
|---|---:|---:|---:|
| % positivos (r ≥ min(4; media_u)) | 59,6 | 56,0 | 57,0 |
| % notas bajas (r ≤ media_u − 1) | 13,4 | 15,3 | 15,2 |
| % r ≥ 4,0 | 50,8 | 46,6 | 47,6 |

Usuarios elegibles en test: 19 190 (definición de train) y 18 832 (r ≥ 4,0). Ningún usuario sin positivos en train.
Corrección de revisión: 4 ratings de usuarios con media_u = 5,0 eran a la vez positivo y nota baja; ahora gana positivo.

## Historiales (F)

1 503 681 ejemplos (uno por positivo de train). Tamaño por usuario en train: bolsa A (positivos + neutras) p50 51, p99 893,
máx 3 770; bolsa B (notas bajas) p50 7, p99 163; 1 105 usuarios sin bolsa B. Ningún ejemplo queda sin historial al quitar
el objetivo. Todos los historiales: 43,5 MB.

## Features de películas (E)

| | Kaggle | + API de TMDB | Cobertura |
|---|---:|---:|---:|
| Director | 19 787 | 3 322 | 99,0 % |
| Reparto (5) | 19 533 | 3 307 | 97,9 % |
| Duración | 19 775 | 3 342 | 99,1 % |
| Keywords | 16 643 | 3 111 | 84,7 % |

Películas de 2017 en adelante: director del 9,2 % al 99,1 % gracias a la API. Géneros (16 comunes): 99,8 %. Vocabularios:
1 241 directores y 4 491 actores (≥ 5 películas en train, más 3 reservados). Embeddings MiniLM: 23 335 × 384 en float16;
125 películas sin texto (vector cero). Comprobación: los vecinos por sinopsis de *Toy Story* son *Toy Story 2, 3 y 4*.

## Invariantes comprobados (`tests/test_invariantes_dev.py`)

Orden temporal train ≤ val ≤ test por usuario; val/test sin películas frías; media_u y logQ solo con train; positivo y
nota baja nunca a la vez; máscaras e historiales cuadran con los recuentos; features y embeddings alineados con `item_idx`.
