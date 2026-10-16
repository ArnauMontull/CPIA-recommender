# Split (bloque D) y features de la torre de ítem (bloque E)

Generado por `python -m src.analisis.split_y_features --muestra ratings_dev_10pct.parquet --semilla 42`. Submuestra: 3 154 155 ratings, 20 085 usuarios, 22 787 películas.
Todos los splits son 80/10/10. «Positivo» = definición provisional de C2: r ≥ min(4,0; media_u), con media_u de la historia visible (train, y contexto en el split por usuarios). «Elegible» = usuario con ≥ 1 positivo en la partición evaluada y ≥ 1 en su historia visible.

## D1. Tamaño de las particiones

| Medida | Aleatorio | Último por usuario | Temporal global | Por usuarios |
|---|---:|---:|---:|---:|
| Ratings en train | 2 523 007 | 2 505 121 | 2 523 324 | 2 528 833 |
| Ratings en val | 315 963 | 324 517 | 315 429 | 64 835 |
| Ratings en test | 315 185 | 324 517 | 315 402 | 61 867 |
| Ratings de contexto (no entran en la pérdida) | 0 | 0 | 0 | 498 620 |
| Usuarios con ratings en train | 20 085 | 20 085 | 16 958 | 15 960 |
| Usuarios con ratings en val | 19 715 | 20 085 | 2 352 | 2 056 |
| Usuarios con ratings en test | 19 690 | 20 085 | 2 259 | 2 069 |

Corte temporal global: train hasta 2018-07-22, val hasta 2020-10-22, test después.

## D2. Arranque en frío en cada split

| Medida | Aleatorio | Último por usuario | Temporal global | Por usuarios |
|---|---:|---:|---:|---:|
| Usuarios de test sin ningún rating en train | 0 | 0 | 1 809 | 2 069 |
| % de ratings de test de usuarios sin train | 0,0 | 0,0 | 86,3 | 100,0 |
| % de ratings de test en películas sin ningún rating en train | 0,06 | 0,43 | 13,24 | 0,15 |
| Películas distintas sin train que salen en test | 168 | 704 | 3 180 | 88 |
| % de ratings de test en películas con < 5 en train | 0,90 | 2,73 | 15,63 | 1,54 |
| % de ratings de test en películas con < 20 en train | 3,32 | 7,72 | 19,43 | 5,35 |
| Películas sin ningún positivo en train (P(i) = 0) | 2 790 | 3 806 | 6 045 | 2 885 |
| % de positivos de test en películas sin positivo en train | 0,22 | 0,90 | 12,90 | 0,43 |

## D3. ¿Test está en el futuro del usuario? (tiempo)

Para los ratings de test (y val) con historia visible: posición respecto al último rating de la historia visible del mismo usuario.

| Medida | Aleatorio | Último por usuario | Temporal global | Por usuarios |
|---|---:|---:|---:|---:|
| % de ratings de test ANTES del último rating visible (ve el futuro) | 98,9 | 0,0 | 0,0 | 0,0 |
| % de ratings de test el mismo día (< 24 h después) | 1,0 | 50,2 | 0,0 | 62,2 |
| % de ratings de test más de 24 h después | 0,1 | 49,8 | 100,0 | 37,8 |
| % de ratings de train hechos en las primeras 24 h del usuario | 62,6 | 68,4 | 62,9 | 62,4 |
| % de ratings de test hechos en las primeras 24 h del usuario | 62,5 | 36,7 | 58,0 | 42,5 |

### Último por usuario: empates en el corte

- Usuarios cuyo corte train/val cae dentro de un empate de segundo: 3 561 (17,7 %); corte val/test: 3 273 (16,3 %); alguno de los dos: 4 901 (24,4 %). En ellos «el último» depende del desempate (aquí, movieId).

### Temporal global: en qué particiones aparece cada usuario

| Particiones | Usuarios | % |
|---|---:|---:|
| train | 16 175 | 80,5 |
| test | 1 462 | 7,3 |
| val | 1 318 | 6,6 |
| test + train + val | 354 | 1,8 |
| test + val | 347 | 1,7 |
| train + val | 333 | 1,7 |
| test + train | 96 | 0,5 |

## D4. Usuarios elegibles y techo de las métricas (test)

| Medida | Aleatorio | Último por usuario | Temporal global | Por usuarios |
|---|---:|---:|---:|---:|
| Usuarios de test con ≥ 1 positivo en test y en su historia (C2) | 18 759 | 19 396 | 439 | 2 051 |
|   % de los usuarios de test | 95,3 | 96,6 | 19,4 | 99,1 |
| Usuarios de test que salen de las métricas (C8, C2) | 931 | 689 | 1 820 | 18 |
| Usuarios elegibles con r ≥ 4,0 fijo (C7) | 18 429 | 19 052 | 436 | 2 027 |
| Positivos totales en test de los elegibles | 187 353 | 185 379 | 21 786 | 35 019 |
| Positivos por usuario elegible (mediana) | 5 | 5 | 20 | 9 |
| % de elegibles con < 3 positivos en test | 24,9 | 20,0 | 10,3 | 6,5 |
| Techo de Recall@10 (%) | 87,5 | 88,7 | 55,1 | 77,1 |
| Techo de Recall@20 (%) | 95,4 | 96,0 | 71,1 | 89,9 |
| Techo de Recall@50 (%) | 99,3 | 99,4 | 86,9 | 97,8 |
| Techo de Recall@200 (%) | 100,0 | 100,0 | 98,5 | 100,0 |

Techo de Recall@K = media de min(K, |R_u|)/|R_u| sobre los usuarios elegibles: el Recall@K máximo alcanzable aunque el modelo fuera perfecto.

## D5. Cuánta fuga habría (split «último por usuario»)

- Media de usuario con todo el historial frente a solo train: diferencia absoluta media 0,062 (p95 0,178, máx. 0,92).
- Ratings de train cuya etiqueta de positivo cambia si media_u usa todo el historial: **1,46 %** (36 534 ratings); en test: 1,25 %.
- Películas con algún positivo en la muestra completa pero ninguno en train: 1 730 (con P(i) calculada con todo se les daría un P(i) que solo existe gracias a test). Spearman de log P(i) entre ambos conteos: 0,972.

## D6. Variación entre semillas (5 semillas, media ± desviación típica)

| Split | Ratings de test | Usuarios de test | Elegibles | Positivos en test | % test en películas con < 5 en train | % test en películas sin train |
|---|---:|---:|---:|---:|---:|---:|
| Aleatorio por rating | 315 203 ± 223 | 19 711 ± 20 | 18 839 ± 11 | 187 264 ± 281 | 0,9 ± 0,0 | 0,1 ± 0,0 |
| Por usuarios | 66 565 ± 1 569 | 2 025 ± 38 | 2 005 ± 38 | 37 716 ± 494 | 1,8 ± 0,5 | 0,2 ± 0,1 |

## D7. Submuestra frente a conjunto completo

- Corte temporal global (cuantiles 80/90 de los timestamps): submuestra 2018-07-22 / 2020-10-22; completo 2018-09-01 / 2020-10-21.
- Completo: 31 725 154 ratings, 200 948 usuarios. La submuestra tiene el 9,9 % de los ratings.

## E1. Cobertura de cada feature de la torre de ítem

Catálogo: 23 335 películas (22 787 con algún rating en la submuestra). «% ratings» = ponderado por los ratings de la submuestra.

| Feature | % películas | % ratings |
|---|---:|---:|
| Géneros de MovieLens (no vacíos) | 99,0 | 99,9 |
| Géneros de TMDB (no vacíos) | 98,7 | 99,8 |
| Géneros de alguna de las dos | 99,8 | 100,0 |
| Sinopsis | 99,5 | 99,9 |
| Director (credits.csv) | 84,8 | 97,0 |
| Reparto (credits.csv) | 83,7 | 96,9 |
| Palabras clave (keywords.csv o API) | 84,7 | 98,5 |
| Año de estreno | 99,7 | 99,9 |

### Cobertura por época de estreno

| Época | Películas | % con director | % con reparto | % con keywords | % con géneros TMDB | % con sinopsis | Mediana de ratings (completo) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤1989 | 6 336 | 98,0 | 97,1 | 87,2 | 99,1 | 99,7 | 114 |
| 1990-99 | 3 424 | 97,5 | 96,5 | 81,8 | 97,4 | 99,3 | 370 |
| 2000-09 | 5 432 | 97,8 | 96,3 | 82,5 | 98,8 | 99,5 | 155 |
| 2010-16 | 4 709 | 97,7 | 96,2 | 80,8 | 99,0 | 99,5 | 76 |
| ≥2017 | 3 361 | 9,2 | 9,0 | 91,9 | 99,1 | 99,2 | 58 |

Fuente de la sinopsis: kaggle = 19 804, api = 3 391,  = 127, correccion = 13.

## E2. Géneros de MovieLens frente a los de TMDB (decisión B3)

- Géneros distintos en el catálogo filtrado: MovieLens 19, TMDB 23.
- Géneros por película (media): MovieLens 2,09, TMDB 2,36.
- Películas con géneros en MovieLens y no en TMDB: 258; en TMDB y no en MovieLens: 201; en ninguna: 38.
- En las 22 838 películas con las dos listas, sobre los 16 géneros comunes (Sci-Fi = Science Fiction, Children ≈ Family): Jaccard medio **0,75**; listas idénticas en el **50,2 %**.
- Películas con IMAX, Film-Noir o Musical (solo ML): 1 065; con Music, History, Foreign o TV Movie (solo TMDB): 2 597.

## E3. Películas poco vistas (split «último por usuario»; submuestra)

| Grupo | Películas | % con sinopsis | % con director | % con géneros ML |
|---|---:|---:|---:|---:|
| Todas | 23 335 | 99,5 | 84,8 | 99,0 |
| < 20 ratings en train | 15 422 | 99,3 | 79,7 | 98,5 |
| < 5 positivos en train | 12 872 | 99,3 | 77,8 | 98,4 |
| 0 positivos en train | 4 354 | 99,2 | 70,8 | 97,9 |

## E4. Cardinalidades y tamaño de los parámetros

- Directores distintos: 7 827; el 40,5 % de las películas tiene un director con ≥ 5 películas en el catálogo.
- Actores distintos en el top-3 del reparto: 22 100; con ≥ 10 películas: 1 114 (36,4 % de las apariciones).
- Longitud de la sinopsis (palabras): p5 16, mediana 46, p95 114; con < 10 palabras: 0,5 %. MiniLM (all-MiniLM-L6-v2) trunca a 256 *word pieces* (≈ 190 palabras).
- Tabla de embeddings de movieId (23 335 × 128): 2 986 880 parámetros, frente a 1 491 641 positivos de train en la submuestra (≈ 0,50 positivos por parámetro).
- Correlación de Spearman entre nº de ratings (completo) y año de estreno: -0,14.

- Duración (runtime de Kaggle): disponible en el 84,7 % de las películas (mediana 100 min, p5–p95 70–142); las ≥ 2017 con duración: 8,9 %.

### Películas y mediana de ratings por década de estreno (completo)

| Década | Películas | Mediana de ratings |
|---|---:|---:|
| ≤1969 | 2 921 | 81 |
| 70s | 1 275 | 111 |
| 80s | 2 140 | 194 |
| 90s | 3 424 | 370 |
| 00s | 5 432 | 155 |
| 10s | 6 785 | 70 |
| ≥2020 | 1 285 | 56 |
