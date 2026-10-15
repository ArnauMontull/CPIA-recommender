# Señal objetivo: qué significa «le gusta» (bloque C)

Generado por `python -m src.analisis.senal_objetivo --muestra ratings_dev_10pct.parquet --frac-test 0.2`. Submuestra: 3 154 155 ratings, 20 085 usuarios, 22 787 películas, nota media 3,542.
«Test» = último 20 % de los ratings de cada usuario por fecha (corte provisional: lo decide el bloque D). «Top 10 %» = el 10 % de películas con más ratings en la muestra.

## 1. Sesgo de cada usuario

| Medida | p5 | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| media del usuario | 2,90 | 3,42 | 3,73 | 4,03 | 4,46 |
| desviación típica del usuario | 0,51 | 0,73 | 0,89 | 1,07 | 1,37 |

- Usuarios con media < 3,0: **6,9 %**; con media ≥ 4,0: **27,4 %**.
- Usuarios que ponen siempre la misma nota (desviación 0): 30; con desviación < 0,5: 904 (4,5 %).
- Usuarios que nunca usan medias estrellas: 6 891 (34,3 %).
- Correlación entre log(nº de ratings) y la media del usuario: -0,19.
- Histograma: `reports/senal_objetivo/media_usuario.csv`.

- Primer año con medias estrellas: **2003**. Ratings anteriores: 766 184 (24,3 %), de los que son 3 o 4 «enteros» el 63,9 %. Por año: `reports/senal_objetivo/por_anio.csv`.

## 2. Positivos con cada definición

| Esquema | % ratings | Pos./usuario (mediana) | p10 | Usuarios sin pos. | Usuarios < 5 pos. | Usuarios sin pos. en test | % último rating positivo | Películas con ≥ 1 pos. | Películas con ≥ 20 pos. | % pos. en top 10 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r ≥ 3,0 | 82,2 | 63 | 21 | 6 | 22 | 47 | 86,7 | 21 862 | 8 130 | 80,9 |
| r ≥ 3,5 | 63,4 | 49 | 15 | 17 | 115 | 249 | 69,8 | 20 706 | 6 911 | 82,2 |
| r ≥ 4,0 | 50,0 | 42 | 13 | 26 | 196 | 374 | 60,1 | 18 872 | 5 805 | 83,8 |
| r ≥ 4,5 | 23,7 | 20 | 5 | 229 | 2 002 | 2 490 | 32,2 | 15 167 | 3 689 | 85,9 |
| r = 5,0 | 14,5 | 12 | 2 | 884 | 4 336 | 5 093 | 21,0 | 12 704 | 2 703 | 86,9 |
| r > media_u | 54,4 | 40 | 13 | 30 | 63 | 293 | 56,8 | 20 514 | 6 568 | 81,6 |
| r ≥ media_u | 54,7 | 40 | 13 | 0 | 23 | 234 | 57,4 | 20 543 | 6 581 | 81,6 |
| r ≥ media_u + 0,5 | 30,4 | 22 | 6 | 833 | 1 533 | 2 249 | 32,9 | 17 702 | 4 621 | 83,1 |
| z_u ≥ 0,5 | 33,8 | 25 | 8 | 278 | 596 | 1 305 | 36,7 | 18 041 | 4 836 | 83,1 |
| r ≥ max(3,5; media_u) | 51,7 | 39 | 12 | 17 | 119 | 350 | 55,9 | 19 929 | 6 229 | 82,3 |
| r ≥ min(4,0; media_u) | 59,4 | 46 | 15 | 0 | 19 | 151 | 65,1 | 20 692 | 6 777 | 81,9 |
| residuo r − (μ+b_u+b_i) > 0 | 53,4 | 37 | 13 | 1 | 24 | 195 | 55,8 | 20 826 | 7 017 | 79,5 |

Referencia: en todos los ratings, el top 10 % de películas reúne el 78,8 %; el top 10 % de usuarios más activos aporta el 46,6 % de los ratings (columna `pct_pos_usuarios_top10` del CSV para cada esquema). Todo en `reports/senal_objetivo/umbrales.csv`.

## 3. Tasa de positivos según la popularidad de la película

Decil 1 = el 10 % de películas más vistas; decil 10 = el 10 % menos vistas (por número de películas, no de ratings).

| Decil | Ratings | Nota media | % ≥ 3,5 | % ≥ 4,0 | % > media_u |
|---|---:|---:|---:|---:|---:|
| 1 | 2 484 756 | 3,62 | 66,1 | 53,3 | 56,4 |
| 2 | 373 563 | 3,29 | 54,0 | 39,8 | 47,2 |
| 3 | 141 500 | 3,24 | 52,5 | 37,5 | 46,9 |
| 4 | 65 771 | 3,23 | 52,7 | 36,8 | 47,8 |
| 5 | 35 640 | 3,19 | 51,6 | 35,0 | 47,4 |
| 6 | 20 892 | 3,14 | 50,1 | 32,5 | 46,4 |
| 7 | 13 494 | 3,09 | 48,9 | 30,8 | 45,9 |
| 8 | 9 308 | 3,09 | 49,1 | 31,1 | 46,0 |
| 9 | 6 033 | 3,04 | 47,1 | 31,1 | 44,6 |
| 10 | 3 198 | 3,01 | 47,7 | 30,0 | 44,1 |

Correlación entre log(nº de ratings) y nota media de la película (películas con ≥ 50 ratings): **0,26**.

## 4. Notas bajas

| Definición | % ratings | Ratings | % usuarios con ≥ 1 | Mediana por usuario | % usuarios con ≥ 5 | % en top 10 % |
|---|---:|---:|---:|---:|---:|---:|
| r ≤ 2,0 | 12,7 | 399 133 | 85,6 | 5 | 53,7 | 68,8 |
| r ≤ 2,5 | 17,8 | 560 863 | 90,2 | 7 | 62,2 | 69,1 |
| r ≤ 3,0 | 36,6 | 1 155 877 | 97,7 | 20 | 87,8 | 72,8 |
| r ≤ media_u − 1 | 13,4 | 421 901 | 95,9 | 8 | 70,3 | 71,8 |
| residuo < −1 | 11,8 | 371 885 | 96,3 | 8 | 69,5 | 76,1 |

### Negativos del lote (*in-batch*) frente a negativos explícitos

Si los negativos se muestrean como en un lote (proporcional a la frecuencia de positivos con r ≥ 4,0), probabilidad de que la película sacada para un usuario sea…

| … | media por usuario (%) | mediana (%) | p90 (%) |
|---|---:|---:|---:|
| un positivo suyo (falso negativo) | 7,21 | 5,10 | 16,04 |
| una nota baja suya (r ≤ 2,5) | 1,10 | 0,37 | 2,89 |
| algo que ya valoró | 11,08 | 7,52 | 25,09 |

## 5. Ponderar los positivos (sobre positivos r ≥ 3,5)

ESS = tamaño efectivo de muestra (Σw)²/(N·Σw²): 100 % = todos pesan igual; cuanto más bajo, más ruido en el gradiente.

| Pesos | ESS (%) | % del peso en notas 5,0 | % del peso en top 10 % películas | % del peso en top 10 % usuarios |
|---|---:|---:|---:|---:|
| w = 1 | 100,0 | 22,9 | 82,2 | 42,6 |
| w = r − 2,5 (3,5→1 … 5→2,5) | 91,2 | 33,7 | 83,3 | 39,8 |
| w = 2^(r − 3,5) (1 … 2,83) | 87,3 | 37,3 | 83,5 | 39,3 |
| w = 1 + max(0, r − media_u) | 90,2 | 31,2 | 82,5 | 43,9 |
| w = 1/√n_u (iguala usuarios) | 73,4 | 27,0 | 86,6 | 22,0 |
| w = (r − 2,5)/√n_u | 64,0 | 38,4 | 87,3 | 20,0 |

## 6. logQ: P(i) con todos los ratings o solo con los positivos

Ratio = P_pos(i) / P_todos(i) para películas con ≥ 100 ratings: si fuera 1 siempre, daría igual cómo se cuente.

| Positivos | Spearman log P | Películas con 0 positivos (P = 0) | Ratio p5 | Ratio p95 |
|---|---:|---:|---:|---:|
| r ≥ 3,5 | 0,944 | 2 081 | 0,35 | 1,33 |
| r ≥ 4,0 | 0,922 | 3 915 | 0,29 | 1,46 |
| r > media_u | 0,939 | 2 273 | 0,33 | 1,40 |

Entre las 300 más vistas, las que más bajan (y suben) de rango al contar solo positivos r ≥ 4,0:

| Película | Rango (todos) | Rango (positivos) | % ≥ 4,0 |
|---|---:|---:|---:|
| Charlie's Angels (2000) | 242 | 913 | 18,4 |
| Ace Ventura: When Nature Calls (1995) | 202 | 694 | 20,7 |
| Nutty Professor, The (1996) | 244 | 679 | 24,1 |
| Mission: Impossible II (2000) | 253 | 681 | 24,2 |
| Addams Family Values (1993) | 271 | 694 | 25,2 |
| 12 Angry Men (1957) | 254 | 133 | 82,9 |
| North by Northwest (1959) | 285 | 168 | 78,0 |
| Chinatown (1974) | 282 | 172 | 76,2 |
| Rear Window (1954) | 210 | 115 | 79,8 |
| City of God (Cidade de Deus) (2002) | 198 | 112 | 78,9 |

## 7. Tiempo

- Ratings hechos en las primeras 24 h del usuario: **62,5 %**. Usuarios con ≥ 50 % de sus ratings el primer día: 82,6 %; con todos el primer día: 59,0 %.
- Nota media el primer día: 3,603; después: 3,441. % ≥ 4,0: 53,1 % frente a 45,0 %.
- Ratings que comparten segundo exacto con otro del mismo usuario: 22,3 % (el orden temporal dentro del empate no existe). Instantes distintos por usuario (mediana): 56.
- Ratings más de 1 año antes del estreno (flag `antes_estreno`): 42 en la muestra, nota media 3,46, 59,5 % ≥ 4,0.
- Re-ratings: MovieLens solo guarda la última nota de cada par (usuario, película) (0 pares repetidos en B4); los únicos «re-ratings» fueron los 836 de duplicados fusionados (B1b), donde ya se quedó la más reciente.

## 8. Coincidencia entre definiciones

| Comparación | % ratings |
|---|---:|
| r ≥ 4,0 pero residuo ≤ 0 (le gustó menos de lo esperable) | 8,3 |
| r < 4,0 pero residuo > 0 (le gustó más de lo esperable) | 11,7 |
| r ≥ 4,0 pero r ≤ media_u | 4,9 |
| r < 4,0 pero r > media_u | 9,3 |

Sesgos con encogimiento (λ_u = 10, λ_i = 25): b_u entre -0,56 y 0,77 (p5–p95); b_i entre -0,49 y 0,26.
