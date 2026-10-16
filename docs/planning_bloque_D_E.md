# Planning de los bloques D (split) y E (features de la torre de películas)

Estado: **decidido el 16/10/2026** → ver `decisiones_preprocesamiento.md` (bloques D y E). Versión corta de `docs/guia_bloque_D_E/guia_bloque_D_E.pdf`. Las cifras salen de
`reports/split_y_features.md` (`python -m src.analisis.split_y_features`, submuestra de desarrollo: 3 154 155 ratings,
20 085 usuarios, 22 787 películas con rating; semilla 42). D1–D3 y E1–E4 son las preguntas originales de
`docs/planning_bloques_D_H.md` (15/10/2026); D4–D10 y E5–E9 son nuevas. Los ⭐ **no son decisiones**.

Supuesto para E: **E = features de la torre de película** (sinopsis, géneros, créditos, keywords, numéricas), tal como lo define
`planning_bloques_D_H.md`; la torre de usuario es el bloque F. El PDF «Etapas de desarrollo del proyecto» no define los bloques (no tiene
texto oculto ni instrucciones dirigidas a una IA).

Ya decidido y que condiciona esto: C1 b (implícito binario, contrastiva con negativos del lote y logQ), C2 (r ≥ min(4,0; media_u), **T = 4,0 confirmado**),
C3 c (anti-preferencias; nota baja = r ≤ media_u − 1), C4 d (w = 1 + max(0, r − media_u)), C5 a (cada positivo es un ejemplo), C6 a (media cruda, solo train),
C7 c (relevante con la definición de train y con r ≥ 4,0), C8 b, C9 c (×0,75 a las primeras 24 h), C10 a, C11 b, C12 a, C13 b (P(i) con positivos de train, +1), C14 c (máscara de lo valorado en train).

## Hoja de respuestas (rápida)

| # | Pregunta | Opciones | ⭐ | Depende de / afecta C |
|---|---|---|---|---|
| D1 | Tipo de split | a) aleatorio · b) temporal global · c) último por usuario · d) por usuarios (80 % historia, predecir 20 %) | d | **F1**; C1, C5, C8, C14 |
| D2 | Orden en el usuario de test | a) cronológico (desempate por movieId) · b) aleatorio · c) cronológico + control aleatorio | a | C9, C12 |
| D3 | Estadísticas solo con train | a) todo con train (fit/transform), re-ajuste train+val al final · b) siempre solo train · c) con todo | a | C6, C13, C14, E2, E4 |
| D4 | Validación | a) usuarios de validación como el test · b) 10 % de ratings de train · c) CV por usuarios · d) submuestra de val | a | C7, C8 |
| D5 | Primer día y empates | a) nada, desempatar con movieId · b) objetivo solo tras 24 h · c) usuarios de un día: 80/20 aleatorio · d) quitar usuarios con corte en empate | a | C9 c, C12 |
| D6 | Métricas y K | a) @20 · b) Recall@200 + NDCG@20 · c) K ∈ {10,20,50,200} + cobertura y novedad · d) 100 negativos | c | C7 c, C8 b, C14 c |
| D7 | Tamaño de las particiones | a) 80/10/10 de usuarios · b) 90/5/5 · c) número fijo (10 000) | a | C5, C8 |
| D8 | Arranque en frío | a) ignorar · b) quitar del test · c) reportar aparte · d) c + experimento de películas retiradas | d | E6; C11, C13 |
| D9 | Submuestra o completo | a) submuestra, completo al final · b) submuestra + 2–3 configs en completo · c) completo | b | C8, C13 |
| D10 | Semillas | a) una · b) hash estable de userId + semilla, 5 semillas · c) semilla por componente | b | C1, C13 |
| E1 | Embedding de la sinopsis | a) MiniLM (384) · b) TF-IDF + SVD · c) ambos | a | E8 |
| E2 | Créditos | a) nada · b) director + 3–5 actores con frecuencia mínima / hashing · c) todo | b | E9, D3 |
| E3 | Keywords | a) no · b) añadirlas al texto de MiniLM · c) multi-hot con frecuencia mínima | b | E1 |
| E4 | Numéricas | a) año y duración · b) todas · c) a + popularidad como sesgo de entrenamiento · d) ninguna | a | C13 (logQ) |
| E5 | Géneros (B3) | a) MovieLens · b) TMDB · c) unión (TMDB + relleno ML; sin IMAX ni Foreign) · d) 16 comunes | c | inferencia (Letterboxd) |
| E6 | ID de película | a) solo contenido · b) contenido + ID · c) contenido + ID con dropout | c | D8, F1 |
| E7 | Combinar y normalizar | a) sin normalizar · b) L2 en u y v_i + temperatura · c) solo v_i | b | MMR, C13 |
| E8 | Encoder | a) congelado + capa densa · b) ajustar últimas capas · c) ajustar todo | a | E1 |
| E9 | Datos que faltan | a) token «desconocido» + bandera + dropout · b) descargar créditos de la API · c) excluir | a | E2, E4 |

## Cifras clave (submuestra; splits 80/10/10)

| Medida | Aleatorio | Último/usuario | Temporal global | Por usuarios |
|---|---:|---:|---:|---:|
| Ratings train / val / test | 2 523 007 / 315 963 / 315 185 | 2 505 121 / 324 517 / 324 517 | 2 523 324 / 315 429 / 315 402 | 2 528 833 / 64 835 / 61 867 (+498 620 de contexto) |
| Usuarios de test sin nada en train | 0 | 0 | 1 809 de 2 259 | 2 069 de 2 069 |
| % test en películas sin train | 0,06 | 0,43 | 13,24 | 0,15 |
| % test en películas con < 5 en train | 0,90 | 2,73 | 15,63 | 1,54 |
| % de test anterior al último rating visible | 98,9 | 0 | 0 | 0 |
| % de test el mismo día del último rating visible | 1,0 | 50,2 | 0 | 62,2 |
| Usuarios elegibles en test (C2, ≥ 1 positivo en test e historia) | 18 759 | 19 396 | 439 | 2 051 |
| Techo de Recall@10 / @20 / @200 (%) | 87,5 / 95,4 / 100 | 88,7 / 96,0 / 100 | 55,1 / 71,1 / 98,5 | 77,1 / 89,9 / 100 |

- Corte temporal global: train hasta 22/07/2018, val hasta 22/10/2020 (completo: 01/09/2018 y 21/10/2020).
- Empates: en el split «último por usuario», el corte cae en un empate de segundo en el 17,7 % de los usuarios (train/val), 16,3 % (val/test), 24,4 % alguno.
- Fuga (split último): el 1,46 % de las etiquetas de train cambiaría si media_u usara todo el historial; 1 730 películas tendrían un P(i) gracias a test (Spearman de log P 0,972).
- Semillas (5): split por usuarios → ratings de test 66 565 ± 1 569; elegibles 2 005 ± 38; aleatorio 315 203 ± 223.
- Películas con < 20 ratings en train: 15 422 de 23 335; con < 5 positivos 12 872; con 0 positivos 4 354 (el frío de la submuestra es peor que el del completo).

## Cobertura de features (23 335 películas)

| Feature | % películas | % ratings |
|---|---:|---:|
| Géneros ML / TMDB / alguna | 99,0 / 98,7 / 99,8 | 99,9 / 99,8 / 100 |
| Sinopsis | 99,5 | 99,9 |
| Director / reparto (credits.csv) | 84,8 / 83,7 | 97,0 / 96,9 |
| Keywords | 84,7 | 98,5 |
| Año / duración | 99,7 / 84,7 | 99,9 / — |

- **Películas ≥ 2017** (3 361): director 9,2 %, reparto 9,0 %, duración 8,9 %, keywords 91,9 %, sinopsis 99,2 %. Hasta 2016: director ≈ 98 %.
- Géneros: ML 19 distintos, TMDB 23; Jaccard medio 0,75 sobre los 16 comunes; listas idénticas 50,2 %; solo en ML 258 películas, solo en TMDB 201, en ninguna 38.
- Créditos: 7 827 directores (40,5 % de las películas con director de ≥ 5 películas); 22 100 actores en el top-3, 1 114 con ≥ 10 películas (36,4 % de las apariciones).
- Sinopsis: mediana 46 palabras (p5 16, p95 114); MiniLM 23 335 × 384 = 17,9 MB en float16, 35,8 MB en float32.
- Tabla de ID de película 23 335 × 128 = 2 986 880 parámetros frente a 1 491 641 positivos de train (≈ 0,50 por parámetro).

## Bloque D

### D1. Tipo de split
- a) Aleatorio: el 98,9 % del test es anterior al último rating visible (entrena con el futuro).
- b) Temporal global: el 86,3 % del test es de usuarios sin historia; solo 439 elegibles; 13,2 % del test en películas sin train.
- c) Último por usuario: estándar en NCF/SASRec; 50,2 % del test el mismo día que el último train; empates en el corte en el 24,4 % de los usuarios.
- d) ⭐ Por usuarios (generalización fuerte, como Mult-VAE): usuarios apartados con 80 % de contexto y 20 % a predecir. Es el escenario Letterboxd. Exige torre de usuario basada en historial (F1 b).
- Afecta C: C1 b, C5 a, C8 b, C14 c.

### D2. Orden en los usuarios de test
- ⭐ Cronológico con desempate fijo; control con aleatorio si la diferencia importa. El 62,2 % de los ratings a predecir es del mismo día que el último de contexto. Afecta C9 c y C12 a.

### D3. Estadísticas solo con train
- Entran: media_u (C6), P(i) (C13), máscara (C14), vocabularios y frecuencias mínimas (E2), media/desviación de año y duración (E4). Patrón fit/transform; ⭐ re-ajustar con train+val solo para el modelo final.

### D4. Validación
- ⭐ Usuarios de validación apartados como el test (≈ 2 056 en la submuestra); criterio de parada Recall@200 o NDCG@20 sobre el catálogo completo (≈ 47 millones de productos escalares por pasada).

### D5. Primer día y empates
- ⭐ Aceptar el orden, desempatar con movieId y documentarlo; C9 c ya baja el peso en entrenamiento y la evaluación va sin ponderar.

### D6. Métricas y K
- ⭐ K ∈ {10, 20, 50, 200}, cobertura@20 y novedad@20, siempre sobre el catálogo completo (nada de 100 negativos). Principal: Recall@200 (HNSW) y NDCG@20 (antes y después del MMR), con las dos definiciones de relevante (C7 c) e intervalos por bootstrap de usuarios.

### D7. Tamaño
- ⭐ 80/10/10 de usuarios (≈ 2 000 de val y test en la submuestra, ≈ 20 000 en el completo). Mult-VAE usa 10 000 + 10 000 de 136 677 usuarios.

### D8. Arranque en frío
- ⭐ Reportar aparte (ratings en películas con < 5 / < 20 en train) y un experimento de películas retiradas (quitar de train todos los ratings de un 5 % de películas). Sin él, solo el 1,54 % del test del split por usuarios es de películas con < 5 en train.

### D9. Submuestra o completo
- ⭐ Desarrollar en submuestra y repetir 2–3 configuraciones finales en el completo antes de cada entrega.

### D10. Semillas
- ⭐ Grupo de cada usuario por hash estable (userId + semilla), tabla guardada como artefacto, 5 semillas en los resultados finales (media ± desviación). Todo va al archivo de configuración.

## Bloque E

### E1. Embedding de la sinopsis
- ⭐ MiniLM (all-MiniLM-L6-v2), precalculado en .npy (float16 si hace falta memoria). 127 sinopsis vacías: vector de ceros y bandera. TF-IDF + SVD como ablación.

### E2. Créditos
- ⭐ Director + 3–5 actores con frecuencia mínima (≥ 5 películas; el resto a `<OTRO>`) o hashing; el vocabulario se cuenta con las películas de train (D3).

### E3. Keywords
- ⭐ Añadirlas al texto antes de MiniLM (sin columnas nuevas). Cobertura 84,7 % (91,9 % en ≥ 2017).

### E4. Numéricas
- ⭐ Año y duración normalizados con train (o por décadas). Sin popularidad ni vote_average: con C13 b, logQ ya lleva la popularidad a la pérdida, y como feature da un atajo hacia lo popular (el 10 % más visto reúne el 78,8 % de los ratings).

### E5. Géneros (B3)
- ⭐ Unión: TMDB primero y MovieLens como relleno, sin IMAX ni Foreign; vocabulario común de 16 mapeado. Motivo: las películas nuevas y las de Letterboxd solo tendrán géneros de TMDB.

### E6. ID de película
- ⭐ Contenido + embedding de ID con dropout del ID y un ID «desconocido» para películas nuevas. Se decide con el experimento de D8.

### E7. Combinar y normalizar
- ⭐ Concatenar + red densa; L2 en u y v_i con temperatura (Yi et al.); así HNSW (producto escalar) y MMR (coseno) coinciden.

### E8. Encoder
- ⭐ Congelado con una capa densa entrenable encima.

### E9. Datos que faltan
- ⭐ Token «desconocido» + bandera `falta_credito` + dropout de los créditos al entrenar. Opción b (descargar los créditos de 3 361 películas ≥ 2017 con la API de TMDB) como mejora posterior.

## Qué depende de qué

| Pregunta | Depende de | Porque |
|---|---|---|
| D1 | F1, C1 | el split por usuarios exige torre de usuario sobre el historial |
| D2, D5 | C9 | el orden del primer día es el de tecleo |
| D3 | C6, C13, C14 | definen qué se calcula con train |
| D4, D6 | C7, C8 | dos definiciones de relevante y usuarios elegibles |
| D8 | E6 | el frío de películas solo se mide con películas retiradas |
| E1–E3 | D3 | vocabularios, frecuencias e idf solo con train |
| E4 | C13 | logQ y popularidad como feature se pisan |
| E6 | D8, F1 | decide si la torre funciona con solo contenido |
| E9 | E2, E4 | las películas ≥ 2017 no tienen créditos |

Orden propuesto: D1 + F1 + E6 juntas; luego D3 y E1–E4; después D2, D4, D5, D7; E5 y E7–E9; D6 y D8–D10 al escribir el pipeline de evaluación y el archivo de configuración.
