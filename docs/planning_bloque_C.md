# Planning del bloque C: qué significa «le gusta» (la señal objetivo)

Estado: **abierto, sin decidir** (15/10/2026). Las cifras salen de `reports/senal_objetivo.md`
(`python -m src.analisis.senal_objetivo`, submuestra de desarrollo: 3 154 155 ratings, 20 085 usuarios).
Cada pregunta tiene opciones, qué implica cada una (modelo, pérdida, evaluación, E2) y una recomendación ⭐ que
**no es una decisión**: decide Arnau (y Uriel), y lo decidido pasa a `decisiones_preprocesamiento.md`.

Qué prometimos en E1 (`ENTREGAS/E1_PROPUESTA/main.tex`): Two-Tower entrenada de forma contrastiva con corrección
logQ (Yi et al. 2019), ANN (HNSW) para sacar 200 candidatos y MMR para dejar 20; y en la tabla de datos,
«binarización implícita ponderada por r_ui» → torre de usuario. Si cambiamos eso, hay que justificarlo en la E2.

## Hoja de respuestas (rápida)

| # | Pregunta | Opciones | ⭐ | Depende de |
|---|---|---|---|---|
| C1 | Tipo de señal | a) regresión de r_ui · b) implícito binario · c) binario relativo al usuario · d) multitarea | b (con el umbral de C2) | **D1, F1** |
| C2 | Umbral de positivo | a) ≥ 3,5 · b) ≥ 4,0 · c) ≥ media_u · d) ≥ min(4,0; media_u) · e) ≥ max(3,5; media_u) · f) residuo > 0 | d (y b como ablación) | D (media_u solo con train) |
| C3 | Notas bajas | a) fuera · b) negativos duros · c) anti-preferencias en la torre de usuario · d) a + c | a en v1; c como experimento | **F** |
| C4 | Ponderar positivos por la nota | a) no · b) lineal · c) exponencial · d) relativa a media_u | b | C2 |
| C5 | Peso de cada usuario | a) nada · b) 1/√n_u · c) ejemplos por usuario con tope | c (o b si F no usa historial) | **F1**, C13 |
| C6 | Cómo se calcula media_u | a) media cruda · b) media encogida · c) mediana | b, solo con train | **D** |
| C7 | Qué es «relevante» en test | a) igual que en train · b) siempre ≥ 4,0 · c) ambas | c | C2, **D** |
| C8 | Usuarios sin positivos | a) quitarlos · b) solo como historia, fuera de las métricas | b | **D** |
| C9 | Ratings en bloque el primer día | a) ignorar · b) solo como historia, no como objetivo · c) bajarles el peso | a en C (se resuelve en D) | **D** |
| C10 | Usuarios sin medias estrellas | a) nada · b) umbral por época | a (si C2 es c o d) | C2 |
| C11 | Ratings antes del estreno | a) mantener · b) quitar · c) solo historia | b | — |
| C12 | Re-ratings | (no hay en MovieLens) en Letterboxd: a) la última · b) la media | a | — |
| C13 | logQ: con qué se cuenta P(i) | a) todos los ratings · b) solo positivos del train · c) b ponderado como el muestreo | c (= b si no hay muestreo especial) | C2, C4, C5 |
| C14 | Falsos negativos del lote | a) ignorar · b) enmascarar positivos del usuario · c) enmascarar todo lo valorado | c | C3 |

## Un ejemplo con tres usuarios inventados

| Película | **Ana** (generosa, media 4,4) | **Bruno** (exigente, media 2,8) | **Carla** (2001, solo estrellas enteras, media 3,4) |
|---|---:|---:|---:|
| Pulp Fiction | — | 4,0 | 5 |
| The Matrix | 5,0 | 3,5 | 4 |
| Shrek | 4,5 | 3,0 | 4 |
| Titanic | 5,0 | 1,5 | 3 |
| Notting Hill | 4,0 | 2,0 | — |
| Batman & Robin | 3,5 | — | 1 |

Positivos según el umbral (C2):

| Umbral | Ana | Bruno | Carla |
|---|---|---|---|
| ≥ 3,5 | las 5 (incluida *Batman & Robin*, la que menos le gustó) | Pulp, Matrix | Pulp, Matrix, Shrek |
| ≥ 4,0 | 4 (todas menos *Batman & Robin*) | solo Pulp | Pulp, Matrix, Shrek (para ella 3,5 = 4,0) |
| ≥ media_u | Titanic, Matrix, Shrek (su 4,0 a *Notting Hill* ya no cuenta) | Pulp, Matrix, Shrek (su 3,0 sí cuenta) | Pulp, Matrix, Shrek |
| ≥ min(4,0; media_u) | 4 (umbral 4,0) | Pulp, Matrix, Shrek (umbral 2,8) | Pulp, Matrix, Shrek (umbral 3,4) |
| ≥ max(3,5; media_u) | Titanic, Matrix, Shrek (umbral 4,4) | Pulp, Matrix (umbral 3,5) | Pulp, Matrix, Shrek |

Lo que enseña: con un umbral fijo, Ana aporta muchos positivos «flojos» y Bruno casi ninguno; con uno relativo,
el 4,0 de Ana deja de ser positivo, que es discutible (un 4 es un 4). El híbrido min(4,0; media_u) mantiene el
significado absoluto para los generosos y rescata a los exigentes. *Titanic* es a la vez positivo (Ana),
nota baja (Bruno) y neutra (Carla): eso es justo lo que la torre de usuario tiene que aprender a separar.

---

## C1. Tipo de señal: ¿qué predice el modelo?

Datos: la nota media es 3,54 y el 50 % de los ratings son ≥ 4,0. El sesgo de cada usuario es grande: la media
de los usuarios va de 2,90 (p5) a 4,46 (p95); el 27,4 % tiene media ≥ 4,0 y el 6,9 % < 3,0.

| Opción | Modelo y pérdida | Evaluación | E2 |
|---|---|---|---|
| **a) Regresión de r_ui** (MSE sobre ⟨u, v_i⟩) | Producto escalar ≈ nota. No hay negativos: el modelo solo ve lo que el usuario valoró, y aprende «qué nota pondría *si la viera*», no «qué querrá ver». logQ no tiene sentido (no hay muestreo). Hay que añadir sesgos b_u, b_i. | RMSE/MAE (estilo Netflix Prize). Un buen RMSE no garantiza un buen top-K: las ANN sacan los mayores ⟨u, v⟩ de **todo** el catálogo, incluidas películas que nadie como él vería. | Contradice E1 (contrastiva + logQ + «binarización»): hay que justificar el cambio. |
| **b) Implícito binario** (positivo = r ≥ umbral; softmax con negativos del lote + logQ) ⭐ | Lo que dijimos en E1. Pérdida contrastiva: el positivo contra los demás del lote. El producto escalar ordena, no predice notas. Las notas solo deciden quién es positivo (C2) y cuánto pesa (C4). | Recall@K, NDCG@K, cobertura y novedad, que es lo que mide la recuperación de 200 candidatos y el MMR. | Cuadra con E1. |
| **c) Binario relativo al usuario** (positivo = r ≥ media_u) | Igual que b, cambia solo la etiqueta. Quita el sesgo de cada usuario, pero hace falta media_u (C6). | Igual que b; los positivos de test dependen de media_u del train. | Cuadra con E1. En la práctica es una opción de C2. |
| **d) Multitarea** (contrastiva + cabeza de regresión de la nota) | Dos pérdidas a equilibrar (un hiperparámetro más). La cabeza de regresión puede servir para el reranking. | Recall@K + RMSE. | Más trabajo; mejor como mejora para la E3 si sobra tiempo. |

⭐ **b**, con el umbral que salga de C2 (c es b con un umbral relativo, por eso se decide allí).
Depende de **D1** (el split decide qué positivos hay en test) y de **F1** (si la torre de usuario recibe el
historial, la misma definición decide qué películas entran en él). C1, D1 y F1 se deberían cerrar juntas.

## C2. Umbral de binarización

| Umbral | % ratings positivos | Pos./usuario (mediana; p10) | Usuarios sin pos. | Usuarios sin pos. en test* | Películas con ≥ 20 pos. | % pos. en top 10 % |
|---|---:|---:|---:|---:|---:|---:|
| a) r ≥ 3,5 | 63,4 | 49; 15 | 17 | 249 | 6 911 | 82,2 |
| b) r ≥ 4,0 | 50,0 | 42; 13 | 26 | 374 | 5 805 | 83,8 |
| r ≥ 4,5 (referencia) | 23,7 | 20; 5 | 229 | 2 490 | 3 689 | 85,9 |
| c) r ≥ media_u | 54,7 | 40; 13 | 0 | 234 | 6 581 | 81,6 |
| d) r ≥ min(4,0; media_u) | 59,4 | 46; 15 | 0 | **151** | 6 777 | 81,9 |
| e) r ≥ max(3,5; media_u) | 51,7 | 39; 12 | 17 | 350 | 6 229 | 82,3 |
| f) residuo r − (μ + b_u + b_i) > 0 | 53,4 | 37; 13 | 1 | 195 | 7 017 | 79,5 |

\* «Test» provisional = último 20 % de cada usuario por fecha. Referencia: con todos los ratings el top 10 % de
películas reúne el 78,8 %. Cuanto más estricto el umbral, más se concentran los positivos en las populares.

Implicaciones:
- **a) 3,5**: más datos, pero mete los «aprobados raspados» de los generosos (el 3,5 de Ana). Con estrellas
  enteras (34,3 % de los usuarios nunca usa medias) 3,5 equivale a 4,0.
- **b) 4,0**: lo más habitual en los papers con MovieLens → resultados comparables. Deja 26 usuarios sin ningún
  positivo y 374 sin positivos en test (salen de las métricas). Penaliza a los exigentes como Bruno.
- **c) relativo**: quita el sesgo, pero el 4,0 de un generoso deja de contar (Ana) y los que ponen siempre la
  misma nota (30 usuarios) quedan con todo positivo (o nada con «>»). Necesita media_u (C6).
- **d) híbrido min(4,0; media_u)** ⭐: nadie se queda sin positivos, menos usuarios sin positivos en test (151),
  y para el 72,6 % de los usuarios (media < 4) sigue siendo un umbral relativo… pero solo baja, nunca sube de 4,0.
  Coste: un parámetro más que explicar y media_u calculada solo con train (si no, fuga de información del test).
- **e) max(3,5; media_u)**: el contrario: exige más a los generosos. Más «puro», pero deja 350 usuarios sin test.
- **f) residuo**: quita también la «calidad» de la película (le gustó *más de lo esperado*): un 4,0 a *Cadena
  perpetua* no cuenta. Es lo que hace un recomendador de sorpresa, no de recuperación; además el 8,3 % de los
  ratings ≥ 4,0 pasarían a no positivos. No lo recomiendo como objetivo, sí como análisis.

E2: el umbral es un parámetro del pipeline (`--positivo {abs,rel,min,max}` y `--umbral 4.0`).
⭐ **d**, con **b** (≥ 4,0) como ablación para poder compararnos con la literatura.

## C3. Qué hacer con las notas bajas

| Definición de «nota baja» | % ratings | % usuarios con ≥ 1 | Mediana por usuario | % en top 10 % |
|---|---:|---:|---:|---:|
| r ≤ 2,0 | 12,7 | 85,6 | 5 | 68,8 |
| r ≤ 2,5 | 17,8 | 90,2 | 7 | 69,1 |
| r ≤ media_u − 1 | 13,4 | 95,9 | 8 | 71,8 |

Dato clave: si los negativos salen del lote (∝ frecuencia de positivos), la probabilidad de que el negativo
sacado sea una nota baja del propio usuario es del **1,1 %** de media. Es decir, los negativos del lote casi nunca
son lo que el usuario rechazó: si queremos usar esa información, hay que meterla a propósito.

| Opción | Modelo y pérdida | Evaluación | E2 |
|---|---|---|---|
| **a) Tirarlas** (ni positivos ni negativos; solo negativos del lote) ⭐ v1 | Lo de Yi et al.: la pérdida solo ve positivos. Sencillo. Se pierde la señal «vio esto y no le gustó». | Ninguna diferencia. | Una línea. |
| **b) Negativos duros explícitos** (cada positivo compite también con 1–5 notas bajas del usuario) | Más señal por ejemplo, pero las notas bajas se parecen a lo que le gusta (las eligió para ver): pueden empujar a separar películas casi iguales. Hay que mezclarlas con los del lote y logQ no se aplica a ellas (no salen del mismo muestreo). | Puede subir NDCG y bajar el recall. Hay que medirlo. | Muestreador propio. |
| **c) Anti-preferencias en la torre de usuario** (un segundo pooling con las películas que no le gustaron) | Es una decisión de **F**: el vector u recibe «lo que no» además de «lo que sí». No cambia la pérdida. | Experimento limpio (con y sin). | Más trabajo en F. |
| **d) a + c** | Pérdida sencilla y la información negativa entra por la entrada. | — | — |

Zona neutra: con C2 = d, lo que queda entre «nota baja» y el umbral (p. ej. 2,5 < r < 4,0) no es ni positivo
ni negativo; también hay que decir si entra en el historial de la torre de usuario (F).
⭐ **a** en la v1; **c** como experimento cuando se haga F. Depende de **F**.

## C4. Ponderar los positivos por la nota

Sobre positivos r ≥ 3,5. ESS = tamaño efectivo de muestra (100 % = todos pesan igual).

| Pesos | ESS | % del peso en notas 5,0 | % en top 10 % películas |
|---|---:|---:|---:|
| a) w = 1 | 100 % | 22,9 | 82,2 |
| b) lineal: w = 1 + (r − umbral) (con umbral 3,5: 1 … 2,5) | 91,2 % | 33,7 | 83,3 |
| c) exponencial: w = 2^(r − umbral) (1 … 2,83) | 87,3 % | 37,3 | 83,5 |
| d) relativo: w = 1 + max(0, r − media_u) | 90,2 % | 31,2 | 82,5 |

Implicaciones: el peso multiplica la pérdida de cada positivo (o la probabilidad de muestrearlo). Con b, un 5,0
de Ana pesa el doble que su 4,0. Cuesta poco (ESS 91 %) y es literalmente lo que pusimos en E1 («ponderada por
r_ui»). Si se ponderan muestreando (en vez de multiplicando la pérdida), cambia la distribución de los negativos
del lote y logQ tiene que contarlo (C13).
⭐ **b**, multiplicando la pérdida (no el muestreo). Depende de C2 (el «umbral» del peso es el de C2).

## C5. Cuánto pesa cada usuario

El 10 % de usuarios más activos aporta el **46,6 %** de los ratings y el 42,6 % de los positivos. Las métricas
(Recall@K…) promedian por usuario, así que entrenar por rating hace que el modelo se optimice para los que más
valoran.

| Opción | Pérdida | Evaluación | E2 |
|---|---|---|---|
| a) Nada (cada rating pesa igual) | Dominan los muy activos. | Peor en usuarios con poco historial (justo los de Letterboxd). | — |
| b) w = 1/√n_u | El top 10 % pasa a 22,0 % del peso; ESS 73,4 %. | Más equilibrio. | Una línea. |
| c) Ejemplos por usuario con tope (cada época: k = min(n_pos, tope) ejemplos por usuario, «historial → siguiente positivo») ⭐ | Equilibra por construcción, como YouTube (Covington 2016). Encaja si la torre de usuario recibe el historial. | Igual que b. | Es el generador de ejemplos de F. |

⭐ **c** si F1 es «historial» (lo más probable por Letterboxd); si F1 es «embedding de userId», **b**.
Depende de **F1** y afecta a **C13**.

## C6. Cómo se calcula la media de cada usuario (si C2 es c, d o e)

- **a) Media cruda**: simple, pero con pocos ratings es ruidosa (usuarios nuevos de Letterboxd con 5 notas).
- **b) Media encogida** hacia la global: media_u = (Σ r + λ·μ)/(n_u + λ), λ ≈ 10 ⭐. Para MovieLens apenas cambia
  (mediana 73 ratings por usuario); para usuarios nuevos evita umbrales absurdos.
- **c) Mediana**: robusta, pero con notas discretas salta a escalones (3,5 → 4,0).

En todos los casos: **solo con los ratings de train** (si no, el umbral ve el test → fuga). Depende de **D**.
⭐ **b**.

## C7. Qué cuenta como «relevante» al evaluar

- **a) La misma definición que en train**: coherente, pero cambiar C2 cambia también la vara de medir; dos
  modelos con umbrales distintos no se pueden comparar.
- **b) Siempre r ≥ 4,0**: comparable entre experimentos y con la literatura.
- **c) Las dos** ⭐: métrica principal con la de train y una columna fija con ≥ 4,0.

Depende de C2 y de **D**.

## C8. Usuarios sin positivos (en train o en test)

Con C2 = b: 26 sin ningún positivo y 374 sin positivos en test; con d: 0 y 151 (de 20 085).
- a) Quitarlos del todo. b) Mantenerlos en train (su historial ayuda a las películas) y excluirlos solo de las
  métricas ⭐. Depende de **D**.

## C9. Valoraciones en bloque el primer día

- El **62,5 %** de los ratings se hace en las primeras 24 h del usuario; el 59,0 % de los usuarios hace **todas**
  sus valoraciones el primer día; el 22,3 % de los ratings comparte segundo exacto con otro del mismo usuario.
- El primer día se puntúa más alto: media 3,60 frente a 3,44; ≥ 4,0 el 53,1 % frente al 45,0 %.

Es decir: la mayoría de los ratings son **recuerdos** que el usuario introduce de golpe (a menudo sus favoritas),
no lo que acaba de ver. Para C significa que «implícito» aquí es «declaró que le gustó», no «lo eligió ahora».
Opciones: a) ignorarlo en C ⭐; b) usar el primer día solo como historial, nunca como objetivo; c) bajarle el
peso. Lo importante es para **D**: un split temporal *dentro* de cada usuario no tiene orden real en el 59 % de
los usuarios (todo pasa el mismo día y con empates de segundo).

## C10. Usuarios sin medias estrellas

Antes de 2003 MovieLens no tenía medias estrellas: el 24,3 % de los ratings es anterior y el 34,3 % de los usuarios
nunca usa medias (como Carla). Para ellos 3,5 = 4,0 y el «3» mezcla «bien» con «regular».
- a) Nada ⭐ si C2 es relativo o híbrido (media_u ya absorbe la escala). b) Umbral por época, si C2 es absoluto.

## C11. Ratings antes del estreno

218 en el conjunto completo (42 en la submuestra; flag `antes_estreno`, nota media 3,46). Pueden ser preestrenos
o fechas mal puestas. a) Mantener; b) quitarlos ⭐ (0,0007 %, y no sabemos si la vio); c) solo como historial.

## C12. Re-ratings

MovieLens solo guarda la última nota de cada par (0 pares repetidos en B4); los 836 de duplicados fusionados ya se
resolvieron con la más reciente (B1b). Solo importa en Letterboxd (`diary.csv` tiene re-visionados):
a) la última ⭐ (coherente con B1b); b) la media.

## C13. logQ: con qué se calcula P(i)

logQ corrige que los negativos del lote salen con probabilidad ∝ P(i): s* = ⟨u, v_i⟩ − log P(i). Lo correcto es
usar la distribución **con la que se muestrean de verdad** los ejemplos, no la popularidad «en general».

| P(i) contada con | Spearman con «todos» | Películas con P = 0 | Ratio P_pos/P_todos (p5–p95, películas con ≥ 100 ratings) |
|---|---:|---:|---|
| positivos r ≥ 3,5 | 0,944 | 2 081 | 0,35 – 1,33 |
| positivos r ≥ 4,0 | 0,922 | 3 915 | 0,29 – 1,46 |
| positivos r > media_u | 0,939 | 2 273 | 0,33 – 1,40 |

Un ratio de 0,29 es una diferencia de log P de −1,2: no es un detalle. Ejemplos (positivos ≥ 4,0, entre las 300
más vistas): *Charlie's Angels* (2000) baja del puesto 242 al 913 (solo el 18 % le da ≥ 4,0); *12 Angry Men*
sube del 254 al 133 (83 %).

- **a) Todos los ratings**: corrige una popularidad que no es la del muestreo; penalizaría de más los clásicos
  bien valorados y de menos los taquillazos flojos.
- **b) Solo positivos del train, contados exactos** (no hace falta el *count-min sketch* de Yi et al.: cabe en
  memoria) + suavizado (+1) para las películas sin positivos.
- **c) b, pero contando como se generan los ejemplos** ⭐: si C5 = c (ejemplos por usuario con tope), P(i) es la
  frecuencia en ese flujo, no en los ratings. Si C4 se hace por muestreo, también hay que ponderar.

Ojo: la tasa de positivos depende de la popularidad (decil 1: 53,3 % ≥ 4,0; deciles 2–10: 30–40 %; correlación
popularidad–nota 0,26). Es decir, las populares **sí** gustan más de verdad; logQ quita el sesgo del muestreo, no
ese. Si queremos más cola larga, eso es cosa del MMR (o de τ), no de logQ.
⭐ **c**. Depende de C2, C4 y C5.

## C14. Falsos negativos del lote

Con negativos del lote, la película que hace de negativo es un positivo del propio usuario el **7,2 %** de las
veces de media (p90: 16,0 %), y algo que ya valoró el 11,1 %.
- a) Ignorarlo: el modelo aprende a alejar cosas que le gustan (sobre todo en usuarios muy activos).
- b) Enmascarar sus positivos en la softmax.
- c) Enmascarar todo lo que ha valorado ⭐: coste bajo (una máscara por lote) y deja la puerta abierta a C3-b.
  Si C3 = b, las notas bajas se enmascaran como negativos del lote pero entran como negativos explícitos.

## Qué depende de D (split) y de F (torre de usuario)

| Pregunta | Depende de D porque… | Depende de F porque… |
|---|---|---|
| C1 | el split decide qué positivos quedan para test | si u se calcula del historial, C decide qué entra en él |
| C2, C6 | media_u solo con train | — |
| C3 | — | opción c = anti-preferencias en la torre de usuario |
| C5 | — | «ejemplos por usuario» solo existe si F usa historial |
| C7, C8 | definen el test | — |
| C9 | split temporal sin orden real en el 59 % de usuarios | — |
| C13 | P(i) solo con train | depende del generador de ejemplos (C5) |

Orden propuesto: cerrar **C1 + D1 + F1** juntas; luego C2, C4, C6 (umbral y pesos); C3/C5 con F; C7–C9 con D;
C13/C14 al escribir la pérdida.
