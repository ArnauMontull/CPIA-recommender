# Planning: plataformas de streaming y recomendación para grupos (presentación del modelo final)

Pedido de Arnau el 20/10/2026. Tarea T-085 en «LIFE OS Tareas» (área CPIA). Estado: **pendiente**, sin empezar.

## Qué hay que añadir en la presentación del modelo final

1. **Toggle de plataformas de streaming.** Casillas para elegir plataformas (Netflix, Prime Video, Max, Disney+, Movistar+, Filmin, Apple TV+…) y que las recomendaciones solo incluyan películas de su catálogo.
2. **Recomendar para varias personas.** Poder meter los datos de varias personas (exports de Letterboxd o perfiles ya procesados) y obtener películas para ver en **familia, con amigos o en pareja**.

## Qué ya existe y se reutiliza

- `src/demo_pareja.py`: puntuación de dos personas sobre el mismo catálogo con las reglas `interseccion`, `minimo` (nadie sufre), `media`, excluyendo la unión de lo visto. Hay que generalizarlo de 2 a N personas (el código ya trabaja con una matriz `[P, N]` de scores).
- `src/demo_letterboxd.py`: `recomendar_usuario` y `mmr` (MMR sobre los 200 mejores, como en el motor en C). El grupo puede puntuar con `minimo` o `media` y luego pasar por MMR.
- `movies.parquet` ya trae `tmdbId`, el puente natural hacia los catálogos de streaming.

## Decisiones pendientes

| # | Pregunta | Opciones | Estado |
|---|---|---|---|
| S1 | ¿De dónde sale el catálogo por plataforma? | API de TMDB `watch/providers` (datos de JustWatch, exige atribución a JustWatch); o un export estático | pendiente |
| S2 | ¿País/región del catálogo? | España por defecto; selector si la presentación lo admite | pendiente |
| S3 | ¿Estático o en vivo? | Descargar los catálogos una vez (con caché, como `cache_tmdb.jsonl`) y guardarlos con fecha, para que la demo funcione sin red | pendiente |
| S4 | ¿Cómo se combinan los scores de un grupo? | `minimo` (más justo), `media`, o una mezcla; con **N** personas el mínimo se vuelve muy estricto | pendiente |
| S5 | ¿Pesos por persona? | Pesos iguales por defecto; opcional «esta vez elige más X» | pendiente |
| S6 | ¿Cómo se marca lo ya visto? | Unión de lo visto por todos; las películas no emparejadas con el catálogo no se pueden excluir (ver límite abajo) | pendiente |
| S7 | ¿Modo familia con menores? | Filtro de clasificación por edad opcional | pendiente |

## Límites a recordar

- Los catálogos de streaming cambian cada semana: hay que guardar la **fecha de la consulta** y avisarlo en la presentación.
- Solo se excluyen las películas vistas que están emparejadas con el catálogo. Con perfiles con muchas películas sin emparejar puede salir recomendada alguna que ya se vio.
- Los datos de otras personas solo se usan con su permiso expreso, y en el repo no se guardan exports de terceros (`data/` está fuera de git).

## Orden de trabajo propuesto

1. Generalizar `demo_pareja.py` a N personas (`--alias a b c ...`) con la regla elegida y MMR.
2. Resolver S1–S3 y guardar los catálogos por plataforma en `data/external/streaming/` (fuera de git).
3. Añadir el filtro por plataforma antes del MMR (así el MMR diversifica dentro de lo disponible).
4. Exponerlo en la presentación: casillas de plataformas y selector de personas.
5. Tests: filtro por plataforma, exclusión de la unión de lo visto y caso de una sola persona igual al de `demo_letterboxd`.
