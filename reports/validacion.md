# Validación de integridad (B4)

Generado por `python -m src.preprocesamiento.validar`. Corrige lo que no implica decisiones de modelado y marca el resto con columnas booleanas en `data/processed/*.parquet`.

| Tabla | Comprobación | Filas afectadas | Acción |
|---|---|---:|---|
| movies | filas leídas | 87.585 | — |
| movies | movieId duplicado | 0 | se conserva la primera |
| movies | sin año en el título | 616 | flag sin_anio (anio = NA) |
| movies | «(no genres listed)» | 7.080 | flag sin_generos. lista vacía (imputación: decisión B3) |
| movies | mismo título y año con distinto movieId | 401 | flag titulo_duplicado |
| links | película sin tmdbId | 124 | flag (no podrá usar metadatos TMDB) |
| links | tmdbId compartido por varios movieId | 72 | flag tmdb_compartido |
| tmdb | filas leídas | 45.466 | — |
| tmdb | id no numérico (filas desplazadas del CSV) | 3 | se eliminan |
| tmdb | id duplicado | 30 | se conserva la primera |
| tmdb | sinopsis vacía o de relleno («No overview found.»…) | 1.103 | se deja vacía |
| movies×tmdb | película de ML sin fila en TMDB Kaggle (2017) | 44.634 | flag en_tmdb = False |
| movies×tmdb | película de ML sin sinopsis | 45.679 | flag tiene_sinopsis = False (B2: se mantiene) |
| ratings | filas leídas | 32.000.204 | lectura en 5.2 s |
| ratings | valores nulos | 0 | se eliminan |
| ratings | nota fuera de {0.5. 1.0. …. 5.0} | 0 | se eliminan |
| ratings | movieId inexistente en movies.csv | 0 | se eliminan |
| ratings | timestamp fuera de 1995-01-01 … 2023-10-31 | 0 | se eliminan |
| ratings | par (usuario. película) repetido | 0 | se conserva el más reciente |
| ratings | valorada más de 1 año antes del estreno | 218 | flag antes_estreno (no se borra) |
| ratings | usuarios | 200.948 | — |
| ratings | usuarios con < 5 ratings | 0 | se mantienen (B1: sin filtro) |

## Resumen
- Ratings finales: 32.000.204; usuarios: 200.948; películas con al menos un rating: 84.432 de 87.585.
- Ratings cuya película tiene sinopsis TMDB: **96.6%** (aunque solo el 47.8% del catálogo la tiene: las películas sin sinopsis son sobre todo de la cola larga).
