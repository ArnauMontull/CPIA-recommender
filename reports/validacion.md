# Validación de integridad (B4)

Generado por `python -m src.preprocesamiento.validar --min-ratings 20`. Corrige lo que no implica decisiones de modelado y marca el resto con columnas booleanas en `data/processed/*.parquet`.

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
| movies×tmdb | sinopsis recuperadas de la API de TMDB | 3.412 | rellenadas |
| movies×tmdb | película de ML sin fila en TMDB Kaggle (2017) | 41.294 | flag en_tmdb = False |
| movies×tmdb | película de ML sin sinopsis | 42.267 | flag tiene_sinopsis = False (B2: se mantiene) |
| ratings | filas leídas | 32.000.204 | lectura en 2.5 s |
| ratings | valores nulos | 0 | se eliminan |
| ratings | nota fuera de {0.5. 1.0. …. 5.0} | 0 | se eliminan |
| ratings | movieId inexistente en movies.csv | 0 | se eliminan |
| ratings | timestamp fuera de 1995-01-01 … 2023-10-31 | 0 | se eliminan |
| ratings | par (usuario. película) repetido | 0 | se conserva el más reciente |
| ratings | valorada más de 1 año antes del estreno | 218 | flag antes_estreno (no se borra) |
| movies | duplicados por tmdbId fusionados | 36 | 36 movieId absorbidos en otro (ver data/processed/fusiones.parquet) |
| ratings | usuario que valoró las dos copias de un duplicado | 836 | se conserva la nota más reciente |
| movies | películas con < 20 ratings | 64.214 | se eliminan (274.214 ratings; 0.86%) |
| ratings | usuarios | 200.948 | — |
| ratings | usuarios que quedan con < 20 ratings tras el filtro | 185 | se mantienen |

## Resumen (después de fusionar y filtrar)
- Ratings: 31.725.154; usuarios: 200.948; películas: 23.335.
- Películas con sinopsis: 23.195 (99.4%); ratings cuya película tiene sinopsis: **99.9%**.
- Películas sin géneros ni sinopsis: 15.

## Duplicados fusionados (mismo tmdbId)
| movieId absorbido | → movieId | Título | Año | tmdbId | Ratings que aporta |
|---:|---:|---|---:|---:|---:|
| 144606 | 6003 | Confessions of a Dangerous Mind | 2002 | 4912 | 410 |
| 178755 | 34330 | November | 2003 | 9775 | 6 |
| 174533 | 4241 | Pokémon: Spell of the Unown | 2000 | 10991 | 92 |
| 174543 | 5672 | Pokémon 4Ever: Celebi - Voice of the Forest | 2001 | 12600 | 53 |
| 202599 | 33154 | Enron: The Smartest Guys in the Room | 2005 | 13020 | 131 |
| 144426 | 5264 | Clockstoppers | 2002 | 15028 | 64 |
| 181751 | 6776 | Lagaan: Once Upon a Time in India | 2001 | 19666 | 160 |
| 222065 | 145972 | Love Surreal | 2015 | 21753 | 2 |
| 232217 | 79329 | Passing Strange | 2009 | 24487 | 5 |
| 191775 | 66511 | Berlin Calling | 2008 | 30508 | 24 |
| 137813 | 65633 | Fuel | 2009 | 34696 | 1 |
| 137809 | 93928 | After | 2009 | 39172 | 2 |
| 219145 | 144858 | God's Angry Man | 1980 | 45555 | 5 |
| 213187 | 110515 | Superdad | 1973 | 46667 | 0 |
| 258991 | 200908 | The Adventures of Pinocchio | 1972 | 58451 | 1 |
| 123107 | 31026 | The Phantom of the Opera | 1990 | 69234 | 31 |
| 181393 | 895 | Venice | 2010 | 79782 | 3 |
| 103208 | 2679 | Place at the Table, A | 2012 | 84198 | 21 |
| 163246 | 70155 | Seven Years Bad Luck | 1921 | 97995 | 6 |
| 134006 | 864 | Wife | 1953 | 132641 | 3 |
| 272905 | 7396 | Scenes from a Marriage | 1974 | 133919 | 16 |
| 275753 | 167510 | The Boxing Cats | 1894 | 159898 | 3 |
| 266856 | 176543 | That's Sexploitation! | 2013 | 264068 | 1 |
| 175813 | 114342 | Force Majeure | 2014 | 265189 | 433 |
| 162632 | 142861 | Cemetery of Splendor | 2015 | 298721 | 35 |
| 215011 | 148976 | Marvel Super Hero Adventures: Frost Fight! | 2015 | 372631 | 5 |
| 215119 | 174527 | Chasing Trane | 2017 | 411012 | 4 |
| 218855 | 168000 | Summer Love | 2016 | 411081 | 2 |
| 195339 | 208020 | Ratsasan | 2018 | 412862 | 2 |
| 187127 | 163585 | David Lynch: The Art Life | 2017 | 413765 | 47 |
| 258279 | 203218 | Good Boys | 2019 | 521777 | 74 |
| 215891 | 203453 | Hope You Die Next Time :-) | 2018 | 542757 | 5 |
| 229671 | 204294 | The Great Hack | 2019 | 566222 | 25 |
| 206117 | 203449 | The Lonely Island Presents: The Unauthorized Bash Brothers Experience | 2019 | 604196 | 24 |
| 258689 | 254100 | Roadrunner: A Film About Anthony Bourdain | 2021 | 642732 | 14 |
| 239670 | 242100 | Dani Rovira: Odio | 2021 | 792517 | 3 |
