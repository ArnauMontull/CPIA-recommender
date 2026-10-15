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
| B2 | Las películas sin sinopsis se mantienen (sinopsis vacía + flag) y los huecos se rellenan con la API de TMDB | Kaggle es de 2017. Tras el filtro faltaban 3 508 sinopsis; la API recupera 3 412 (117 404). Ahora el 99,4 % de las películas y el 99,9 % de los ratings tienen sinopsis. Quedan 15 películas sin géneros ni sinopsis. |
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
