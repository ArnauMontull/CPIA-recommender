# Análisis de perfiles de Letterboxd sobre el modelo final

Scripts de la sesión del 20/10/2026: comparar los vectores `u` de varias personas entre sí y con los usuarios de MovieLens, agrupar películas y usuarios, y generar el mapa 3D y el dashboard. Usan el checkpoint `modelos/completo_tau0.05_s42/mejor.pt` y los perfiles procesados con `python -m src.letterboxd`.

## Privacidad: qué NO está en el repo

Los exports de Letterboxd y los perfiles procesados de otras personas viven en `data/external/letterboxd/` (ignorado por git; el repo es público). Los resultados que se generan con estos scripts (`*.json`, `*.parquet`, `*.html`) contienen datos de esas personas y **no se suben**: escribe siempre en `data/analisis/` y no en esta carpeta. Un `.gitignore` local lo refuerza. Solo se versionan los scripts, la plantilla del dashboard y los nombres de los grupos.

## Cómo se ejecuta

Desde la raíz del repo, con el venv del proyecto y los alias de los perfiles ya procesados en `data/external/letterboxd/<alias>/`:

```
mkdir -p data/analisis && cp scripts/analisis_perfiles/nombres*.json data/analisis/
export PERSONAS=alias1_api,alias2_api,...            # los alias, separados por comas
S=data/analisis
PYTHONPATH=. python scripts/analisis_perfiles/barrido_k.py $S/mapa12            # barrido de K (películas 2-30, usuarios 2-20)
PYTHONPATH=. python scripts/analisis_perfiles/mapa_calc.py 15 $S/k15            # PCA + k-means de películas (K=15) y un .md por grupo
PYTHONPATH=. python scripts/analisis_perfiles/usuarios_grupos.py $S/k5 5         # k-means de 10.000 usuarios (K=5) y un .md por grupo
KF=15 KU=5 PYTHONPATH=. python scripts/analisis_perfiles/combo_calc.py $S/k15    # dos PCA (películas / usuarios) y todo proyectado
python scripts/analisis_perfiles/combo_html.py $S/k15 $S/nombres15.json $S/nombres_usuarios5.json $S/mapa_combo_3d.html
# dashboard (necesita lo anterior más las recomendaciones por persona)
PYTHONPATH=. python scripts/analisis_perfiles/analisis_personas.py $S
PYTHONPATH=. python scripts/analisis_perfiles/dashboard_data.py $S
cp scripts/analisis_perfiles/{dash_template.html,build_dash.py} $S/ && (cd $S && python build_dash.py)   # -> $S/dashboard.html
```

| Script | Qué hace |
|---|---|
| `ang3.py` | matriz de ángulos entre personas |
| `subespacio.py` | distancia de cada persona al subespacio (y al cono) de las demás |
| `analisis_personas.py` | por persona: vecinos del dataset, afinidad con grupos y top 20 con MMR |
| `grupo7.py` | película para ver todos juntos (reglas `minimo` y `media` sobre N personas) |
| `barrido_k.py` | silhouette, Davies-Bouldin y Calinski-Harabasz para elegir K |
| `mapa_calc.py`, `usuarios_grupos.py` | grupos de películas y de usuarios, con un `.md` por grupo para nombrarlos |
| `combo_calc.py`, `combo_html.py` | vista 3D con los dos espacios, elipsoides de grupo y casillas de mostrar/ocultar |
| `dashboard_data.py`, `build_dash.py`, `dash_template.html` | dashboard completo en un único HTML |

Los nombres de los grupos (`nombres*.json`) los propusieron agentes Haiku, uno por grupo, a partir del `.md` de cada grupo; son una etiqueta descriptiva, no una verdad de los datos. Los silhouettes del barrido son bajos (< 0,12): no hay grupos nítidos y K es una decisión descriptiva.

## Pendiente relacionado

`docs/planning_streaming_y_grupos.md` (tarea T-085): filtro por plataformas de streaming y recomendación para grupos en la presentación del modelo final.
