# Submuestra de desarrollo (A2b)

Generada con `python -m src.preprocesamiento.submuestra --frac 0.1 --semilla 42`. Se muestrean usuarios enteros, estratificados por género dominante × quintil de actividad × época del primer rating.

## Tamaño
- Usuarios: 20.096 de 200.948 (10.0%)
- Ratings: 3.210.086 de 32.000.204 (10.0%)
- Películas con algún rating: 40.512 de 84.432 (las que faltan son de la cola larga: solo las vio algún usuario no muestreado)
- Estratos: 227 (de ellos 68 con menos de 1/frac usuarios, donde el redondeo puede dar 0 o 1)
- Concentración en el top 10 % de películas: 95.70% completo, 88.02% submuestra
- Mayor diferencia en el reparto de géneros: **0.27 pp**
- Ratings por usuario (mediana): 73 completo, 72 submuestra

## Comparación
| Medida | Completo | Submuestra | Diferencia (pp) |
|---|---:|---:|---:|
| género: Drama | 20.76 % | 20.49 % | -0.27 |
| género: Comedy | 16.34 % | 16.42 % | +0.07 |
| género: Action | 9.32 % | 9.51 % | +0.19 |
| género: Thriller | 9.04 % | 9.17 % | +0.14 |
| género: Adventure | 7.12 % | 7.01 % | -0.11 |
| género: Romance | 6.52 % | 6.40 % | -0.12 |
| género: Sci-Fi | 5.82 % | 5.71 % | -0.11 |
| género: Crime | 5.80 % | 5.75 % | -0.04 |
| género: Fantasy | 3.27 % | 3.34 % | +0.06 |
| género: Horror | 3.25 % | 3.33 % | +0.08 |
| género: Mystery | 2.47 % | 2.52 % | +0.04 |
| género: Children | 2.34 % | 2.36 % | +0.02 |
| género: War | 1.77 % | 1.77 % | +0.00 |
| género: Animation | 1.77 % | 1.79 % | +0.02 |
| género: IMAX | 1.15 % | 1.15 % | -0.00 |
| género: Documentary | 1.14 % | 1.15 % | +0.01 |
| género: Musical | 1.06 % | 1.06 % | +0.00 |
| género: Western | 0.78 % | 0.78 % | -0.00 |
| género: Film-Noir | 0.29 % | 0.29 % | +0.00 |
| nota 0.5 | 1.64 % | 1.62 % | -0.02 |
| nota 1.0 | 2.96 % | 2.90 % | -0.06 |
| nota 1.5 | 1.66 % | 1.57 % | -0.09 |
| nota 2.0 | 6.34 % | 6.31 % | -0.03 |
| nota 2.5 | 5.27 % | 5.07 % | -0.19 |
| nota 3.0 | 18.92 % | 19.19 % | +0.27 |
| nota 3.5 | 13.41 % | 13.46 % | +0.05 |
| nota 4.0 | 26.15 % | 26.24 % | +0.09 |
| nota 4.5 | 9.29 % | 9.32 % | +0.02 |
| nota 5.0 | 14.36 % | 14.33 % | -0.04 |
| usuarios que empiezan en 1995-2004 | 39.33 % | 39.32 % | -0.01 |
| usuarios que empiezan en 2005-2014 | 27.28 % | 27.29 % | +0.01 |
| usuarios que empiezan en 2015-2023 | 33.39 % | 33.39 % | +0.01 |
