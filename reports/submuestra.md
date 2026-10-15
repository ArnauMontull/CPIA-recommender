# Submuestra de desarrollo (A2b)

Generada con `python -m src.preprocesamiento.submuestra --frac 0.1 --semilla 42`. Se muestrean usuarios enteros, estratificados por género dominante × quintil de actividad × época del primer rating.

## Tamaño
- Usuarios: 20.085 de 200.948 (10.0%)
- Ratings: 3.154.155 de 31.725.154 (9.9%)
- Películas con algún rating: 22.787 de 23.335 (las que faltan son de la cola larga: solo las vio algún usuario no muestreado)
- Estratos: 227 (de ellos 69 con menos de 1/frac usuarios, donde el redondeo puede dar 0 o 1)
- Concentración en el top 10 % de películas: 79.02% completo, 78.77% submuestra
- Mayor diferencia en el reparto de géneros: **0.34 pp**
- Ratings por usuario (mediana): 72 completo, 73 submuestra

## Comparación
| Medida | Completo | Submuestra | Diferencia (pp) |
|---|---:|---:|---:|
| género: Drama | 20.70 % | 20.36 % | -0.34 |
| género: Comedy | 16.33 % | 16.54 % | +0.21 |
| género: Action | 9.36 % | 9.57 % | +0.21 |
| género: Thriller | 9.06 % | 9.16 % | +0.10 |
| género: Adventure | 7.16 % | 7.08 % | -0.08 |
| género: Romance | 6.53 % | 6.45 % | -0.07 |
| género: Sci-Fi | 5.85 % | 5.74 % | -0.11 |
| género: Crime | 5.82 % | 5.75 % | -0.06 |
| género: Fantasy | 3.28 % | 3.36 % | +0.08 |
| género: Horror | 3.22 % | 3.22 % | -0.00 |
| género: Mystery | 2.48 % | 2.51 % | +0.03 |
| género: Children | 2.34 % | 2.38 % | +0.04 |
| género: War | 1.77 % | 1.76 % | -0.01 |
| género: Animation | 1.75 % | 1.78 % | +0.03 |
| género: IMAX | 1.16 % | 1.17 % | +0.01 |
| género: Musical | 1.07 % | 1.07 % | -0.00 |
| género: Documentary | 1.06 % | 1.04 % | -0.02 |
| género: Western | 0.78 % | 0.77 % | -0.01 |
| género: Film-Noir | 0.29 % | 0.28 % | -0.01 |
| nota 0.5 | 1.60 % | 1.72 % | +0.12 |
| nota 1.0 | 2.94 % | 2.99 % | +0.04 |
| nota 1.5 | 1.64 % | 1.60 % | -0.03 |
| nota 2.0 | 6.32 % | 6.34 % | +0.03 |
| nota 2.5 | 5.21 % | 5.13 % | -0.08 |
| nota 3.0 | 18.91 % | 18.86 % | -0.04 |
| nota 3.5 | 13.37 % | 13.31 % | -0.05 |
| nota 4.0 | 26.25 % | 26.32 % | +0.07 |
| nota 4.5 | 9.33 % | 9.23 % | -0.10 |
| nota 5.0 | 14.43 % | 14.48 % | +0.05 |
| usuarios que empiezan en 1995-2004 | 39.33 % | 39.32 % | -0.01 |
| usuarios que empiezan en 2005-2014 | 27.28 % | 27.29 % | +0.01 |
| usuarios que empiezan en 2015-2023 | 33.39 % | 33.39 % | +0.00 |
