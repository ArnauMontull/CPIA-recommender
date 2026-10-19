# Comprobaciones `s44` (muestra `dev`)

| Comprobación | Resultado | Detalle |
|---|:---:|---|
| finitud_y_normas | OK | V (23335, 128), U (3000, 128); |‖·‖-1| máx: V 1.8e-07, U 1.8e-07 |
| determinismo | OK | dos evaluaciones idénticas (recall@200 = 0.6774) |
| exclusion_c14 | OK | 0 recomendaciones ya valoradas de 600000 (top-200, val y test) |
| metricas_azar_y_oraculo | OK | k=200: azar 0.0076 (≈ 0.0086); oráculo 0.9999 = techo 0.9999 |
| numpy_igual_torch | OK | 150 usuarios, diferencia máxima 2.8e-06 |
| id_de_peliculas_frias | OK | 6 películas frías, cambio máximo al perturbar el ID: 0.0e+00 |
| invariancia_orden_historial | OK | bolsa de 32 películas permutada: diferencia 5.2e-08 |

Latencia (CPU): {'ms_por_usuario_solo': 2.12, 'ms_por_usuario_en_lote_512': 0.217, 'ms_torre_de_items_completa': 60.8}
