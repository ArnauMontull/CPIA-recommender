# Resultados en `completo`

Checkpoint `mejor.pt` (época 9, τ = 0.05, semilla 42). Catálogo completo, sin lo valorado (C14). Test: historial = train + val (F8). EASE sobre las 10000 películas más populares, λ = 500.
Submuestra fija de 20 000 usuarios elegibles (la misma para los tres métodos). Generado en Colab (T4) con
`python -m src.evaluar --muestra completo --checkpoint .../mejor.pt --device cuda --max-items-ease 10000 --usuarios-max 20000`.

## Val: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0548 | 0.1193 | 0.1359 |
| Recall@20 | 0.0936 | 0.2010 | 0.2225 |
| Recall@50 | 0.1665 | 0.3539 | 0.3797 |
| Recall@200 | 0.3561 | 0.6331 | 0.6715 |
| NDCG@10 | 0.0518 | 0.1066 | 0.1228 |
| NDCG@20 | 0.0638 | 0.1317 | 0.1492 |
| NDCG@50 | 0.0862 | 0.1781 | 0.1971 |
| NDCG@200 | 0.1355 | 0.2540 | 0.2763 |
| Cobertura@20 | 0.0113 | 0.1597 | 0.1946 |
| Novedad@20 | 8.3210 | 10.0078 | 10.0811 |

## Val: relevante = rating ≥ 4 (C7)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0567 | 0.1216 | 0.1403 |
| Recall@20 | 0.0962 | 0.2038 | 0.2291 |
| Recall@50 | 0.1708 | 0.3584 | 0.3885 |
| Recall@200 | 0.3631 | 0.6382 | 0.6813 |
| NDCG@10 | 0.0505 | 0.1034 | 0.1209 |
| NDCG@20 | 0.0631 | 0.1294 | 0.1488 |
| NDCG@50 | 0.0861 | 0.1763 | 0.1974 |
| NDCG@200 | 0.1349 | 0.2505 | 0.2750 |
| Cobertura@20 | 0.0113 | 0.1597 | 0.1946 |
| Novedad@20 | 8.3210 | 10.0078 | 10.0811 |

## Test: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0491 | 0.1149 | 0.1291 |
| Recall@20 | 0.0839 | 0.1897 | 0.2074 |
| Recall@50 | 0.1495 | 0.3297 | 0.3521 |
| Recall@200 | 0.3332 | 0.6030 | 0.6346 |
| NDCG@10 | 0.0497 | 0.1072 | 0.1231 |
| NDCG@20 | 0.0598 | 0.1292 | 0.1451 |
| NDCG@50 | 0.0796 | 0.1709 | 0.1885 |
| NDCG@200 | 0.1273 | 0.2444 | 0.2643 |
| Cobertura@20 | 0.0144 | 0.1634 | 0.2041 |
| Novedad@20 | 8.3395 | 10.0466 | 10.1403 |

## Test: relevante = rating ≥ 4 (C7)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0510 | 0.1178 | 0.1338 |
| Recall@20 | 0.0868 | 0.1935 | 0.2140 |
| Recall@50 | 0.1541 | 0.3351 | 0.3604 |
| Recall@200 | 0.3401 | 0.6080 | 0.6433 |
| NDCG@10 | 0.0482 | 0.1037 | 0.1207 |
| NDCG@20 | 0.0589 | 0.1267 | 0.1443 |
| NDCG@50 | 0.0796 | 0.1692 | 0.1885 |
| NDCG@200 | 0.1268 | 0.2410 | 0.2629 |
| Cobertura@20 | 0.0144 | 0.1634 | 0.2041 |
| Novedad@20 | 8.3395 | 10.0466 | 10.1403 |
