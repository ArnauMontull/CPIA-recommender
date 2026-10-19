# Resultados en `dev`

Checkpoint `mejor.pt` (época 5, τ = 0.05, semilla 42). Catálogo completo, sin lo valorado (C14). Test: historial = train + val (F8). EASE sobre las 8000 películas más populares, λ = 500.

## Val: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0554 | 0.1189 | 0.1130 |
| Recall@20 | 0.0943 | 0.1978 | 0.1835 |
| Recall@50 | 0.1689 | 0.3445 | 0.3206 |
| Recall@200 | 0.3616 | 0.6181 | 0.5919 |
| NDCG@10 | 0.0527 | 0.1057 | 0.1025 |
| NDCG@20 | 0.0645 | 0.1298 | 0.1242 |
| NDCG@50 | 0.0874 | 0.1748 | 0.1663 |
| NDCG@200 | 0.1375 | 0.2495 | 0.2392 |
| Cobertura@20 | 0.0105 | 0.1088 | 0.1075 |
| Novedad@20 | 8.3304 | 9.7923 | 9.7667 |

## Val: relevante = rating ≥ 4 (C7)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0572 | 0.1212 | 0.1166 |
| Recall@20 | 0.0972 | 0.2007 | 0.1891 |
| Recall@50 | 0.1735 | 0.3495 | 0.3287 |
| Recall@200 | 0.3693 | 0.6233 | 0.6007 |
| NDCG@10 | 0.0512 | 0.1025 | 0.1004 |
| NDCG@20 | 0.0638 | 0.1276 | 0.1234 |
| NDCG@50 | 0.0872 | 0.1732 | 0.1663 |
| NDCG@200 | 0.1369 | 0.2460 | 0.2376 |
| Cobertura@20 | 0.0105 | 0.1088 | 0.1075 |
| Novedad@20 | 8.3304 | 9.7923 | 9.7667 |

## Test: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0475 | 0.1106 | 0.1021 |
| Recall@20 | 0.0827 | 0.1815 | 0.1664 |
| Recall@50 | 0.1510 | 0.3197 | 0.2942 |
| Recall@200 | 0.3374 | 0.5873 | 0.5591 |
| NDCG@10 | 0.0494 | 0.1050 | 0.0981 |
| NDCG@20 | 0.0595 | 0.1255 | 0.1165 |
| NDCG@50 | 0.0803 | 0.1670 | 0.1548 |
| NDCG@200 | 0.1285 | 0.2390 | 0.2252 |
| Cobertura@20 | 0.0117 | 0.1097 | 0.1100 |
| Novedad@20 | 8.3501 | 9.8155 | 9.8188 |

## Test: relevante = rating ≥ 4 (C7)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0489 | 0.1131 | 0.1053 |
| Recall@20 | 0.0852 | 0.1850 | 0.1708 |
| Recall@50 | 0.1543 | 0.3235 | 0.3000 |
| Recall@200 | 0.3437 | 0.5916 | 0.5668 |
| NDCG@10 | 0.0477 | 0.1015 | 0.0956 |
| NDCG@20 | 0.0586 | 0.1232 | 0.1154 |
| NDCG@50 | 0.0799 | 0.1651 | 0.1545 |
| NDCG@200 | 0.1278 | 0.2355 | 0.2236 |
| Cobertura@20 | 0.0117 | 0.1097 | 0.1100 |
| Novedad@20 | 8.3501 | 9.8155 | 9.8188 |
