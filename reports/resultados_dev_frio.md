# Resultados en `dev_frio`

Checkpoint `mejor.pt` (época 5, τ = 0.05, semilla 42). Catálogo completo, sin lo valorado (C14). Test: historial = train + val (F8). EASE sobre las 8000 películas más populares, λ = 500.

## Val: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0588 | 0.1524 | 0.1290 |
| Recall@20 | 0.0991 | 0.2397 | 0.2034 |
| Recall@50 | 0.1731 | 0.3982 | 0.3420 |
| Recall@200 | 0.3673 | 0.6695 | 0.6144 |
| NDCG@10 | 0.0569 | 0.1398 | 0.1217 |
| NDCG@20 | 0.0688 | 0.1656 | 0.1433 |
| NDCG@50 | 0.0912 | 0.2140 | 0.1855 |
| NDCG@200 | 0.1415 | 0.2888 | 0.2585 |
| Cobertura@20 | 0.0112 | 0.1062 | 0.0916 |
| Novedad@20 | 8.4328 | 9.8387 | 9.6679 |

## Val: relevante = rating ≥ 4 (C7)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0604 | 0.1558 | 0.1334 |
| Recall@20 | 0.1017 | 0.2435 | 0.2102 |
| Recall@50 | 0.1776 | 0.4037 | 0.3509 |
| Recall@200 | 0.3747 | 0.6749 | 0.6239 |
| NDCG@10 | 0.0551 | 0.1358 | 0.1189 |
| NDCG@20 | 0.0678 | 0.1629 | 0.1423 |
| NDCG@50 | 0.0909 | 0.2121 | 0.1855 |
| NDCG@200 | 0.1408 | 0.2850 | 0.2571 |
| Cobertura@20 | 0.0112 | 0.1062 | 0.0916 |
| Novedad@20 | 8.4328 | 9.8387 | 9.6679 |

## Test: relevante = positivo (C2)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0446 | 0.0988 | 0.0963 |
| Recall@20 | 0.0807 | 0.1733 | 0.1543 |
| Recall@50 | 0.1517 | 0.3074 | 0.2811 |
| Recall@200 | 0.3370 | 0.5682 | 0.5490 |
| NDCG@10 | 0.0749 | 0.1407 | 0.1384 |
| NDCG@20 | 0.0824 | 0.1568 | 0.1487 |
| NDCG@50 | 0.1034 | 0.1977 | 0.1873 |
| NDCG@200 | 0.1621 | 0.2843 | 0.2741 |
| Cobertura@20 | 0.0066 | 0.0690 | 0.0604 |
| Novedad@20 | 8.4163 | 9.8590 | 9.6629 |

## Test: relevante = rating ≥ 4 (C7)

| Métrica | Popularidad | EASE | Two-Tower |
|---|---:|---:|---:|
| Recall@10 | 0.0471 | 0.1019 | 0.0999 |
| Recall@20 | 0.0858 | 0.1781 | 0.1603 |
| Recall@50 | 0.1590 | 0.3159 | 0.2892 |
| Recall@200 | 0.3469 | 0.5748 | 0.5575 |
| NDCG@10 | 0.0713 | 0.1327 | 0.1312 |
| NDCG@20 | 0.0802 | 0.1508 | 0.1437 |
| NDCG@50 | 0.1033 | 0.1947 | 0.1846 |
| NDCG@200 | 0.1624 | 0.2797 | 0.2709 |
| Cobertura@20 | 0.0066 | 0.0690 | 0.0604 |
| Novedad@20 | 8.4163 | 9.8590 | 9.6629 |
