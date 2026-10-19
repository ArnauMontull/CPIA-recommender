# Entrenamiento en `completo` (Colab, T4)

Ejecutado el 19/10/2026 con `python -m src.entrenar --muestra completo --device cuda --tau 0.05`: 10 épocas (lote 4 096, ~3 700 pasos por
época, calendario de lr con calentamiento de 200 pasos y decaimiento coseno), semilla 42. Tiempo total: 54,8 min.
Checkpoints y log en Drive: `MyDrive/CPIA_PROJECT/modelos/completo_tau0.05_s42/` (`mejor.pt` = época 9).

Validación (submuestra fija de 20 000 usuarios elegibles, catálogo completo, sin lo valorado):

| Época | Pérdida | Recall@200 | Minutos |
|---:|---:|---:|---:|
| 1 | 6,6937 | 0,6461 | 5,5 |
| 2 | 6,3067 | 0,6565 | 11,0 |
| 3 | 6,2238 | 0,6621 | 16,5 |
| 4 | 6,1746 | 0,6661 | 21,9 |
| 5 | 6,1398 | 0,6662 | 27,4 |
| 6 | 6,1130 | 0,6673 | 32,9 |
| 7 | 6,0920 | 0,6700 | 38,4 |
| 8 | 6,0756 | 0,6702 | 43,8 |
| 9 | 6,0637 | **0,6715** | 49,3 |
| 10 | 6,0567 | 0,6708 | 54,8 |

La mejora es casi plana desde la época 3 (+0,01 en 6 épocas); la época 10 no mejora a la 9. Pendiente: evaluar `mejor.pt` en test y compararlo con
EASE sobre el mismo `completo` (los 0,618 de EASE en `reports/resultados_dev.md` son de `dev`, no comparables).
