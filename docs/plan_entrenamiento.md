# Plan: entrenamiento, evaluación y E2 (siguientes pasos)

Estado: **por empezar** (escrito el 16/10/2026 al cerrar la sesión). Todo lo anterior está hecho y en `main`: preprocesamiento
de C a G (`docs/plan_implementacion_CDE.md`, `docs/plan_antes_entrenamiento.md`), métricas, Colab (`docs/colab.md`) y la guía
de la inferencia en C para Uriel (`docs/guia_H10_inferencia_C/`). Decisiones: `docs/decisiones_preprocesamiento.md`.

**Fechas** (Atenea): Expo **27/10 15:00** · E2 Diseño **29/10 23:59** · E3 Implementación 08/12 · E4 Informe y vídeo 22/12.

## Preguntas para empezar la próxima sesión

| # | Pregunta | Opciones | ⭐ |
|---|---|---|---|
| T1 | Temperatura τ de la softmax | a) fija 0,05 · b) aprendida (como CLIP) · c) barrido en val {0,02; 0,05; 0,1} | c |
| T2 | Tamaño de lote | 1 024 (CPU, dev) · 4 096 (GPU, completo) | según dispositivo |
| T3 | Dropout del embedding de `movieId` (E6) | 0,3 · 0,5 · 0,7 | 0,5 |
| T4 | Capa oculta de las torres (contrato provisional de H10) | h = 256 · 512; sin ReLU tras la última capa (va L2) | 256, sin ReLU final |
| T5 | Baselines para comparar | a) popularidad · b) a + EASE (Steck 2019, está en `docs/papers`) · c) b + LightGCN | b |
| T6 | Métrica de early stopping | Recall@50 · NDCG@20 · Recall@200 en val | Recall@200 (lo que recupera HNSW/top-200) |
| T7 | Dataset scrapeado de Letterboxd (G2) y export de Uriel | ¿dónde están? | — |

## Tareas (agentes en paralelo, como en esta sesión)

| # | Quién | Qué | Archivos |
|---|---|---|---|
| 1 | Sonnet | **Modelo** (`src/modelo.py`). Torre de ítem: emb. de texto (384, congelado) → densa; géneros (16, multi-hot); directores y actores (`EmbeddingBag` con los vocabularios); `anio_z`, `duracion_z` y banderas; emb. de `movieId` con dropout (E6) → MLP → d = 128 → L2 (E7). Torre de usuario: V = torre de ítem sobre **todo** el catálogo una vez por paso; medias ponderadas de las bolsas A y B (`index_add_` o `EmbeddingBag` con `per_sample_weights`, restando el objetivo en train como `historial.media_bolsa`); [media A, media B, vacía A, vacía B, log1p(n)] → MLP de 2 capas → d → L2 (F7). Tests: la media en torch = la de numpy; forma y norma de u y v. | `src/modelo.py`, `tests/test_modelo.py` |
| 2 | Sonnet | **Pérdida** (`src/perdida.py`): softmax con negativos del lote, logits = u·v_j/τ − log P(j) (logQ, C13); máscara de lo valorado en train por el usuario salvo el propio objetivo (C14); peso por ejemplo (`peso` de `ejemplos_train`, C4 × C9). Tests a mano con un lote de 3. | `src/perdida.py`, `tests/test_perdida.py` |
| 3 | Sonnet | **Entrenamiento** (`src/entrenar.py`): carga de `data/processed/<muestra>/`, barajado por época con semilla (D10), AdamW, early stopping en val con `metricas.evaluar_por_lotes` (catálogo completo, máscara C14), checkpoints cada N pasos en `--salida` y **reanudar desde el último** (Colab se corta), `--device cpu|cuda`, `marcar()` para la barra de progreso. CLI: `python -m src.entrenar --muestra dev --device cpu --salida modelos/`. | `src/entrenar.py`, `tests/test_entrenar.py` (una época en datos sintéticos) |
| 4 | Sonnet | **Evaluación** (`src/evaluar.py`): test con historial train + val (F8) y máscara train + val; las dos definiciones de relevante (C7); `dev_frio` (G6); demo con Letterboxd (top-20 para `data/external/letterboxd/<alias>/`). Escribe `reports/resultados_<muestra>.md`. | `src/evaluar.py`, `tests/test_evaluar.py` |
| 5 | Sonnet | **Baselines** (T5): popularidad y EASE con las mismas métricas. | `src/baselines.py`, `tests/test_baselines.py` |
| 6 | Sonnet | **Exportar a C** (`src/exportar_c.py`): V, pesos del MLP de usuario y un usuario de ejemplo con el contrato de la guía H10 (§2), cerrando lo provisional (h, banderas, orden de la entrada). | `src/exportar_c.py` |
| 7 | Haiku | Ejecutar en dev (CPU) y medir el tiempo de una época; empaquetar `completo` para Colab (`scripts/empaquetar_datos.py`). | — |
| 8 | Arnau | Entrenar el completo en Colab siguiendo `docs/colab.md`. | — |
| 9 | Uriel | Motor de inferencia en C con OpenMP (`docs/guia_H10_inferencia_C/`) y HNSW con `hnswlib` (H9). | `c/` |
| 10 | Opus | Revisión, invariantes, resultados y E2. | — |

Orden: 1, 2 y 5 en paralelo → 3 → 7 (dev) → 4 → 8 (completo en Colab) → 6 → 9. Uriel puede empezar ya el paso 9 con el
modelo falso del §5 de su guía.

## E2 (Diseño, 29/10): qué hay que preparar

- **Pipeline de preprocesamiento**: diagrama de flujo de las etapas (split → señal → historiales → features), con las cifras
  de `reports/pipeline_dev.md` y `reports/pipeline_completo/`.
- **Decisiones y cambios respecto a E1**, justificados. Los dos grandes: (1) las notas bajas entran como anti-preferencias
  (C3 c) además de la binarización ponderada, y (2) el arranque en frío se mide con un experimento aparte (G6), no con el split.
- **Cifras que hay que corregir de E1**: el catálogo es de 23 335 películas (no 84 000; filtro B1 de ≥ 20 ratings) y los
  vectores ocupan 23 335 × 128 × 4 B ≈ 12 MB (no 43 MB).
- **Arquitectura**: las dos torres con sus entradas (E y F), la pérdida (C13, C14) y la evaluación (D6).
- **Benchmarks preliminares en CPU**: búsqueda exacta en numpy frente a C (cuando Uriel la tenga) y, si da tiempo, pandas
  frente a Polars al leer los 32 M de ratings (H1).
