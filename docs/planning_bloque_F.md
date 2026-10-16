# Planning del bloque F: la torre de usuario

Estado: **decidido el 16/10/2026 (todo)** → `decisiones_preprocesamiento.md`. F1–F3 son las preguntas originales del 15/10 (`docs/planning_bloques_D_H.md`), actualizadas con
las decisiones de C, D y E; F4–F8 son nuevas. Las ⭐ **no son decisiones**. Cifras: submuestra dev, train del split D1
(`data/processed/dev/senal.parquet`).

Datos: positivos de train por usuario p10 = 12, mediana = 37, p90 = 173, p99 = 563; el 20,3 % de los usuarios tiene > 100 y el
8,0 % > 200 (y aportan el 39,6 % de los positivos). Notas bajas por usuario: mediana 7, p90 39; el 5,5 % no tiene ninguna.
Ejemplos de entrenamiento con C5 a: 1 503 681 (uno por positivo de train).

| # | Pregunta | Opciones | ⭐ | Implicación |
|---|---|---|---|---|
| F1 | Cómo se representa al usuario | a) embedding por `userId` · b) agregación del historial · c) secuencial (GRU/atención) | b | a) no sirve para Letterboxd ni para C3 c (anti-preferencias); c) más caro y difícil en C++. Con D1 c el split ya no obliga, pero Letterboxd sí. |
| F2 | Longitud del historial | a) últimas 50 · b) últimas 100 · c) todo, con muestreo de 100 en cada época · d) todo | b | 100 cubre entero al 80 % de los usuarios. Longitud fija = arrays rectangulares con relleno (fácil en C++). «Últimas» depende del desempate (D2) en los usuarios de un solo día. |
| F3 | Qué historial lleva cada ejemplo (C5 a: uno por positivo) | a) solo lo anterior al objetivo (cronológico) · b) todos los demás positivos de train del usuario, sin el objetivo · c) un corte aleatorio | b | a) es lo de YouTube (no ve el futuro), pero en el 59 % de los usuarios todo es del mismo día y el orden es el desempate por movieId. b) encaja con un pooling sin orden y con la evaluación (en test el historial es todo train); es «completar el conjunto». Se genera al vuelo (no se materializan 1,5 M historiales). |
| F4 | Qué entra en el historial | a) solo positivos · b) todo lo valorado · c) positivos + un segundo historial de notas bajas (C3 c) | c | La zona neutra (ni positivo ni nota baja, el ~27 %) no entra en v1. Bandera para el 5,5 % sin notas bajas. |
| F5 | Pesos del pooling | a) media simple · b) media ponderada con el peso de C4 × C9 · c) atención aprendida | b | b reutiliza los pesos que ya calcula `senal.py`; c es una mejora para la E3. |
| F6 | Embedding de las películas del historial | a) el mismo v_i de la torre de ítem (pesos compartidos) · b) una tabla de embeddings aparte | a | a) una película nueva o de Letterboxd ya tiene vector por su contenido, y hay menos parámetros. b) es lo de YouTube; más memoria y no sirve para películas sin ID. |
| F7 | Cómo se combina | a) concatenar [media positivos, media notas bajas, banderas, log n] → MLP → d · b) solo la media de positivos · c) diferencia positivos − notas bajas | a | La salida d y la torre de ítem deben tener la misma dimensión, y van con L2 (E7). Falta elegir d: 64, 128 ⭐ o 256 y el número de capas (1–2 ⭐). |
| F8 | Historial al evaluar | a) val: train; test: train + val · b) siempre solo train | a | a) es lo realista (el test es lo siguiente a todo lo anterior). No es fuga: el historial es entrada, no un estadístico ajustado (D3 sigue aplicando a media_u, P(i)…). La máscara C14 en test: lo valorado en train + val. |
