# Planning de preprocesamiento: bloques D–H (preguntas abiertas)

Preguntas planteadas el 15/10/2026 junto con A–C; Arnau las dejó para más adelante («el bloque CDEFGH lo dejamos para un futuro»). Copiadas tal cual del planning original. **Ninguna está decidida.** Ojo: son anteriores a las decisiones del bloque C (16/10/2026), que pueden cambiar alguna implicación (p. ej. F3 frente a C5 a). Las decisiones van a `decisiones_preprocesamiento.md`.

## Bloque D — Partición train / validación / test (la más delicada)

**D1. ¿Cómo partes?**
- (a) Aleatoria por filas: **fuga de información**, porque entrenas con el futuro para predecir el pasado. Infla las métricas.
- (b) Temporal global: un corte por fecha (p. ej. test = último año). Es la más realista.
- (c) *Leave-last-out* por usuario: la última película de cada usuario va a test. Es la estándar en los papers (SASRec, NCF).
- **(d) Por usuarios: un 10 % de usuarios fuera del entrenamiento, y de cada uno se le da al modelo el 80 % de su historial y se predice el resto** ⭐.
- *Implicación*: (d) es exactamente tu escenario de Letterboxd (un usuario nuevo con su diario). Solo funciona si la torre de usuario **no usa un ID de usuario** (E1), y en ese caso es la más coherente con lo que vendes en E1.

**D2. Dentro de cada usuario de test, ¿el 80/20 es cronológico o aleatorio?**
- *Implicación*: el cronológico es más honesto, porque predices lo siguiente que verá. El aleatorio es más fácil. ⭐ Cronológico.

**D3. ¿Qué estadísticas se calculan solo con train?** Popularidad de cada película (para la corrección logQ de Yi 2019), vocabularios y medias para normalizar.
- *Implicación*: si se calculan con todo, hay **fuga silenciosa**. La pipeline debe ajustar con train y transformar el resto (el patrón `fit/transform`).

---

## Bloque E — Features de la torre de películas

**E1. ¿Cómo se generan los embeddings de las sinopsis?**
- (a) MiniLM (`all-MiniLM-L6-v2`, d = 384), como dice E1. Son ~45k sinopsis; en CPU, del orden de 5–15 minutos y **una sola vez**.
- (b) TF-IDF + SVD: sin red neuronal y muy rápido, pero semánticamente más pobre.
- *Implicación*: (a) se precalcula y se guarda en caché (`.npy`), así que no entra en el bucle de entrenamiento. Hay que decidir si se guarda en float16 (la mitad de memoria) o en float32.

**E2. ¿Qué parte de los créditos?**
- (a) Nada.
- **(b) Director + los 3–5 primeros actores** ⭐.
- (c) Todo el reparto y el equipo.
- *Implicación*: hay decenas de miles de actores. Si haces multi-hot de todos, el vocabulario explota y la mayoría aparecen una sola vez. Alternativas: una **frecuencia mínima** (p. ej. ≥ 5 películas; los demás van a `<OTRO>`) o *feature hashing* a un tamaño fijo.

**E3. ¿Incluyes las keywords?**
- *Implicación*: aportan matiz temático («time travel», «heist»), pero se solapan con la sinopsis. Opciones: incluirlas con frecuencia mínima, o añadirlas al texto antes de pasarlo por MiniLM (sin features nuevas).

**E4. ¿Features numéricas?** Año, duración, popularidad, `vote_average`.
- *Implicación importante*: meter la **popularidad como feature refuerza el sesgo de popularidad** que E1 dice combatir. ⭐ Año (normalizado o por décadas) y duración sí; popularidad no, o solo a través de logQ.

---

## Bloque F — La torre de usuario

**F1. ¿Cómo se representa a un usuario?**
- (a) Un embedding por `userId`: es clásico, pero **no sirve para usuarios nuevos**, y Letterboxd deja de funcionar.
- **(b) Una agregación del historial: la media (ponderada) de los embeddings de sus películas** ⭐.
- (c) Un modelo secuencial (GRU/atención, tipo SASRec): más potente, pero más caro y más difícil en C++.
- *Implicación*: esta decisión **define el formato de los datos**. Con (b)/(c) hay que guardar, para cada ejemplo, la secuencia de películas previas, no solo pares (usuario, película).

**F2. ¿Longitud del historial?**
- Las últimas N (p. ej. 50 o 100), o todo con muestreo.
- *Implicación*: hay usuarios con miles de ratings. Una N fija permite usar arrays rectangulares con relleno (*padding*), que es rápido y fácil en C++. N afecta a la RAM y a la latencia.

**F3. ¿Cómo se generan los ejemplos de entrenamiento?**
- (a) Para cada positivo, historial = todo lo anterior a él (cronológico). Es realista, pero da ~25M ejemplos.
- (b) Un ejemplo por usuario y época, con un corte aleatorio.
- *Implicación*: (a) no cabe materializado en memoria si guardas historiales de 50 IDs: 25M × 50 × 4 B = 5 GB. Hay que generarlo **al vuelo** en el `DataLoader` a partir de arrays ordenados.

---

## Bloque G — Letterboxd

**G1. ¿Cómo se emparejan las películas?** El export de Letterboxd trae `Name`, `Year` y una URI, pero **no trae ni tmdbId ni imdbId**.
- (a) Coincidencia exacta título + año (normalizando mayúsculas y acentos).
- (b) Coincidencia difusa (`rapidfuzz`) con un umbral.
- (c) Consultar la API de TMDB.
- *Implicación*: con (a) se pierden títulos traducidos o con «The» al final. Con (b) hay falsos positivos (remakes del mismo año). Hay que decidir qué % de coincidencias se acepta y registrar las que fallan.

**G2. ¿De dónde sale el export de prueba?** ¿El tuyo, el de tu compañero o uno simulado?
- *Implicación*: si es real, es un dato personal; mejor no subirlo al repo (`.gitignore`).

---

## Bloque H — Ingeniería: herramientas y formato

**H1. ¿Qué librería para leer y transformar?**
- (a) pandas: la conoces. Con los tipos por defecto (int64/float64), los 32M ocupan ~1 GB y el pico de lectura es de varios GB.
- **(b) Polars** ⭐: multihilo (aprovecha tus 12 hilos), lectura *lazy* y suele ir de 5 a 10 veces más rápido.
- (c) pyarrow directamente.
- *Implicación*: la E2 pide justificar las librerías y la E3 valora la eficiencia. Medir pandas frente a Polars en la lectura del CSV es **un benchmark gratis para el informe**.

**H2. ¿Tipos de datos?** `userId`/`movieId` en int32, `rating` en float32 (o uint8 guardando r × 2) y `timestamp` en int32 o int64.
- *Implicación*: 32M × (4 + 4 + 1 + 4) B ≈ 400 MB frente a ~1 GB por defecto. Encaja con lo que E1 promete («decenas de MB» para la inferencia).

**H3. ¿Remapeo de IDs?** Los `movieId` van hasta ~290 000 con huecos.
- ⭐ Remapear a índices contiguos `0..N-1` y guardar la tabla de mapeo.
- *Implicación*: las tablas de embeddings y los arrays en C++ lo necesitan. Además, **el mismo mapeo debe usarse en entrenamiento, en inferencia y con Letterboxd**, así que se guarda como un artefacto más.

**H4. ¿Formato de salida?**
- (a) Parquet: compacto, con tipos, ideal para Python.
- (b) `.npy` / binario crudo + un `meta.json`: **se lee trivialmente desde C++** con `fread` o `mmap`.
- (c) Ambos: Parquet para tablas y `.npy` para matrices de embeddings e índices.
- *Implicación*: piensa ya en la E3. El motor en C++ leerá embeddings y mapeos, y con (b)/(c) te ahorras un parser.

**H5. ¿Cómo se organiza la pipeline?**
- (a) Un único script.
- **(b) Etapas con cachés: `01_cargar` → `02_limpiar` → `03_split` → `04_features` → `05_export`** ⭐. Cada etapa escribe en `data/processed/` y se salta si su salida ya existe.
- *Implicación*: (b) es el «diagrama de flujo» de la E2 casi sin esfuerzo, y permite repetir solo la parte que cambia.

**H6. ¿Preprocesamiento en Python o en C++?**
- ⭐ En Python. El preprocesamiento se ejecuta una sola vez; donde la eficiencia se nota (y se evalúa) es en la **inferencia**: ANN + MMR en C++/OpenMP.
- *Implicación*: si alguien pregunta por qué no se hace todo en C++, este es el argumento para el informe.

**H7. ¿Reproducibilidad?** Semilla fija, versiones en `requirements.txt` y un `config.yaml` con todos los umbrales (k-core, umbral de rating, N, fracción).
- *Implicación*: cada decisión de esta lista acaba siendo **un parámetro del config**, no una constante escondida en el código.
