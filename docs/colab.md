# Entrenar en Google Colab (H8)

Guía paso a paso. El portátil no tiene GPU NVIDIA, así que el entrenamiento del conjunto **completo** se hace en Colab (GPU T4 gratis).
El código no depende del dispositivo: `dev` también se puede probar en local (CPU).

**El repo `ArnauMontull/CPIA-recommender` es público** (comprobado con `git ls-remote` sin credenciales el 16/10/2026): el notebook lo clona sin
token. La sección del token (paso 6) solo hace falta si algún día lo pasas a privado.

## 0. Antes de empezar (en tu PC)

1. **Haz push del código a GitHub.** El notebook clona desde GitHub, así que lo que no esté en `main` no existe para Colab
   (incluidos `notebooks/entrenar_colab.ipynb`, `src/entrenar.py` cuando esté, etc.):
   ```
   git add -A
   git commit -m "Colab y entrenamiento"
   git push origin main
   ```
2. `data/` está fuera de git, por eso los datos viajan aparte, en un zip por Drive (pasos 2 y 3).

## 1. Crear la carpeta en Drive

1. Abre <https://drive.google.com> con tu cuenta de Google.
2. Botón **Nuevo** > **Carpeta nueva** > nombre exacto: `CPIA` > **Crear**. Debe quedar en «Mi unidad» (`MyDrive/CPIA`).

## 2. Empaquetar los datos (en tu PC)

Desde la raíz del repo (`CPIA-recommender/`), con el venv del proyecto:

```
C:\Users\arnau\Desktop\LIFE_OS\UPC\CPIA\CPIA_Project\venv\Scripts\python.exe scripts\empaquetar_datos.py --muestra completo
```

(Para una prueba rápida usa `--muestra dev`: ~140 MB.) Crea en `dist/`:
- `datos_completo.zip`: `data/processed/completo/`, `emb_texto.npy` y `movies.parquet`;
- `datos_completo.sha256`: la huella para comprobar que no se corrompió al subirlo.

El script imprime el tamaño y el sha256. `dist/` está en `.gitignore`.

## 3. Subir el zip a Drive

1. En Drive, entra en la carpeta `CPIA`.
2. Arrastra `dist\datos_completo.zip` y `dist\datos_completo.sha256` a la ventana (o **Nuevo** > **Subida de archivos**).
3. Espera a que termine la barra de subida (el zip completo puede ser de varios cientos de MB o más).

Al final debes ver en `MyDrive/CPIA/`: `datos_completo.zip` y `datos_completo.sha256`.

## 4. Abrir el notebook desde GitHub

Abre este enlace (iniciado sesión con la misma cuenta de Google):

<https://colab.research.google.com/github/ArnauMontull/CPIA-recommender/blob/main/notebooks/entrenar_colab.ipynb>

Si da error 404, el push del paso 0 no se hizo.

## 5. Activar la GPU T4

1. Menú **Entorno de ejecución** > **Cambiar tipo de entorno de ejecución**.
2. En «Acelerador por hardware», elige **GPU T4** > **Guardar**.
3. Si no te deja elegir T4, es que la cuota gratis está agotada: prueba más tarde (horas) o al día siguiente.

La celda 1 del notebook lo comprueba: debe mostrar la tabla de `nvidia-smi` y `CUDA disponible: True`.

## 6. Token de GitHub (solo si el repo es privado)

1. GitHub > tu foto > **Settings** > **Developer settings** > **Personal access tokens** > **Fine-grained tokens** > **Generate new token**.
2. Nombre: `colab-cpia`. Caducidad: 30 días.
3. **Repository access** > **Only select repositories** > `CPIA-recommender`.
4. **Permissions** > **Repository permissions** > **Contents: Read-only** (nada más).
5. **Generate token** y copia el valor (empieza por `github_pat_`; no se vuelve a mostrar).
6. En Colab, icono de la **llave** (Secretos) en la barra izquierda > **Añadir secreto nuevo**:
   - Nombre: `GITHUB_TOKEN` (exacto);
   - Valor: el token;
   - activa **Acceso desde el notebook**.
7. No pegues el token en ninguna celda ni lo subas a git. El notebook lo lee de los secretos y no lo imprime.

## 7. Ejecutar

1. En la primera celda, deja `MUESTRA = "completo"` (o `"dev"` para probar).
2. **Entorno de ejecución** > **Ejecutar todas**, o celda a celda con Mayús+Intro (mejor la primera vez).
3. Celda 2 (Drive): aparece una ventana pidiendo permiso. Acepta.
4. Celda de tests: debe terminar en verde.
5. Celda de entrenamiento: **`src/entrenar.py` aún no existe**. Hasta que se escriba (tras el preprocesamiento), esa celda da error;
   las anteriores (GPU, repo, dependencias, datos, tests) sí sirven para dejar el entorno listo.
6. Los checkpoints se escriben en `MyDrive/CPIA/modelos/`, así que sobreviven aunque la sesión muera.

## 8. Límites del plan gratis

- Una sesión dura **como máximo ~12 h** y se corta antes (~**90 min sin actividad** en la pestaña, o si cierras el navegador/portátil).
  Puede cortarse sin avisar.
- La GPU no está garantizada: a veces te desconectan si has usado mucha en los últimos días.
- El disco de Colab (`/content`) **se borra** al terminar la sesión. Solo lo que esté en Drive sobrevive: por eso los checkpoints van a
  `MyDrive/CPIA/modelos`.
- Mantén la pestaña abierta y el portátil enchufado y sin suspender mientras entrena.

## 9. Si se corta la sesión

1. Abre de nuevo el notebook (paso 4) y **Entorno de ejecución** > **Conectar**.
2. Vuelve a ejecutar las celdas 1 a 6 (GPU, Drive, repo, dependencias, datos, tests): el disco estaba vacío, así que se repite todo.
3. Revisa en Drive `MyDrive/CPIA/modelos/` qué checkpoints hay.
4. Ejecuta de nuevo la celda de entrenamiento. Cuando `src.entrenar` exista, deberá reanudar desde el último checkpoint de esa carpeta
   (requisito para quien lo escriba); si no reanuda, empezará de cero.
5. Si no te dan GPU: espera unas horas, o entrena `dev` (`MUESTRA = "dev"`) para seguir avanzando.

## Problemas típicos

| Síntoma | Causa y arreglo |
|---|---|
| `No encuentro .../datos_completo.zip` | Carpeta o nombre mal en Drive: debe ser `MyDrive/CPIA/datos_completo.zip`. |
| `sha256 distinto` | El zip se subió incompleto: vuélvelo a subir. |
| `git ha fallado` al clonar | Repo privado sin secreto `GITHUB_TOKEN` (paso 6), o secreto sin «Acceso desde el notebook». |
| `CUDA disponible: False` | GPU sin activar (paso 5). Tras cambiarla, la sesión se reinicia: ejecuta desde la celda 1. |
