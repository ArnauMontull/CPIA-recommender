# Ideas

Cosas que no están en el alcance de E2/E3 pero que el modelo ya permite o casi. Cada una dice en qué estado está.

## Recomendar a una pareja o a un grupo (19/10/2026, idea de Arnau)

**Idea.** Dos personas (p. ej. Arnau y su novia) meten sus datos de Letterboxd y el modelo propone películas que **no ha visto ninguna
de las dos** y que **encajen con las dos**.

**Por qué el modelo lo permite sin reentrenar.** La torre de usuario da un vector `u` por persona y las películas comparten el mismo
espacio (`v`, norma 1). El score de una película para una persona es `u · v`, así que se pueden combinar los scores de varias personas
sobre el mismo catálogo. La inferencia de cada persona es la misma que la de un usuario nuevo (ver `src/demo_letterboxd.py`).

**Reglas para juntar dos personas** (`src/demo_pareja.py`, hecho):

| Regla | Qué hace | Cuándo conviene |
|---|---|---|
| Intersección de los top-k | Lo que sale a la vez en el top-k de las dos | Es la pregunta original. Con k pequeño sale casi vacía (k = 20 → 3 coincidencias de 20 en la prueba) |
| `minimo` | Ordena por `min(score₁, score₂)` («nadie sufre», *least misery*) | Cuando importa que ninguna odie la elección |
| `media` | Ordena por `(score₁ + score₂) / 2` | Cuando una puede compensar a la otra |

Siempre se excluye la **unión** de lo visto por las dos. Detalle útil: hay que combinar scores, no solo listas, porque la intersección
exacta tira información (una película 7.ª para una y 25.ª para la otra se pierde con k = 20).

**Prueba (Arnau + Uriel como pareja ficticia, semilla 42).** Intersección de top-k: k = 20 → 3 películas (15 %), k = 50 → 11 (22 %),
k = 200 → 64 (32 %). Con `minimo` y `media` el top-15 cambia poco: Marriage Story, Before Sunrise, The Departed, The Godfather,
Amélie, Dunkirk, Everything Everywhere All at Once... El resultado de `minimo` es más equilibrado (los dos scores quedan cerca); el de
`media` se inclina hacia quien tiene scores más altos en general (aquí, Arnau).

**Lo que no está resuelto.**
- **Sin validar.** No hay forma objetiva de saber si a una pareja real le gusta lo que sale. Una validación posible: parejas de usuarios
  de MovieLens con películas del test que han puntuado alto los dos (recall de la regla combinada).
- **Escala de los scores.** Cada persona tiene su propio rango de scores (los de Uriel son más bajos que los de Arnau); la media y el
  mínimo favorecen a quien tiene scores más altos. Normalizar cada persona (z-score o rango [0, 1] sobre los candidatos) antes de combinar.
- **Diversidad.** Se puede aplicar MMR encima (ver `demo_letterboxd --mmr`) para que no salgan sagas enteras.
- **Más de dos personas.** `scores_pareja` y `combinar` ya aceptan una lista de perfiles; falta probarlo.
- **Datos personales.** Cada persona debe dar su export de Letterboxd; todo se queda en `data/external/` (fuera de git).

**Siguientes pasos si se retoma:** normalizar scores por persona, validar con parejas de MovieLens, y como pieza de la demo de la exposición
(E3/E4): «dos perfiles → una película para ver juntos».

## Otras ideas pendientes

- **Watchlist como señal.** La watchlist (intereses declarados, no gustos) ya se usa como evaluación sin etiquetas
  (`src/evaluar_watchlist.py`); podría entrar como señal débil de entrada con un peso pequeño (requiere reentrenar).
- **Emparejado de Letterboxd con la API de TMDB como parte del pipeline por defecto.** Sin ella, ~31 % de las notas no entran
  (títulos distintos entre Letterboxd y MovieLens: «Star Wars» frente a «Star Wars: Episode IV - A New Hope»).
- **Corrección de títulos al mostrar.** MovieLens llama «Avengers: Infinity War - Part II» a *Avengers: Endgame*; una pequeña tabla
  por `tmdbId` arreglaría lo que se muestra sin tocar el modelo.
