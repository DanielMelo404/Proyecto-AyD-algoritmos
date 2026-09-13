Tu restricción real no es "que no esté en internet" — eso se resuelve solo generando. La restricción difícil es que la familia **separe configuraciones**: que la brecha entre el mejor prompt y el peor sea ≥15 puntos. Muchas tareas generables fallan ahí: o todas las configuraciones aciertan, o todas fallan.

Con ese filtro, cinco familias que funcionan:

| Familia | x se genera | y se calcula | Ranuras que activa |
|---|---|---|---|
| **Tabla + consulta agregada** | 15–40 filas de datos falsos + pregunta en lenguaje natural | pandas | descomposición, formato, verificación |
| **Mundo ficticio con reglamento** | 500 palabras de normativa inventada con umbrales aleatorios + un caso | la regla que el generador aplicó | extracción, CoT, verificación citando cláusula |
| **Autómata inventado** | reglas de transición inventadas + N pasos | simulas | CoT (brutal), ejemplos |
| **Puzzle de asignación** | permutación aleatoria → pistas que la fijan | la permutación | estrategia (tabla vs prosa), formato |
| **Restricciones de escritura compuestas** | k restricciones tomadas del catálogo | predicado, no igualdad | formato, verificación, rol |

**La que yo empezaría: tabla + consulta agregada.** El generador son 30 líneas, el verificador es `==` sobre un número, y la dificultad tiene tres perillas independientes (filas, número de condiciones del filtro, si pide aritmética de fechas o redondeo). Es la más fácil de calibrar al 40–70%.

**La más bonita para el enunciado: mundo ficticio.** Inventas "Reglamento de la Cooperativa Vhalen", artículos con umbrales numéricos aleatorios, y preguntas un caso que requiere dos saltos. La perilla de dificultad son las *cláusulas distractoras* — artículos que parecen aplicar y no aplican. Ahí es donde una configuración con verificación explícita le gana por 20 puntos a una sin ella, que es exactamente el efecto que tu proyecto quiere medir.

**Advertencia que te ahorra un piloto perdido:** no uses nada a nivel de caracteres — cifrados, contar letras, invertir palabras. Generan instancias perfectas y resultados inútiles, porque el modo de falla es el tokenizador, no el razonamiento. Precisión cerca de cero y ninguna configuración la mueve: ruido sin señal.

**Mezcla, no una sola familia.** Tres o cuatro, para que ninguna estrategia única domine el ranking. Si solo pones puzzles, gana quien escriba "piensa paso a paso" y se acabó el proyecto.

**Cómo aterrizarlo con lo que ya tienes:**

1. Cada generador emite `familia`, `semilla`, `nivel` y un verificador con la misma firma booleana que ya usa `oraculo.py` — no tocas el oráculo.
2. Generas cada instancia en tres niveles y corres el piloto sobre los tres. Te quedas con el nivel que caiga en banda. Esto reemplaza el tanteo a ciegas que tenías pendiente.
3. El oculto son **los mismos generadores con semilla distinta y parámetros corridos** (más distractoras, tabla más grande) más una familia que nunca vieron. Así el conjunto oculto mide generalización de verdad y no memorización de instancias.
4. Las 63 restricciones adversarias de la semana 6 entran como sexta familia en el privado.

Si te sirve, armo los dos primeros generadores (`genera_tablas.py` y `genera_reglamentos.py`) con los tres niveles ya parametrizados y el cribado de ítems incorporado, listos para el piloto. ¿Arranco por ahí?