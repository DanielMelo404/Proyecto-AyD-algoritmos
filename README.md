# Proyecto Oráculo

Curso de Análisis y Diseño de Algoritmos. El trabajo es encontrar una buena configuración de prompt consultando un oráculo. Busquen y validen cuantas veces quieran: el caché guarda lo ya generado y las trazas de fallos vienen con cada resultado.

| Notebook | Colab |
|---|---|
| Proyecto (estudiantes) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo.ipynb) |
| Solución Backtracking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo_solucion_backtracking.ipynb) |
| Solución NPO · Qwen 1.7B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo_solucion_NPO_qwen17B.ipynb) |
| Solución NPO · Llama 3B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo_solucion_NPO_llama3B.ipynb) |
| Solución NPO · Ministral 3B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo_solucion_NPO_mistral3B.ipynb) |
| Solución NPO · Qwen 8B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo_solucion_NPO_qwen8B.ipynb) |

Los cuatro notebooks NPO implementan el **Algoritmo 1** de Chang y Chen (2026) [1]. El de
backtracking es la otra solución de ejemplo: recorre el catálogo con poda, sin teacher.

## Cómo se juega

1. Abran el notebook del curso en Colab (primera fila de la tabla).
2. Corran las celdas 1–3. En la 1, **reinicien el entorno** cuando lo pida.
3. Escriban su heurística **solo en la celda 4**.
4. Exporten `entrega.json` en la celda 5.

```
r = oraculo.evaluar(config, instancias, semilla)
r_val = oraculo.validar(config, n)

r.precision        # 0.55
r.trazas           # [{id, violo, salida}, ...]
```

Hay 1024 configuraciones (`espacio()`, temperatura 0.0): un índice por ranura entre `rol`, `estrategia`, `formato`, `estilo` y `cierre`, más la temperatura. Cada ranura es un consejo de prompting; algunos combinan mal con las restricciones que mide el verificador, y las trazas dicen cuál falló. El lote de búsqueda del notebook está fijo en 40 instancias. Medir con cuántas instancias buscar es parte del problema.

## Modelos

`cargar_modelo` acepta un alias o un id de Hugging Face:

| Alias | Checkpoint | Notas |
|---|---|---|
| `qwen17b` | `Qwen/Qwen3-1.7B` | el de por defecto; el más rápido |
| `ministral3b` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 3.8B en fp16 (~7.7 GB), T4 |
| `llama3b` | `unsloth/Llama-3.2-3B-Instruct` | 3.2B en fp16 (~6.4 GB), T4 |
| `qwen8b` | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | 4-bit, T4 |
| `mistral7b` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | 4-bit, T4 |

```
modelo = cargar_modelo("qwen8b")
```

Los 7-8B son lentos en T4: tengan paciencia con las primeras corridas.

## Búsqueda y validación

`datos_visibles.json` tiene 450 instancias: **150 de búsqueda y 300 de validación**. Son el mismo split que usan [GEPA](https://arxiv.org/abs/2507.19457) y NPO [1] sobre IF-RLVR Train, de donde salen estos datos. No son las filas exactas de los papers —no publican los índices—, sí la fuente, los tamaños y la disyunción.

```
datos = cargar_datos()
busqueda, validacion = dividir(datos)
oraculo = Oraculo(modelo, busqueda, validacion)
```

Las dos particiones salen del mismo pool y comparten distribución: validar mide si la config aguanta **instancias** nuevas, no restricciones de un tipo nuevo. Ese otro salto se mide al calificar, con familias que no están en este archivo.

Busquen solo sobre `busqueda`. `oraculo.validar` mide la config elegida con una muestra fija de `n` instancias (las 300 tardan) e imprime qué restricciones se cayeron más.

```
r_val = oraculo.validar(mejor[1], n=30)
```

### Ojo con el todo-o-nada

Una instancia trae **hasta 5 restricciones** (2.3 en promedio) y se puntúa todo-o-nada: basta que falle una para que valga 0. En las instancias más duras el modelo acierta poco, así que **con pocas instancias muchas configuraciones van a marcar `0.0` y parecer iguales**. Si su búsqueda no logra distinguir nada, suban el número de instancias antes de cambiar de heurística — decidir cuántas medir es parte del problema.

## Archivos públicos

| Archivo | Qué es |
|---|---|
| `proyecto_oraculo.ipynb` | El notebook del curso |
| `proyecto_oraculo_solucion_backtracking.ipynb` | Ejemplo de backtracking con poda, sobre `qwen17b` |
| `proyecto_oraculo_solucion_NPO_qwen17B.ipynb` | Ejemplo NPO [1] con `qwen17b` |
| `proyecto_oraculo_solucion_NPO_llama3B.ipynb` | Ejemplo NPO [1] con `llama3b` |
| `proyecto_oraculo_solucion_NPO_mistral3B.ipynb` | Ejemplo NPO [1] con `ministral3b` |
| `proyecto_oraculo_solucion_NPO_qwen8B.ipynb` | Ejemplo NPO [1] con `qwen8b` |
| `oraculo.py` | La caja negra. Se consulta, no se abre |
| `ayudas.py` | Cargar modelo, datos, dividir, ver un prompt, curva, entrega |
| `datos_visibles.json` | 450 instancias: 150 de búsqueda, 300 de validación |

## Entrega

Un `entrega.json` con grupo, configuración y semana:

```json
{
 "grupo": "G07",
 "config": {"rol": 3, "estrategia": 3, "formato": 3, "estilo": 3, "cierre": 3, "temperatura": 0.0},
 "semana": 3
}
```

## Para el profesor · calificar sobre el test privado

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/profesor/calificar.ipynb)

La nota final no sale de `datos_visibles.json`: sale de un test de 294 instancias con
restricciones que no están ahí. Los archivos están en `profesor/`:

| Archivo | Qué es |
|---|---|
| `profesor/calificar.ipynb` | Notebook para poner nota |
| `profesor/calificar.py` | `calificar` y `calificar_entregas` |
| `profesor/datos_test.json` | Las 294 instancias del test |

`calibracion.ipynb` (raíz del repo) se corre una vez en Colab, con `qwen17b`, antes de soltar el catálogo: barrido por ranura, piso, techo y ascenso por coordenadas.

`datos_ocultos.json` y los scripts de preparación no entran a git.

En Drive solo hace falta el caché y las entregas (una sola vez):

1. Crear `MyDrive/oraculo_profesor/entregas/` con un `entrega.json` por grupo, o por
   subcarpeta.
2. Abrir `calificar.ipynb` desde GitHub (badge de arriba) y correr las celdas 1–3.
   La celda 1 instala, clona `open-instruct` e `IFBench` y baja `calificar.py` y
   `datos_test.json`; al terminar pide reiniciar el entorno. La celda 2 monta Drive
   (caché + entregas) y carga el modelo — usar el mismo alias con el que buscó cada
   grupo, la clave de caché lo incluye.

Calificar, con el oráculo ya armado en la celda 3:

```python
r = calificar(oraculo, config, n=8)        # primero una muestra chica, cronometrada
r = calificar(oraculo, config)             # las 294, una vez que se sabe el costo

tabla = calificar_entregas(oraculo, CARPETA / "entregas")  # todos los grupos de una vez
```

`calificar` imprime precisión y fallos por familia, nunca prompts ni respuestas — para
mirar una traza hay que pedirla aparte con `ver_fallos(r)`. El caché queda en Drive:
una desconexión de Colab no cuesta la corrida, y dos grupos con la misma config sólo se
generan una vez.

## Referencias

[1] Yuan Chang y Xiaoqi Chen, *Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search*, arXiv:2608.27266, 2026. https://arxiv.org/abs/2608.27266
