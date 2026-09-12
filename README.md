# Proyecto Oráculo

Curso de Análisis y Diseño de Algoritmos. El trabajo es encontrar una buena configuración de prompt consultando un oráculo: cada evaluación nueva cuesta rollouts; las trazas de fallos son gratis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanielMelo404/Proyecto-AyD-algoritmos/blob/main/proyecto_oraculo.ipynb)

## Cómo se juega

1. Abran el notebook en Colab (botón de arriba).
2. Corran las celdas 1–3. En la 1, **reinicien el entorno** cuando lo pida.
3. Escriban su heurística **solo en la celda 4**.
4. Exporten `entrega.json` en la celda 5.

```
r = oraculo.evaluar(config, instancias, semilla)

r.precision        # 0.55
r.trazas           # [{id, violo, salida}, ...]   ← gratis
oraculo.gastado    # rollouts gastados
```

Hay 72 configuraciones (`espacio()`). El presupuesto lo fija el enunciado de cada semana.

## Modelos

`cargar_modelo` acepta un alias o un id de Hugging Face:

| Alias | Checkpoint | Notas |
|---|---|---|
| `pequeno` | `Qwen/Qwen3-1.7B` | el de ahora; más adelante se retira |
| `qwen8b` | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | 4-bit, T4 |
| `mistral7b` | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | 4-bit, T4 |

```
modelo = cargar_modelo("qwen8b")
```

Los 7-8B son lentos en T4: eso es el tope real, no un contador.

## Búsqueda y validación

`datos_visibles.json` tiene 300 instancias en 20 familias. `dividir` reserva 5 familias (75 instancias) cuyos prefijos no aparecen en las otras 15 (225 instancias):

```
datos = cargar_datos()
busqueda, validacion = dividir(datos)
oraculo = Oraculo(modelo, busqueda)
```

Busquen solo sobre `busqueda`. `validar` mide la config elegida en las familias reservadas y **no gasta presupuesto**. La nota final se mide en familias distintas a las dos particiones.

```
from ayudas import validar
r_val = validar(oraculo, mejor[1], datos)
```

## Archivos públicos

| Archivo | Qué es |
|---|---|
| `proyecto_oraculo.ipynb` | El notebook de Colab |
| `oraculo.py` | La caja negra. Se consulta, no se abre |
| `ayudas.py` | Cargar modelo, datos, dividir, validar, ver un prompt, curva, entrega |
| `datos_visibles.json` | 300 instancias, 20 familias |

## Entrega

Un `entrega.json` con grupo, configuración y semana:

```json
{
 "grupo": "G07",
 "config": {"rol": 2, "estrategia": 1, "formato": 1, "verificacion": 1, "temperatura": 0.0},
 "semana": 3
}
```
