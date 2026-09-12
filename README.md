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

## Archivos públicos

| Archivo | Qué es |
|---|---|
| `proyecto_oraculo.ipynb` | El notebook de Colab |
| `oraculo.py` | La caja negra. Se consulta, no se abre |
| `ayudas.py` | Cargar modelo, datos, ver un prompt, curva, entrega |
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
