"""
ayudas.py — funciones que el estudiante sí puede leer.

    from ayudas import cargar_modelo, cargar_datos, dividir, ver_prompt, curva, entrega

    modelo = cargar_modelo("pequeno")
    datos = cargar_datos()
    busqueda, validacion = dividir(datos)
"""

import json

from oraculo import armar


MODELOS = {
    "pequeno": "Qwen/Qwen3-1.7B",
    # El repo por defecto de Ministral 3 es FP8: la T4 (sm_75) no lo soporta.
    "ministral3b": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "mistral7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
}


class Modelo:
    """Red + tokenizador + nombre, para armar el oráculo en una línea."""

    def __init__(self, red, tok, nombre):
        self.red, self.tok, self.nombre = red, tok, nombre


def _parchear_compute_dtype(cfg):
    """La T4 es sm_75: sin bf16 nativo. Solo se toca el dtype de cómputo
    para no perder llm_int8_skip_modules del checkpoint dynamic-quant."""
    qc = getattr(cfg, "quantization_config", None)
    if not qc:
        return False
    if isinstance(qc, dict):
        qc["bnb_4bit_compute_dtype"] = "float16"
    else:
        import torch

        qc.bnb_4bit_compute_dtype = torch.float16
    return True


def _resolver_modelo(modelo: str) -> str:
    if modelo in MODELOS:
        return MODELOS[modelo]
    if "/" in modelo:
        return modelo
    alias = ", ".join(MODELOS)
    raise ValueError(
        f"{modelo!r} no es un alias. Use {alias} "
        "o un id público de Hugging Face (org/nombre)."
    )


def _cargar_red(nombre, kwargs):
    """Ministral 3 llega como `Mistral3ForConditionalGeneration` (lleva torre
    de visión) y no está en el mapeo de AutoModelForCausalLM, aunque el oráculo
    solo le mande texto. El ValueError salta antes de bajar los pesos."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(nombre, **kwargs)
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(nombre, **kwargs)


def cargar_modelo(modelo="pequeno"):
    """Carga un alias (ver `MODELOS`) o un id público de HF.

    `token=False`: nunca pide login. Todos los checkpoints son públicos.
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No hay GPU. En Colab: Entorno de ejecución ▸ Cambiar tipo de "
            "entorno de ejecución ▸ T4 GPU. Luego reinicia y corre desde la celda 1."
        )

    nombre = _resolver_modelo(modelo)
    print(f"cargando {modelo} → {nombre}")
    tok = AutoTokenizer.from_pretrained(nombre, token=False)
    cfg = AutoConfig.from_pretrained(nombre, token=False)
    kwargs = {"device_map": {"": "cuda:0"}, "config": cfg, "token": False}
    if not _parchear_compute_dtype(cfg):
        kwargs["dtype"] = torch.float16
    red = _cargar_red(nombre, kwargs)
    red.eval()
    print(f"{nombre}  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return Modelo(red, tok, nombre)


def _promedio_restricciones(filas):
    return sum(len(x["ids"]) for x in filas) / len(filas) if filas else 0.0


def cargar_datos(ruta="datos_visibles.json"):
    datos = json.load(open(ruta))
    print(
        f"{len(datos)} instancias, "
        f"{_promedio_restricciones(datos):.1f} restricciones por instancia"
    )
    return datos


def dividir(datos):
    """(busqueda, validacion) — particiones disjuntas del mismo pool.

    El split viene marcado en los datos. Las dos mitades comparten distribución:
    validar mide si la config aguanta instancias nuevas, no restricciones de un
    tipo nuevo (ese salto se mide al calificar, con familias que no están aquí).
    """
    busqueda = [x for x in datos if x["split"] == "busqueda"]
    validacion = [x for x in datos if x["split"] == "validacion"]
    print(f"búsqueda: {len(busqueda)} inst, {_promedio_restricciones(busqueda):.1f} restr/inst")
    print(f"validación: {len(validacion)} inst, {_promedio_restricciones(validacion):.1f} restr/inst")
    return busqueda, validacion


def ver_prompt(config, instancia):
    """El texto exacto que el oráculo le mandaría al modelo."""
    return armar(config, instancia)


def curva(historial):
    """historial: lista de precisiones. Dibuja la mejor hasta cada consulta."""
    import matplotlib.pyplot as plt

    mejor = []
    m = 0.0
    for p in historial:
        m = max(m, p)
        mejor.append(m)
    plt.plot(mejor)
    plt.xlabel("consultas")
    plt.ylabel("mejor precisión")
    plt.show()


def entrega(grupo, config, semana, ruta="entrega.json"):
    d = {"grupo": grupo, "config": config, "semana": semana}
    json.dump(d, open(ruta, "w"), indent=1)
    print(json.dumps(d, indent=1))
    return d
