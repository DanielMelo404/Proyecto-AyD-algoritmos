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
    "qwen8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "mistral7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
}

# Familias reservadas: prefijos disjuntos de los de búsqueda, para imitar
# el salto del set oculto (restricciones de un tipo que no se vio al buscar).
FAMILIAS_VALIDACION = (
    "combination:repeat_prompt",
    "last_word:last_word_sent",
    "last_word:last_word_answer",
    "punctuation:punctuation_exclamation",
    "length_constraints:nth_paragraph_first_word",
)


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


def cargar_modelo(modelo="pequeno"):
    """Carga un alias (`pequeno`, `qwen8b`, `mistral7b`) o un id público de HF.

    `token=False`: nunca pide login. Los tres checkpoints son públicos.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    red = AutoModelForCausalLM.from_pretrained(nombre, **kwargs)
    red.eval()
    print(f"{nombre}  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return Modelo(red, tok, nombre)


def cargar_datos(ruta="datos_visibles.json"):
    datos = json.load(open(ruta))
    print(f"{len(datos)} instancias, {len({x['familia'] for x in datos})} familias")
    return datos


def dividir(datos):
    """(busqueda, validacion) — familias y prefijos disjuntos."""
    validacion = [x for x in datos if x["familia"] in FAMILIAS_VALIDACION]
    busqueda = [x for x in datos if x["familia"] not in FAMILIAS_VALIDACION]
    print(
        f"búsqueda: {len(busqueda)} inst, {len({x['familia'] for x in busqueda})} familias"
    )
    print(
        f"validación: {len(validacion)} inst, {len({x['familia'] for x in validacion})} familias"
    )
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
