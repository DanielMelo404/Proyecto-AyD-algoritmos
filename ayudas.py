"""
ayudas.py — funciones que el estudiante sí puede leer.

    from ayudas import cargar_modelo, cargar_datos, ver_prompt, curva, entrega

    modelo = cargar_modelo()
    datos = cargar_datos()
"""

import json

from oraculo import armar


class Modelo:
    """Red + tokenizador juntos, para armar el oráculo en una línea."""

    def __init__(self, red, tok):
        self.red, self.tok = red, tok


def cargar_modelo(nombre="Qwen/Qwen3-1.7B"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nombre)
    red = AutoModelForCausalLM.from_pretrained(
        nombre, device_map={"": 0}, dtype=torch.float16
    )
    red.eval()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return Modelo(red, tok)


def cargar_datos(ruta="datos_visibles.json"):
    datos = json.load(open(ruta))
    print(f"{len(datos)} instancias, {len({x['familia'] for x in datos})} familias")
    return datos


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
