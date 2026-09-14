"""
oraculo.py — Proyecto Oráculo, Análisis y Diseño de Algoritmos.

Una caja negra que evalúa configuraciones de prompt. Se consulta, no se abre.

    from oraculo import Oraculo, CATALOGO, espacio

    o = Oraculo(modelo, busqueda, validacion)
    r = o.evaluar(config, busqueda[:20], semilla=1)
    r_val = o.validar(config, n=15)

    r.precision    # 0.55
    r.trazas       # [{id, violo, salida}, ...]
"""

import hashlib
import itertools
import json
import os
import random
import sys
from collections import Counter

import torch
from tqdm.auto import tqdm

# Dónde está clonado allenai/open-instruct (trae los verificadores).
RUTA_VERIFICADORES = "/content/open-instruct"

if RUTA_VERIFICADORES not in sys.path:
    sys.path.insert(0, RUTA_VERIFICADORES)
from open_instruct.IFEvalG import instructions_registry as _REG


# ─────────────────────────────────────────────────────────────────────
#  El catálogo de ranuras
# ─────────────────────────────────────────────────────────────────────
# En inglés: las instancias y las restricciones vienen en inglés.

CATALOGO = {
    "rol": [
        "",
        "You are a helpful assistant.",
        "You are a meticulous assistant who follows formatting requirements exactly.",
    ],
    "estrategia": [
        "",
        "Think step by step before answering.",
        "First, list every requirement stated in the request. Then write your answer.",
        "Write a draft, check it against each requirement, "
        "then output only the corrected final version.",
    ],
    "formato": [
        "",
        "Output only the requested text. Do not add explanations, preambles or commentary.",
        "Begin your reply immediately with the answer itself.",
    ],
    "verificacion": [
        "",
        "Before finishing, verify that your answer satisfies every requirement.",
    ],
}

TEMPERATURAS = [0.0, 0.3, 0.7]

RANURAS = ["rol", "estrategia", "formato", "verificacion"]


def espacio(temperaturas=(0.0,)):
    """Todas las configuraciones posibles. Por defecto 72 (temperatura fija)."""
    tamanos = [range(len(CATALOGO[r])) for r in RANURAS]
    return [
        dict(zip(RANURAS, combo), temperatura=t)
        for *combo, t in itertools.product(*tamanos, temperaturas)
    ]


def armar(config, instancia):
    """config + instancia → el texto exacto que se le manda al modelo."""
    cabeza = [CATALOGO[r][config[r]] for r in ("rol", "estrategia", "formato")]
    cola = CATALOGO["verificacion"][config["verificacion"]]
    partes = [t for t in cabeza if t] + [instancia["prompt"]]
    if cola:
        partes.append(cola)
    return "\n\n".join(partes)


def _id_config(config):
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────
#  El resultado de una consulta
# ─────────────────────────────────────────────────────────────────────


class Resultado:
    def __init__(self, precision, trazas, n):
        self.precision = precision
        self.trazas = trazas
        self.n = n

    def __repr__(self):
        return f"Resultado(precision={self.precision:.1%}, fallos={len(self.trazas)}/{self.n})"


# ─────────────────────────────────────────────────────────────────────
#  El oráculo
# ─────────────────────────────────────────────────────────────────────


class Oraculo:
    def __init__(
        self, modelo, datos, validacion=None, cache="cache_oraculo.json",
        max_new_tokens=384, lote=8,
    ):
        self.model = modelo.red
        self.tok = modelo.tok
        self.nombre_modelo = getattr(modelo, "nombre", "desconocido")
        self.datos = datos
        # Mezcla fija: la muestra de n instancias no depende del orden del archivo.
        self.validacion = list(validacion or [])
        random.Random(0).shuffle(self.validacion)
        self.max_new_tokens = max_new_tokens
        self.lote = lote

        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "left"  # necesario para generar por lotes

        # Si el modelo tiene modo pensamiento (Qwen3), lo apagamos: multiplica
        # los tokens por 5-10 y tapa el efecto de la ranura de estrategia.
        # Se mira la plantilla: kwargs extra no lanzan TypeError en Mistral.
        plantilla = self.tok.chat_template or ""
        self._extra = {}
        if "enable_thinking" in plantilla:
            self._extra = {"enable_thinking": False}
            print("modo pensamiento: apagado")

        self.ruta_cache = cache
        self.cache = {}
        if cache and os.path.exists(cache):
            self.cache = json.load(open(cache))
            print(f"caché: {len(self.cache)} respuestas recuperadas")

    # ---- consulta ----------------------------------------------------

    def evaluar(self, config, instancias=None, semilla=1, *, desc="evaluando"):
        """Evalúa una configuración sobre las instancias dadas.
        Lo ya calculado no se vuelve a generar."""
        if instancias is None:
            instancias = self.datos

        k = _id_config(config)
        claves = [f"{self.nombre_modelo}|{k}|{x['id']}|{semilla}" for x in instancias]
        faltan = [(c, x) for c, x in zip(claves, instancias) if c not in self.cache]

        if faltan:
            torch.manual_seed(semilla)
            prompts = [armar(config, x) for _, x in faltan]
            respuestas = self._generar(prompts, config["temperatura"], desc=desc)
            for (c, x), resp in zip(faltan, respuestas):
                r, violo = self._verificar(x, resp)
                self.cache[c] = [r, violo, resp]
            self._guardar()

        res = [self.cache[c] for c in claves]
        precision = sum(r for r, _, _ in res) / len(res)
        trazas = [
            {"id": x["id"], "violo": v, "salida": s}
            for x, (r, v, s) in zip(instancias, res)
            if r == 0
        ]
        return Resultado(precision, trazas, len(res))

    def validar(self, config, n=None, semilla=1, top=8):
        """Precisión en la partición de validación, con muestra fija.

        `n`: cuántas instancias medir. Por defecto todas (tarda más).
        Las mismas n instancias en cada llamada, para comparar parejo.

        Imprime además las restricciones que más veces tumbaron la respuesta.
        Una instancia puede traer varias: se cuenta la primera que falló.
        """
        if not self.validacion:
            raise ValueError(
                "el oráculo se armó sin validación: "
                "Oraculo(modelo, busqueda, validacion)"
            )
        conjunto = self.validacion if n is None else self.validacion[:n]
        r = self.evaluar(config, conjunto, semilla=semilla, desc="validando")

        print(f"validación: {r.precision:.1%}  ({r.n} inst)")
        for iid, veces in Counter(t["violo"] for t in r.trazas).most_common(top):
            print(f"  {veces:3}  {iid}")
        return r

    # ---- por dentro --------------------------------------------------

    @torch.no_grad()
    def _generar(self, prompts, temperatura, desc="evaluando"):
        salidas = []
        barra = tqdm(total=len(prompts), desc=desc, unit="inst", leave=False)
        try:
            for i in range(0, len(prompts), self.lote):
                trozo = prompts[i : i + self.lote]
                textos = [
                    self.tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        add_generation_prompt=True,
                        tokenize=False,
                        **self._extra,
                    )
                    for p in trozo
                ]
                ent = self.tok(textos, return_tensors="pt", padding=True).to(
                    self.model.device
                )
                muestreo = (
                    dict(do_sample=True, temperature=temperatura, top_p=0.9)
                    if temperatura > 0
                    else dict(do_sample=False)
                )
                out = self.model.generate(
                    **ent,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tok.pad_token_id,
                    **muestreo,
                )
                nuevos = out[:, ent["input_ids"].shape[-1] :]
                salidas += self.tok.batch_decode(nuevos, skip_special_tokens=True)
                barra.update(len(trozo))
        finally:
            barra.close()
        return salidas

    def _verificar(self, instancia, respuesta):
        """(1, None) si cumple todo; (0, familia) si falla alguna."""
        for iid, kw in zip(instancia["ids"], instancia["kwargs"]):
            c = _REG.INSTRUCTION_DICT[iid](iid)
            c.build_description(**kw)
            args = c.get_instruction_args()
            if args and "prompt" in args:
                c.build_description(prompt=instancia["prompt"])
            if not c.check_following(respuesta):
                return 0, iid
        return 1, None

    def _guardar(self):
        if self.ruta_cache:
            json.dump(self.cache, open(self.ruta_cache, "w"))
