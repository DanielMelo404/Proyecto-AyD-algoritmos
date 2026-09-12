"""
oraculo.py — Proyecto Oráculo, Análisis y Diseño de Algoritmos.

Una caja negra que evalúa configuraciones de prompt. Se consulta, no se abre.

    from oraculo import Oraculo, CATALOGO, espacio

    o = Oraculo(modelo, datos)
    r = o.evaluar(config, datos[:20], semilla=1)

    r.precision    # 0.55
    r.trazas       # [{id, violo, salida}, ...]
    o.gastado      # 20
"""

import hashlib
import itertools
import json
import os
import sys

import torch

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
    def __init__(self, modelo, datos, cache="cache_oraculo.json", max_new_tokens=384, lote=8):
        self.model = modelo.red
        self.tok = modelo.tok
        self.datos = datos
        self.max_new_tokens = max_new_tokens
        self.lote = lote
        self.gastado = 0

        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "left"  # necesario para generar por lotes

        # Si el modelo tiene modo pensamiento (Qwen3), lo apagamos: multiplica
        # los tokens por 5-10 y tapa el efecto de la ranura de estrategia.
        self._extra = {}
        try:
            self.tok.apply_chat_template(
                [{"role": "user", "content": "x"}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            self._extra = {"enable_thinking": False}
            print("modo pensamiento: apagado")
        except TypeError:
            pass

        self.ruta_cache = cache
        self.cache = {}
        if cache and os.path.exists(cache):
            self.cache = json.load(open(cache))
            print(f"caché: {len(self.cache)} respuestas recuperadas")

    # ---- consulta ----------------------------------------------------

    def evaluar(self, config, instancias=None, semilla=1):
        """Evalúa una configuración. Cobra 1 rollout por tripleta nueva."""
        if instancias is None:
            instancias = self.datos

        k = _id_config(config)
        claves = [f"{k}|{x['id']}|{semilla}" for x in instancias]
        faltan = [(c, x) for c, x in zip(claves, instancias) if c not in self.cache]

        if faltan:
            torch.manual_seed(semilla)
            prompts = [armar(config, x) for _, x in faltan]
            respuestas = self._generar(prompts, config["temperatura"])
            for (c, x), resp in zip(faltan, respuestas):
                r, violo = self._verificar(x, resp)
                self.cache[c] = [r, violo, resp]
            self.gastado += len(faltan)
            self._guardar()

        res = [self.cache[c] for c in claves]
        precision = sum(r for r, _, _ in res) / len(res)
        trazas = [
            {"id": x["id"], "violo": v, "salida": s}
            for x, (r, v, s) in zip(instancias, res)
            if r == 0
        ]
        return Resultado(precision, trazas, len(res))

    # ---- por dentro --------------------------------------------------

    @torch.no_grad()
    def _generar(self, prompts, temperatura):
        salidas = []
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
            ent = self.tok(textos, return_tensors="pt", padding=True).to(self.model.device)
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
