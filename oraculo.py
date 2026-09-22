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
import time
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
# Cada ranura es un índice en su lista. El último índice es la única
# opción segura; las anteriores son extras que el verificador rechaza
# (cierre, largo, markdown, comas, preámbulo). `armar` pega primero las
# seguras y después las dañinas, detrás de la petición, para que una
# línea útil no cancele una dañina. El techo se queda cerca de 30%.

CATALOGO = {
    "rol": [
        "Let me know if you need anything else after finishing the answer.",
        "Hope this helps! Add a friendly closing offer of further help.",
        "Always finish by asking the user what else you can do.",
        "You are a meticulous assistant who follows formatting requirements exactly.",
    ],
    "estrategia": [
        "Write a thorough detailed answer of at least four hundred words.",
        "Expand the answer into several extra paragraphs of additional commentary.",
        "Add two extra paragraphs of personal commentary after the main answer.",
        "Write a draft, check it against each requirement, "
        "then output only the corrected final version.",
    ],
    "formato": [
        "Use markdown headers and bullet points throughout your entire answer.",
        "Wrap the entire answer inside a single fenced markdown code block.",
        "Begin the answer with a markdown heading and a numbered list.",
        "Format the answer as bullet points under at least two markdown headers.",
        "Output only the requested text. No preambles or commentary.",
    ],
    "estilo": [
        "Use a comma in every sentence of the entire written answer.",
        "End every sentence with an exclamation mark instead of a period.",
        "Feel free to use commas and extra punctuation in every sentence.",
        "Write each sentence with at least one comma and one exclamation.",
        "Respect exact word counts, casing and punctuation rules.",
    ],
    "verificacion": [
        "Start the reply with the preamble Sure, here you go.",
        "Add a second paragraph of personal commentary after the main answer.",
        "Before the answer, write a short introduction that restates the question.",
        "Check each requirement silently, then output only the corrected answer.",
    ],
    # Solo la plantilla segura trae `{restricciones}`: lo rellena `_recordatorio`.
    "recordatorio": [
        "Ignore the extra requirements and just answer the question naturally.",
        "End by asking what else you can help the user with.",
        "Add a second paragraph of personal commentary after finishing the answer.",
        "These are the requirements your response must satisfy:\n{restricciones}",
    ],
}

TEMPERATURAS = [0.0, 0.3, 0.7]

RANURAS = ["rol", "estrategia", "formato", "estilo", "verificacion", "recordatorio"]

# Presupuesto de un lote, en lote × largo². El prefill materializa una matriz de
# atención de lote × cabezas × largo² valores, así que el tope no puede ser solo
# el número de prompts: 8e6 son ~0.5 GB en fp16 con 32 cabezas, que es lo que
# aguanta una T4 con un 8B de 4 bits encima.
PRESUPUESTO_ATENCION = 8_000_000

# Segundos mínimos entre volcados del caché a disco durante una tanda.
PERIODO_GUARDADO = 60


def espacio(temperaturas=(0.0,)):
    """Producto cartesiano de los índices del catálogo × temperaturas.

    Por defecto temperatura fija en 0.0 → 4×4×5×5×4×4 = 6400 configs. Pasar
    `TEMPERATURAS` (0.0, 0.3, 0.7) triplica el espacio. Cada config es un dict
    ``{rol, estrategia, formato, estilo, verificacion, recordatorio, temperatura}``.
    """
    tamanos = [range(len(CATALOGO[r])) for r in RANURAS]
    return [
        dict(zip(RANURAS, combo), temperatura=t)
        for *combo, t in itertools.product(*tamanos, temperaturas)
    ]


def _opcion(config, ranura):
    """Texto de esa ranura. Una llave ausente no pega nada.

    Una entrega anterior a `estilo` (o a `recordatorio`) sigue armando el
    prompt de antes: la ranura que no trae equivale al vacío del catálogo,
    no al índice 0, que ahora puede ser una opción dañina.
    """
    if ranura not in config:
        return ""
    return CATALOGO[ranura][config[ranura]]


def _recordatorio(config, instancia):
    """Repite al final las restricciones que el prompt ya trae.

    `restriccion` viene separada por tabuladores, una por restricción y
    verbatim: es el mismo texto que el verificador va a medir, no una
    paráfrasis. Si la instancia no la trae, no se pega la plantilla que
    depende de `{restricciones}` — hay conjuntos (los ocultos) donde el
    campo viene vacío. Una plantilla sin ese hueco se pega igual: es una
    instrucción de cierre, no un recordatorio de la instancia.

    Una restricción puede ocupar varias líneas (las que traen un ejemplo de
    formato). Se copian tal cual, sin sangrar la continuación: el verificador
    busca marcadores exactos y sangrarlos invitaría al modelo a emitirlos con
    espacios de más.
    """
    plantilla = _opcion(config, "recordatorio")
    if not plantilla:
        return ""
    if "{restricciones}" not in plantilla:
        return plantilla
    crudo = instancia.get("restriccion") or ""
    partes = [p.strip() for p in crudo.split("\t") if p.strip()]
    if not partes:
        return ""
    return plantilla.format(restricciones="\n".join(f"- {p}" for p in partes))


def _es_segura(ranura, indice):
    """El último índice de cada ranura es la única opción que no rompe."""
    return indice == len(CATALOGO[ranura]) - 1


def armar(config, instancia):
    """Pega las ranuras no vacías detrás del prompt de la instancia.

    Primero van las opciones seguras (último índice de cada ranura),
    después las dañinas: el modelo de esta talla sigue lo último que lee,
    y una línea útil no puede cancelar un extra que el verificador rechaza.
    El índice 0 de todas las ranuras apila solo dañinas; el último índice
    de todas, solo las líneas que ya llegaban cerca de 30%.

    `config` sin `estilo` o sin `recordatorio` se acepta: una entrega vieja
    tiene que seguir calificando, y la ranura que falta no se pega.
    """
    seguras = []
    daninas = []
    for ranura in RANURAS:
        if ranura not in config:
            continue
        texto = (
            _recordatorio(config, instancia)
            if ranura == "recordatorio"
            else _opcion(config, ranura)
        )
        if not texto:
            continue
        if _es_segura(ranura, config[ranura]):
            seguras.append(texto)
        else:
            daninas.append(texto)
    return "\n\n".join([instancia["prompt"], *seguras, *daninas])


def _id_config(config):
    """Huella corta de la config: entra en la clave de caché.

    `sort_keys` para que dos dicts con las mismas ranuras en otro orden
    no disparen una generación de más.
    """
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────
#  El resultado de una consulta
# ─────────────────────────────────────────────────────────────────────


class Resultado:
    """Lo que devuelven `evaluar` y `validar`.

    `precision` es la fracción de instancias que cumplieron **todas** las
    restricciones. `trazas` trae solo los fallos (`id`, `violo`, `salida`);
    `violo` es la primera familia que no pasó, no un listado de todas.
    `n` es el tamaño del lote, para no inferirlo de las trazas.
    """

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
    """Caja negra: una config + instancias → precisión y trazas de fallo.

    `evaluar` busca sobre el conjunto que le pasen (en el notebook: `busqueda`).
    `validar` mide sobre la partición de validación, con muestra fija, para
    comparar configs en igualdad de condiciones. Lo ya generado se reusa;
    se puede consultar las veces que se quiera.
    """

    def __init__(
        self, modelo, datos, validacion=None, cache="cache_oraculo.json",
        max_new_tokens=384, lote=8, registros=None,
        presupuesto=PRESUPUESTO_ATENCION,
    ):
        """
        Parameters
        ----------
        modelo:
            `Modelo` de `ayudas.cargar_modelo` (red, tokenizador, nombre).
        datos:
            Instancias de búsqueda. `evaluar(config)` sin lista usa estas.
        validacion:
            Instancias de `dividir`. Hace falta para `validar`.
        cache:
            JSON en disco. En Colab conviene un path de Drive: una
            desconexión no tira las generaciones ya pagadas.
        max_new_tokens, lote:
            Tope de tokens nuevos y tamaño **máximo** de batch al generar. Un
            lote con prompts largos se arma más chico solo (ver `presupuesto`).
        registros:
            Verificadores extra (el de IFBench al calificar). El de IFEvalG
            ya va primero y cubre `datos_visibles.json`.
        presupuesto:
            Tope de `lote × largo²` por batch, en tokens². Bajarlo si la GPU
            se queda sin memoria; subirlo si sobra VRAM y se quiere ir rápido.
        """
        self.model = modelo.red
        self.tok = modelo.tok
        self.nombre_modelo = getattr(modelo, "nombre", "desconocido")
        self.datos = datos
        # Mezcla fija: la muestra de n instancias no depende del orden del archivo.
        self.validacion = list(validacion or [])
        random.Random(0).shuffle(self.validacion)
        self.max_new_tokens = max_new_tokens
        self.lote = lote
        self.presupuesto = presupuesto
        # Cadena de registros de verificadores, en orden de consulta. El de IFEvalG
        # va primero y cubre las familias de datos_visibles.json; quien califique
        # con otras familias agrega su registro detrás.
        self.registros = [_REG.INSTRUCTION_DICT, *(registros or [])]

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
        self._ultimo_guardado = 0.0
        if cache and os.path.exists(cache):
            self.cache = json.load(open(cache))
            print(f"caché: {len(self.cache)} respuestas recuperadas")

    # ---- consulta ----------------------------------------------------

    def evaluar(self, config, instancias=None, semilla=1, *, desc="evaluando"):
        """Evalúa una configuración sobre las instancias dadas.

        Lo ya calculado no se vuelve a generar: misma config + misma instancia
        + misma semilla + mismo modelo reusa la respuesta. `semilla` fija el
        muestreo cuando `temperatura > 0`; en 0.0 el decode es greedy.

        Returns
        -------
        Resultado
            `precision`, `trazas` (solo fallos) y `n`.
        """
        if instancias is None:
            instancias = self.datos

        k = _id_config(config)
        claves = [f"{self.nombre_modelo}|{k}|{x['id']}|{semilla}" for x in instancias]
        faltan = [(c, x) for c, x in zip(claves, instancias) if c not in self.cache]

        if faltan:
            torch.manual_seed(semilla)
            prompts = [armar(config, x) for _, x in faltan]
            # Se cachea lote por lote: si Colab se cae a mitad de tanda, lo ya
            # generado queda en disco y la próxima corrida no lo vuelve a pagar.
            for indices, respuestas in self._generar(
                prompts, config["temperatura"], desc=desc
            ):
                for i, resp in zip(indices, respuestas):
                    c, x = faltan[i]
                    r, violo = self._verificar(x, resp)
                    self.cache[c] = [r, violo, resp]
                self._guardar(forzar=False)
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

    def _lotes(self, largos):
        """Índices de `largos` agrupados en lotes, ordenados por longitud.

        Dos razones para no cortar la lista tal como viene: el padding de un
        lote lo fija el prompt más largo que le toque (agrupar por longitud
        ahorra cómputo), y la atención del prefill crece con el **cuadrado** de
        ese largo. Un prompt de 3k tokens metido en un lote de 8 pide 2.7 GB de
        una sola vez; acá va casi solo, aunque tarde más.
        """
        orden = sorted(range(len(largos)), key=lambda i: largos[i])
        lotes, actual = [], []
        for i in orden:
            cabe = (len(actual) + 1) * largos[i] ** 2 <= self.presupuesto
            if actual and (len(actual) >= self.lote or not cabe):
                lotes.append(actual)
                actual = []
            actual.append(i)  # un prompt solo entra siempre, aunque pase el tope
        if actual:
            lotes.append(actual)
        return lotes

    def _generar(self, prompts, temperatura, desc="evaluando"):
        """Itera `(indices, textos)` lote por lote.

        Entrega cada lote apenas sale —en vez de devolver la lista completa al
        final— para que `evaluar` lo cachee enseguida. `indices` apunta a
        `prompts`: los lotes no van en el orden de entrada (ver `_lotes`).
        Temperatura 0 → greedy; si no, muestreo con `top_p=0.9`.
        """
        textos = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
                **self._extra,
            )
            for p in prompts
        ]
        largos = [len(x) for x in self.tok(textos).input_ids]
        muestreo = (
            dict(do_sample=True, temperature=temperatura, top_p=0.9)
            if temperatura > 0
            else dict(do_sample=False)
        )

        barra = tqdm(total=len(prompts), desc=desc, unit="inst", leave=False)
        try:
            for indices in self._lotes(largos):
                yield indices, self._lote([textos[i] for i in indices], muestreo)
                barra.update(len(indices))
        finally:
            barra.close()

    def _lote(self, textos, muestreo):
        """Un lote. Si la GPU se queda sin memoria, reintenta de a un prompt:
        mucho más lento, pero no tira la corrida por un prompt largo suelto."""
        try:
            return self._decodificar(textos, muestreo)
        except torch.cuda.OutOfMemoryError:
            if len(textos) == 1:
                raise
            torch.cuda.empty_cache()
            print(f"sin VRAM con lote de {len(textos)}: reintentando de a uno")
            return [self._decodificar([t], muestreo)[0] for t in textos]

    @torch.no_grad()
    def _decodificar(self, textos, muestreo):
        """Tokeniza, genera sin gradientes y devuelve solo lo nuevo.

        `padding_side=left` (puesto en `__init__`) es lo que permite generar
        varios a la vez: el modelo escribe a la derecha del padding.
        """
        ent = self.tok(textos, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(
            **ent,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            **muestreo,
        )
        nuevos = out[:, ent["input_ids"].shape[-1] :]
        salidas = self.tok.batch_decode(nuevos, skip_special_tokens=True)
        del ent, out, nuevos  # el pico del lote no se suma al del siguiente
        return salidas

    def _clase(self, iid):
        """Clase del verificador para esa familia. IFEvalG primero; el resto
        de `registros` (IFBench al calificar) solo si el id no está ahí."""
        for reg in self.registros:
            if iid in reg:
                return reg[iid]
        raise KeyError(f"ningún registro conoce {iid!r}")

    def _verificar(self, instancia, respuesta):
        """(1, None) si cumple todo; (0, familia) si falla alguna.

        Recorre las restricciones en orden y se queda con la primera que falla:
        el todo-o-nada no necesita las demás, y `violo` apunta a esa familia.
        """
        for iid, kw in zip(instancia["ids"], instancia["kwargs"]):
            c = self._clase(iid)(iid)
            c.build_description(**kw)
            args = c.get_instruction_args()
            if args and "prompt" in args:
                c.build_description(prompt=instancia["prompt"])
            if not c.check_following(respuesta):
                return 0, iid
        return 1, None

    def _guardar(self, forzar=True):
        """Vuelca el caché a disco. Así un corte de Colab —o un OOM— no tira
        lo que ya se generó.

        Con `forzar=False` escribe como mucho una vez cada `PERIODO_GUARDADO`
        segundos: el caché se llena de respuestas largas y volcarlo a Drive
        después de cada lote termina costando más que generar.
        """
        if not self.ruta_cache:
            return
        ahora = time.monotonic()
        if not forzar and ahora - self._ultimo_guardado < PERIODO_GUARDADO:
            return
        json.dump(self.cache, open(self.ruta_cache, "w"))
        self._ultimo_guardado = ahora
