"""
ayudas.py — funciones que el estudiante sí puede leer.

    from ayudas import cargar_modelo, cargar_datos, dividir, ver_prompt, curva, entrega

    modelo = cargar_modelo("qwen17b")
    datos = cargar_datos()
    busqueda, validacion = dividir(datos)

El oráculo (`oraculo.py`) se consulta, no se abre. Acá está lo que sí hace falta
para armarlo: bajar el modelo, leer el JSON visible y partir búsqueda/validación.
"""

import json
from collections import Counter

from oraculo import armar, RANURAS


MODELOS = {
    "qwen17b": "Qwen/Qwen3-1.7B",
    "qwen17b4bit": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    # El repo por defecto de Ministral 3 es FP8: la T4 (sm_75) no lo soporta.
    "ministral3b": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "mistral7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
}


class Modelo:
    """Red + tokenizador + nombre, para armar el oráculo en una línea.

    El oráculo no importa `transformers`: recibe este envoltorio. `nombre` entra
    en la clave de caché, así que cambiar de checkpoint no reusa respuestas
    de otro modelo.
    """

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
    """Alias de `MODELOS` → id de Hugging Face, o el id si ya trae barra.

    Un id `org/nombre` se deja pasar para cargar checkpoints que no están en
    la tabla sin tocar este archivo. Un nombre suelto que no es alias falla
    en claro, con la lista de alias válidos.
    """
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


def cargar_modelo(modelo="qwen17b"):
    """Carga un alias (ver `MODELOS`) o un id público de Hugging Face.

    `token=False`: nunca pide login. Todos los checkpoints de la tabla son
    públicos. Exige GPU: en Colab hay que elegir T4 antes, si no el mensaje
    indica el menú. Devuelve un `Modelo` listo para `Oraculo(...)`.

    Parameters
    ----------
    modelo:
        Alias (`qwen17b`, `llama3b`, …) o id `org/nombre`.
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
    """Restricciones por instancia. El verificador es todo-o-nada: una media
    más alta implica instancias más difíciles de acertar de un solo golpe."""
    return sum(len(x["ids"]) for x in filas) / len(filas) if filas else 0.0


def cargar_datos(ruta="datos_visibles.json"):
    """Lee el JSON visible (búsqueda y validación todavía mezcladas).

    No filtra el split: eso lo hace `dividir`. Imprime cuántas instancias hay
    y cuántas restricciones traen en promedio, para calibrar el tamaño de muestra
    antes de gastar generaciones.

    Parameters
    ----------
    ruta:
        Por defecto `datos_visibles.json` en el directorio de trabajo de Colab.
    """
    datos = json.load(open(ruta))
    print(
        f"{len(datos)} instancias, "
        f"{_promedio_restricciones(datos):.1f} restricciones por instancia"
    )
    return datos


def dividir(datos):
    """(busqueda, validacion) — particiones disjuntas y **curadas** del pool.

    El split viene marcado en los datos. Las dos mitades comparten distribución
    (mismo promedio de restricciones por instancia), así que validar mide si la
    config aguanta instancias nuevas y no un lote más difícil.

    El archivo trae además instancias marcadas `descartada`, que ninguna de las
    dos particiones usa. Se sacaron porque rompían la medición, no por difíciles:

    - piden más texto del que caben en `max_new_tokens` (imposibles de acertar),
    - ninguna opción del catálogo puede romperlas (puntos que nadie puede perder),
    - traen una sola restricción (el modelo las acierta siempre y nada las toca),
    - traen cuatro o cinco (el modelo casi nunca las acierta, no informan).

    Sin esa curaduría el puntaje casi no dependía de la configuración elegida,
    que es justo lo que el ejercicio tiene que medir.
    """
    busqueda = [x for x in datos if x["split"] == "busqueda"]
    validacion = [x for x in datos if x["split"] == "validacion"]
    descartadas = sum(1 for x in datos if x["split"] == "descartada")
    print(f"búsqueda: {len(busqueda)} inst, {_promedio_restricciones(busqueda):.1f} restr/inst")
    print(f"validación: {len(validacion)} inst, {_promedio_restricciones(validacion):.1f} restr/inst")
    if descartadas:
        print(f"({descartadas} instancias descartadas: no medían nada — ver `dividir`)")
    return busqueda, validacion


class Registro:
    """Anota cada consulta al oráculo y dice qué está midiendo la búsqueda.

    Guardar solo la mejor precisión esconde lo que hace falta para mejorar una
    heurística: si una ranura no está haciendo nada, si el lote tiene
    instancias que nadie resuelve (o que resuelve cualquiera), y qué
    restricción se está cayendo siempre.

        reg = Registro(INSTANCIAS)
        for ...:
            r = oraculo.evaluar(config, INSTANCIAS, semilla=1)
            print(reg.anotar(config, r))
        reg.resumen()
    """

    # Por debajo de esto, la ranura no separa sus opciones: el texto no pega.
    RANGO_MUERTO = 0.02
    # Con menos muestras por opción el rango es ruido y no se avisa nada.
    MUESTRA_MINIMA = 3

    def __init__(self, instancias, total=None):
        self.instancias = list(instancias)
        self.total = total
        self.filas = []  # (config, precision, ids que fallaron, familias violadas)
        self.aciertos = {x["id"]: 0 for x in self.instancias}
        self.mejor = None

    # ---- durante la búsqueda -----------------------------------------

    def anotar(self, config, resultado):
        """Guarda una evaluación y devuelve la línea para imprimir."""
        fallaron = {t["id"] for t in resultado.trazas}
        familias = Counter(t["violo"] for t in resultado.trazas if t.get("violo"))
        self.filas.append((dict(config), resultado.precision, fallaron, familias))
        for x in self.instancias:
            if x["id"] not in fallaron:
                self.aciertos[x["id"]] += 1

        nueva = self.mejor is None or resultado.precision > self.mejor[0]
        if nueva:
            self.mejor = (resultado.precision, dict(config))

        n = len(self.filas)
        cuantas = f"{n:3d}/{self.total}" if self.total else f"{n:3d}"
        peor = familias.most_common(1)
        return (
            f"eval {cuantas}  {self._indices(config)}  "
            f"esta {resultado.precision:5.1%}  mejor {self.mejor[0]:5.1%}"
            + ("  ← nueva mejor" if nueva else "")
            + (f"   cae: {peor[0][0]} ×{peor[0][1]}" if peor else "")
        )

    @staticmethod
    def _indices(config):
        """La config como una cadena corta: un dígito por ranura, más la temp."""
        s = "".join(str(config.get(r, "-")) for r in RANURAS)
        t = config.get("temperatura")
        return f"[{s}]" + (f"@{t}" if t else "")

    # ---- después de la búsqueda --------------------------------------

    def resumen(self, top=6):
        """Imprime el panel de diagnóstico completo."""
        if not self.filas:
            print("sin evaluaciones registradas")
            return
        self._puntajes()
        self._por_ranura()
        self._instancias(top)
        self._familias(top)

    def _puntajes(self):
        ps = sorted(p for _, p, _, _ in self.filas)
        n = len(ps)
        q = lambda f: ps[min(n - 1, int(f * n))]
        print(f"=== {n} evaluaciones ===")
        print(
            f"  peor {ps[0]:5.1%}   p25 {q(0.25):5.1%}   mediana {q(0.5):5.1%}   "
            f"p75 {q(0.75):5.1%}   mejor {ps[-1]:5.1%}"
        )
        print(f"  mejor config: {self._indices(self.mejor[1])}  {self.mejor[1]}")

    def _por_ranura(self):
        """Puntaje medio de las configs que llevan cada opción.

        Una ranura cuyo mejor y peor opción sacan casi lo mismo no está
        separando nada: su texto no le hace efecto al modelo, y la búsqueda
        gasta consultas en una dimensión que no informa.
        """
        print("\n=== por ranura (media de las configs que llevan cada opción) ===")
        for ranura in RANURAS:
            medias = {}
            for config, p, _, _ in self.filas:
                if ranura in config:
                    medias.setdefault(config[ranura], []).append(p)
            if len(medias) < 2:
                print(f"  {ranura:11} — solo se probó una opción")
                continue
            prom = {i: sum(v) / len(v) for i, v in medias.items()}
            mejor_i = max(prom, key=prom.get)
            rango = max(prom.values()) - min(prom.values())
            detalle = "  ".join(
                f"[{i}]{prom[i]:.0%}×{len(medias[i])}" + ("*" if i == mejor_i else "")
                for i in sorted(prom)
            )
            flojo = min(len(v) for v in medias.values()) < self.MUESTRA_MINIMA
            if flojo:
                aviso = "   (muestra corta: el rango puede ser ruido)"
            elif rango < self.RANGO_MUERTO:
                aviso = "   ⚠ SIN EFECTO: esta ranura no separa nada"
            else:
                aviso = ""
            print(f"  {ranura:11} rango {rango:5.1%}{aviso}")
            print(f"              {detalle}")
        print("  (* = mejor media;  ×n = cuántas veces se probó)")
        print(
            "  Para leer el rango hace falta probar cada opción varias veces con las\n"
            "  otras ranuras FIJAS. Al azar cada opción sale 1-2 veces y el rango es\n"
            "  ruido; un barrido por ranura (celda 4 de calibracion.ipynb) sí lo mide."
        )

    def _instancias(self, top):
        """Instancias que no informan: nadie las resuelve, o las resuelve cualquiera."""
        nunca = [i for i, n in self.aciertos.items() if n == 0]
        siempre = [i for i, n in self.aciertos.items() if n == len(self.filas)]
        n = len(self.instancias)
        print(f"\n=== lote de {n} instancias ===")
        print(f"  nunca resueltas:   {len(nunca):3}  ({len(nunca)/n:.0%}) — no separan configs")
        print(f"  siempre resueltas: {len(siempre):3}  ({len(siempre)/n:.0%}) — puntos regalados")
        print(f"  informativas:      {n - len(nunca) - len(siempre):3}")
        if nunca[:top]:
            print(f"  ejemplos nunca resueltas: {', '.join(map(str, nunca[:top]))}")

    def _familias(self, top):
        total = Counter()
        for _, _, _, fam in self.filas:
            total += fam
        print("\n=== restricciones que más se cayeron (sobre todas las evaluaciones) ===")
        for familia, veces in total.most_common(top):
            print(f"  {veces:5}  {familia}")


def ver_prompt(config, instancia):
    """El texto exacto que el oráculo le mandaría al modelo.

    No genera tokens: solo sustituye los índices de la config por el texto del
    catálogo y lo pega al prompt de la instancia. Sirve para inspeccionar una
    config sin gastar una consulta.
    """
    return armar(config, instancia)


def curva(historial):
    """historial: lista de precisiones. Dibuja la mejor hasta cada consulta.

    El eje Y no es 'esta config' sino 'el mejor hasta ahora': explorar configs
    peores no baja la curva, y se ve si el presupuesto extra sigue encontrando
    mejoras o ya se estancó.
    """
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
    """Escribe el JSON de entrega: grupo, config ganadora y semana.

    La nota no sale de `datos_visibles.json`: el profesor corre esta config
    sobre el test privado. Imprime el JSON y lo deja en disco para descargarlo.

    Parameters
    ----------
    grupo:
        Identificador del grupo, p. ej. ``"G07"``.
    config:
        Dict con índices de ranura y ``temperatura``.
    semana:
        Número de semana del curso.
    ruta:
        Archivo de salida. El notebook lo descarga con ``files.download``.
    """
    d = {"grupo": grupo, "config": config, "semana": semana}
    json.dump(d, open(ruta, "w"), indent=1)
    print(json.dumps(d, indent=1))
    return d
