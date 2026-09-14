"""
calificar.py — lo corre el PROFESOR desde Colab, sobre el test privado de 294.

    oraculo, test = oraculo_test(modelo, "datos_test.json", cache="cache_test.json")
    r = calificar(oraculo, config)
    tabla = calificar_entregas(oraculo, "entregas")

Vive en `profesor/` junto a `datos_test.json`. El split oculto de búsqueda y
validación (`datos_ocultos.json`) no entra a git.
"""

import json
import sys
from pathlib import Path

from oraculo import Oraculo

RUTA_IFBENCH = "/content/IFBench"


def registro_ifbench(ruta=RUTA_IFBENCH):
    """INSTRUCTION_DICT de IFBench: trae las 44 familias del test.

    Se importa por separado del de IFEvalG (que ya trae `oraculo.py`) y se
    encadena con `registros=` — así el set visible sigue resolviendo por
    IFEvalG primero y su caché no se mueve ni un bit.
    """
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"no está clonado IFBench en {ruta!r}. En Colab:\n"
            f"    !git clone -q https://github.com/allenai/IFBench {ruta}"
        )
    if ruta not in sys.path:
        sys.path.insert(0, ruta)
    from ifbench import instructions_registry as _REGB

    return _REGB.INSTRUCTION_DICT


def oraculo_test(modelo, ruta_datos, cache, ruta_ifbench=RUTA_IFBENCH):
    """(oraculo, test) listos para calificar. `datos=test`, así que
    `oraculo.evaluar(config)` sin más argumentos ya corre las 294."""
    test = json.load(open(ruta_datos))
    o = Oraculo(modelo, test, cache=cache, registros=[registro_ifbench(ruta_ifbench)])
    print(f"{len(test)} instancias de test, {len({x['familia'] for x in test})} familias")
    return o, test


def _tabla_por_familia(trazas, total):
    conteos = {}
    for t in trazas:
        conteos[t["violo"]] = conteos.get(t["violo"], 0) + 1
    for familia, fallos in sorted(conteos.items(), key=lambda kv: -kv[1]):
        print(f"  {fallos:3}/{total:<3}  {familia}")


def calificar(oraculo, config, semilla=1, n=None):
    """Precisión sobre las 294 (o las primeras `n`) + tabla de fallos por
    familia. No imprime prompts ni respuestas — para eso está `ver_fallos`.
    """
    instancias = oraculo.datos if n is None else oraculo.datos[:n]
    r = oraculo.evaluar(config, instancias, semilla=semilla, desc="calificando")
    print(f"\nprecisión: {r.precision:.1%}  ({r.n} inst)")
    if r.trazas:
        print("fallos por familia:")
        _tabla_por_familia(r.trazas, r.n)
    return r


def ver_fallos(r, n=3):
    """Mira las trazas de un resultado a propósito. Un output de Colab se
    comparte por accidente con facilidad — por eso `calificar` no hace esto solo."""
    for t in r.trazas[:n]:
        print("violó:", t["violo"])
        print(t["salida"][:300])
        print("-" * 60)


def calificar_entregas(oraculo, carpeta, semilla=1, n=None):
    """Lee los entrega.json de `carpeta` (uno por subcarpeta o archivo suelto,
    cualquiera de las dos formas sirve) y devuelve la tabla ordenada por
    precisión. Dos grupos con la misma config cuestan una sola pasada: la
    clave de caché es {modelo}|{config}|{id}|{semilla}, no depende del grupo.
    """
    carpeta = Path(carpeta)
    rutas = sorted(carpeta.glob("**/entrega.json"))
    if not rutas:
        raise FileNotFoundError(f"no hay ningún entrega.json bajo {carpeta}")

    filas = []
    for ruta in rutas:
        entrega = json.load(open(ruta))
        r = calificar(oraculo, entrega["config"], semilla=semilla, n=n)
        filas.append((entrega.get("grupo", ruta.parent.name), r.precision, r.n))

    filas.sort(key=lambda x: -x[1])
    print(f"\n{'grupo':<12}{'precisión':>10}   n")
    for grupo, precision, n_inst in filas:
        print(f"{grupo:<12}{precision:>9.1%}   {n_inst}")
    return filas
