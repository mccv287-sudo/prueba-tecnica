"""Señales de desambiguación y probabilidad de cada candidato.

Cada señal vale entre 0 y 1; se combinan linealmente (``WEIGHTS``) y un softmax
sobre los candidatos de una mención da su probabilidad (un logit condicional):

- ``prior``: relevancia según la búsqueda de Wikidata (1 / posición). Se
  calcula a partir del orden.
- ``popularity``: prior empírico, artículos de Wikipedia (sitelinks) del
  candidato frente al más popular de la mención: "Paris" suele ser la ciudad.
- ``context``: palabras de la descripción del candidato cerca de la mención
  (sin contar las de la propia mención: la similitud de nombre ya está en
  ``prior``, porque Wikidata busca por etiqueta y alias).
- ``type``: compatibilidad con el tipo fino del NER (clases P31/P279*).
- ``relation``: entidades ya resueltas unidas al candidato por una declaración
  de Wikidata (separa homónimos de la misma clase, como dos personas).

Los pesos se aprenden por máxima verosimilitud en la muestra de AIDA
validation y se miden en la de test (``scripts/tune_weights.py``). Se
descartaron la similitud de nombre (con homónimos que se llaman igual, su peso
óptimo era 0) y la coherencia por clases compartidas con las entidades del
documento (no aportaba en ningún dataset: ``relation`` ya la captura).
"""

import math
import re
from collections.abc import Mapping

from scipy.special import softmax

from entity_linker.config import CONTEXT_CHARS
from entity_linker.config import WEIGHTS as DEFAULT_WEIGHTS
from entity_linker.schemas import WikidataCandidate
from entity_linker.wikidata.helpers import Classes

# Tipo fino del NER -> clase raíz de Wikidata: un "sports team" encaja con un
# candidato que desciende (P31/P279*) de Q12973014 y no con una ciudad. Sin
# entrada: no se filtra (tipo genérico)
FINE_TYPE_CLASSES = {
    "person": "Q5",  # human
    "sports team": "Q12973014",
    "company": "Q4830453",  # business
    "political party": "Q7278",
    "news agency": "Q192283",
    "international organization": "Q484652",
    "government agency": "Q327333",
    "university": "Q3918",
    "organization": "Q43229",
    "country": "Q6256",
    "nationality": "Q6256",  # AIDA enlaza los gentilicios al país ("Chinese" -> China)
    "city": "Q515",
    "region": "Q82794",
    "location": "Q2221906",  # geographic location
    "sports competition": "Q13406554",
    "event": "Q1190554",  # occurrence
    "award": "Q618779",
    # creative work, which includes books, movies, songs, paintings...
    "work of art": "Q17537576",
}

ROOT_CLASSES = set(FINE_TYPE_CLASSES.values())  # las únicas que se piden a Wikidata

_WORD = re.compile(r"[a-záéíóúüñ]{4,}")


def surroundings(text: str, start: int, end: int) -> str:
    """Texto alrededor de la mención (``CONTEXT_CHARS`` a cada lado), sin la
    propia mención."""
    return text[max(0, start - CONTEXT_CHARS) : start] + text[end : end + CONTEXT_CHARS]


def popularity_scores(
    cands: list[WikidataCandidate], sitelinks: dict[str, int]
) -> list[float]:
    """log(1 + artículos de Wikipedia) de cada candidato, relativo al más
    popular de la mención (entre 0 y 1)."""
    logs = [math.log1p(sitelinks.get(c.qid, 0)) for c in cands]
    top = max(logs, default=0.0)
    return [v / top if top else 0.0 for v in logs]


def context_score(cand: WikidataCandidate, context: str, mention: str) -> float:
    """Fracción de palabras de la descripción del candidato que aparecen cerca
    de la mención ("football club" en una noticia de fútbol).

    Las palabras de la mención no cuentan ni en el contexto ni en la
    descripción: que la descripción repita el nombre ("Nobel Prize in Physics
    controversies") es similitud de nombre, no de contexto."""
    name = _words(mention)
    words = _words(cand.description or "") - name
    return len(words & (_words(context) - name)) / len(words) if words else 0.0


def type_score(fine_label: str | None, qid: str, classes: Classes) -> float:
    """1 si el candidato desciende de la clase del tipo fino, 0 si no, 0,5 si no
    se sabe (tipo genérico o sin clases consultadas)."""
    root = FINE_TYPE_CLASSES.get(fine_label or "")
    if root is None or qid not in classes:
        return 0.5
    return 1.0 if root in classes[qid][1] else 0.0


def relation_score(qid: str, resolved: list[str], links: set[tuple[str, str]]) -> float:
    """Fracción de entidades ya resueltas del documento unidas al candidato por
    una declaración de Wikidata (o que son el mismo elemento): en un texto sobre
    Berkeley, "Michael Jordan" se acerca al profesor (empleador: UC Berkeley)
    y no al jugador de baloncesto."""
    if not resolved:
        return 0.0
    related = sum(q == qid or (qid, q) in links for q in resolved)
    return related / len(resolved)


def contributions(
    signals: dict[str, float], rank: int, weights: Mapping[str, float] = DEFAULT_WEIGHTS
) -> dict[str, float]:
    """Aportación de cada señal (peso × valor) a la puntuación de un candidato,
    más ``prior`` según su posición. También sirve para explicar la decisión."""
    return {
        "prior": weights["prior"] / (rank + 1),
        **{name: weights[name] * value for name, value in signals.items()},
    }


def probabilities(
    signals: list[dict[str, float]], weights: Mapping[str, float] = DEFAULT_WEIGHTS
) -> list[float]:
    """Probabilidad de cada candidato (en el orden de Wikidata): softmax de la
    suma de sus contribuciones."""
    if not signals:
        return []
    scores = [
        sum(contributions(sig, rank, weights).values())
        for rank, sig in enumerate(signals)
    ]
    return softmax(scores).tolist()


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))
