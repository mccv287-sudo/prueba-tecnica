"""Aprende los pesos de las señales (``WEIGHTS``) por máxima verosimilitud.

``signals.probabilities`` es un logit condicional: la probabilidad de cada
candidato es el softmax, sobre los candidatos de su mención, de la combinación
lineal de sus señales. Este script busca los pesos que maximizan la
probabilidad del QID correcto (``scipy.optimize``, pesos ≥ 0).

Protocolo habitual de AIDA: se aprende en la muestra de ``validation`` y se
mide en la de ``test``, frente a los pesos actuales de ``config.py``. El LLM
no tiene peso: si se le consulta, su elección es la respuesta.

Una regresión logística binaria (cada candidato por separado) no sirve: con un
candidato correcto de cada ~10 aprende a separar clases, no a ordenar los
candidatos de una mención, y en test empeoraba al primer candidato.

Las señales se calculan con ``Disambiguator.score_document``, el
mismo código que usa el servicio. relation depende de qué
menciones se resuelven en el paso 1, y eso de los pesos actuales: volver a
ejecutar el script tras actualizar ``config.py`` da pesos algo distintos con
una accuracy similar en test.

Uso:
  uv run python scripts/tune_weights.py
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from entity_linker.config import WEIGHTS
from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.evaluation.datasets import (
    group_by_doc,
    load_disambiguation_sample,
    sample_span,
)
from entity_linker.evaluation.disambiguation import evaluate_disambiguation
from entity_linker.evaluation.metrics import compare
from entity_linker.wikidata import WikidataService

SIGNALS = ["prior", "popularity", "context", "type", "relation"]
L2 = 0.1


@dataclass
class Mention:
    x: np.ndarray  # señales de cada candidato (una fila cada uno)
    gold: int  # posición del QID correcto


async def features(
    rows: list[dict[str, Any]], disambiguator: Disambiguator
) -> list[Mention]:
    """Señales de cada mención, calculadas con el mismo código que el servicio
    (``score_document``), para todas las menciones. Se omiten las que tienen
    el QID correcto fuera de los ``MAX_CANDIDATES`` primeros."""
    mentions = []
    for doc_rows in group_by_doc(rows).values():
        scored = await disambiguator.score_document(
            doc_rows[0]["text"],
            [sample_span(r) for r in doc_rows],
            [r["candidates"] for r in doc_rows],
            all_mentions=True,
        )
        for r, cands, signals in zip(doc_rows, scored.candidates, scored.signals):
            qids = [c.qid for c in cands]
            if r["gold"] not in qids:
                continue
            # prior no está entre las señales: depende solo de la posición
            x = [
                [1 / (rank + 1), *(sig.get(s, 0.0) for s in SIGNALS[1:])]
                for rank, sig in enumerate(signals)
            ]
            mentions.append(Mention(np.array(x), qids.index(r["gold"])))
    return mentions


def fit(mentions: list[Mention]) -> dict[str, float]:
    """Pesos que minimizan -log P(correcto) sumado sobre las menciones."""

    def loss(w: np.ndarray) -> float:
        nll = -sum(log_softmax(m.x @ w)[m.gold] for m in mentions)
        return float(nll + L2 * w @ w)

    bounds = [(0, None)] * len(SIGNALS)  # cada señal es evidencia a favor
    weights = minimize(loss, np.ones(len(SIGNALS)), bounds=bounds).x
    return {s: round(float(w), 2) for s, w in zip(SIGNALS, weights)}


async def main() -> None:
    train = load_disambiguation_sample("validation")
    test = load_disambiguation_sample("test")
    print(f"validation {len(train)} menciones, test {len(test)}")

    wikidata = WikidataService()
    disambiguator = Disambiguator(wikidata, use_llm=False)
    learned = fit(await features(train, disambiguator))

    first = sum(r["gold_rank"] == 1 for r in test) / len(test)
    print(f"{'1er candidato':12} test all {first:.3f}")
    current = dict(WEIGHTS)
    hits = {}
    for name, weights in [("config.py", current), ("aprendidos", learned)]:
        configured_disambiguator = Disambiguator(
            wikidata, use_llm=False, weights=weights
        )
        result = await evaluate_disambiguation(
            test, configured_disambiguator, use_llm=False
        )
        hits[name] = result["hits_by_doc"]
        shown = {s: weights.get(s, 0.0) for s in SIGNALS}
        print(f"{name:12} {shown}")
        print(f"{'':12} test {result['accuracy']} IC 95 % {result['ci95']}")
    await wikidata.aclose()
    # ¿La diferencia es real o ruido? (IC del 95 % por bootstrap pareado)
    print("aprendidos vs config.py:", compare(hits["config.py"], hits["aprendidos"]))


if __name__ == "__main__":
    asyncio.run(main())
