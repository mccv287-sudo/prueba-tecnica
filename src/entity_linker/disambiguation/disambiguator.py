"""Desambiguación probabilística + LLM solo en los casos inciertos.

Las señales y la probabilidad de cada candidato están en ``signals``. Flujo por
documento, con carga diferida (lo caro solo cuando lo barato no basta):

1. Señales baratas: prior, popularity y context. Si la probabilidad (softmax)
   del mejor candidato alcanza ``CONFIDENT_PROB``, se acepta.
2. Solo las menciones inciertas piden a Wikidata las clases de sus candidatos
   y sus relaciones con las ya resueltas, y suman type y relation.
3. Si siguen inciertas y ``use_llm``, el LLM elige entre los candidatos
   (elección múltiple) y **su elección es la respuesta**, sin peso ni señal. La
   confianza es la que las señales daban a ese candidato. Si coincide con el
   mejor de las señales, se acepta sin revisión (``llm_confirmed``); si elige
   otro, queda para revisión; si responde que ninguno encaja, NIL con el más
   probable como sugerencia.

La confianza devuelta es la probabilidad softmax del candidato elegido: una
puntuación de *ranking*, relativa a los candidatos recuperados, no la
probabilidad de que el enlace sea correcto (con un único candidato vale 1).
La marca de revisión (``needs_review``) la pone el pipeline, con el umbral de
cada petición.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from entity_linker.config import (
    CONFIDENT_PROB,
    LLM_MODEL,
    LLM_TIMEOUT,
    MAX_CANDIDATES,
    WEIGHTS,
)
from entity_linker.disambiguation.llm import LLMChooser
from entity_linker.disambiguation.results import (
    LLMOutcome,
    build_linked_entity,
    explain_choice,
)
from entity_linker.disambiguation.signals import (
    ROOT_CLASSES,
    argmax,
    context_score,
    popularity_scores,
    probabilities,
    relation_score,
    surroundings,
    type_score,
)
from entity_linker.schemas import EntitySpan, LinkedEntity, WikidataCandidate
from entity_linker.wikidata import WikidataService

logger = logging.getLogger(__name__)

Signals = list[list[dict[str, float]]]  # mención -> candidato -> señal -> valor


@dataclass
class ScoredDocument:
    """Resultado de los pasos 1 y 2 para las menciones de un documento."""

    candidates: list[list[WikidataCandidate]]  # como mucho MAX_CANDIDATES
    signals: Signals
    probs: list[list[float]]
    uncertain: list[bool]  # inciertas tras las señales baratas
    resolved: dict[int, WikidataCandidate]  # aceptadas en el paso 1, por mención


class Disambiguator:
    def __init__(
        self,
        wikidata: WikidataService | None = None,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT,
        use_llm: bool = True,
        weights: Mapping[str, float] | None = None,
        llm_chooser: LLMChooser | None = None,
    ):
        """
        Args:
            wikidata: Para pedir los sitelinks, las clases y las relaciones de
                los candidatos (None: solo prior y context).
            model: Modelo de Ollama.
            timeout: Segundos por llamada al LLM.
            use_llm: Consultar al LLM en las menciones que siguen inciertas.
        """
        self.wikidata = wikidata
        self.model = model
        self.timeout = timeout
        self.use_llm = use_llm
        self.weights = dict(WEIGHTS if weights is None else weights)
        self.llm_chooser = llm_chooser or LLMChooser(model, timeout)

    @property
    def llm_calls(self) -> int:
        return self.llm_chooser.calls

    async def disambiguate_document(
        self,
        text: str,
        spans: list[EntitySpan],
        candidates_list: list[list[WikidataCandidate]],
        use_llm: bool | None = None,
    ) -> list[LinkedEntity]:
        """Elige el QID de cada mención del documento, o NIL.

        ``use_llm`` anula para esta llamada la opción del servicio.
        """
        use_llm = self.use_llm if use_llm is None else use_llm
        scored = await self.score_document(text, spans, candidates_list)
        names = list(dict.fromkeys(c.label for c in scored.resolved.values()))

        results = []
        for i, span in enumerate(spans):
            cands, probs = scored.candidates[i], scored.probs[i]
            chosen = argmax(probs) if cands else None
            llm: LLMOutcome = None
            # 3. LLM solo si sigue incierta: su elección es la respuesta
            if use_llm and cands and max(probs) < CONFIDENT_PROB:
                succeeded, answer = await self._choose_with_llm(
                    text, span, cands, probs, names
                )
                if succeeded and answer is None:
                    chosen, llm = None, "nil"
                elif succeeded:
                    llm = "confirma" if answer == chosen else "elige"
                    chosen = answer
            explanation = explain_choice(scored.signals[i], probs, cands, self.weights)
            results.append(
                build_linked_entity(span, cands, probs, chosen, llm, explanation)
            )
        return results

    async def score_document(
        self,
        text: str,
        spans: list[EntitySpan],
        candidates_list: list[list[WikidataCandidate]],
        all_mentions: bool = False,
    ) -> ScoredDocument:
        """Pasos 1 y 2: señales y probabilidad de cada candidato.

        Con ``all_mentions`` (para aprender los pesos, ``tune_weights.py``),
        type y relation se calculan para todas las menciones y no
        solo para las inciertas.
        """
        candidates_list = [c[:MAX_CANDIDATES] for c in candidates_list]

        # 1. Señales baratas; se aceptan las menciones con un candidato claro
        signals = await self._cheap_signals(text, spans, candidates_list)
        probs = [probabilities(sig, self.weights) for sig in signals]
        uncertain = [bool(p) and max(p) < CONFIDENT_PROB for p in probs]
        resolved = {
            i: cands[argmax(p)]
            for i, (cands, p, u) in enumerate(zip(candidates_list, probs, uncertain))
            if p and not u
        }

        # 2. Señales de Wikidata, solo para las inciertas
        targets = [
            i
            for i, cands in enumerate(candidates_list)
            if cands and (all_mentions or uncertain[i])
        ]
        if self.wikidata and targets:
            await self._add_wikidata_signals(
                spans, candidates_list, signals, resolved, targets
            )
            probs = [probabilities(sig, self.weights) for sig in signals]
        return ScoredDocument(candidates_list, signals, probs, uncertain, resolved)

    async def _cheap_signals(
        self,
        text: str,
        spans: list[EntitySpan],
        candidates_list: list[list[WikidataCandidate]],
    ) -> Signals:
        """Paso 1: popularity (relativa a los demás candidatos; una consulta
        por cada 50 QIDs, con caché) y context (sin la propia mención). prior
        no se guarda: depende solo de la posición."""
        sitelinks = (
            await self.wikidata.get_sitelinks(
                [c.qid for cands in candidates_list for c in cands]
            )
            if self.wikidata
            else {}
        )
        signals = []
        for span, cands in zip(spans, candidates_list):
            context = surroundings(text, span.start_char, span.end_char)
            signals.append(
                [
                    {"context": context_score(c, context, span.text), "popularity": p}
                    for c, p in zip(cands, popularity_scores(cands, sitelinks))
                ]
            )
        return signals

    async def _add_wikidata_signals(
        self,
        spans: list[EntitySpan],
        candidates_list: list[list[WikidataCandidate]],
        signals: Signals,
        resolved: dict[int, WikidataCandidate],
        targets: list[int],
    ) -> None:
        """Paso 2: añade type y relation a las menciones ``targets`` (las
        clases de sus candidatos y sus relaciones con las resueltas, en una
        consulta SPARQL por cada 50 QIDs)."""
        assert self.wikidata
        qids = [c.qid for i in targets for c in candidates_list[i]]
        known = [c.qid for c in resolved.values()]
        classes, links = await asyncio.gather(
            self.wikidata.get_classes(qids, ROOT_CLASSES),
            self.wikidata.get_links(qids, known),
        )
        for i in targets:
            # Las demás resueltas (una mención incierta no está entre ellas)
            others = [c.qid for j, c in resolved.items() if j != i]
            for sig, cand in zip(signals[i], candidates_list[i]):
                sig["type"] = type_score(spans[i].fine_label, cand.qid, classes)
                sig["relation"] = relation_score(cand.qid, others, links)

    async def _choose_with_llm(
        self,
        text: str,
        span: EntitySpan,
        cands: list[WikidataCandidate],
        probs: list[float],
        resolved_names: list[str],
    ) -> tuple[bool, int | None]:
        """Delegate the uncertain mention to the LLM chooser.

        The boolean indicates a valid response. A successful ``None`` means
        the LLM says no candidate fits; failures preserve the signal choice.
        """
        try:
            choice = await self.llm_chooser.choose_candidate(
                text, span, cands, probs, resolved_names
            )
            return True, choice
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            logger.warning("LLM falló para %r: %s", span.text, exc)
            return False, None
