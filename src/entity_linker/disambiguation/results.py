"""Build user-facing linked entities and explanations from decisions."""

from collections.abc import Mapping
from typing import Literal

from entity_linker.disambiguation.signals import argmax, contributions
from entity_linker.schemas import EntitySpan, LinkedEntity, WikidataCandidate

LLMOutcome = Literal["confirma", "elige", "nil"] | None


def explain_choice(
    signals: list[dict[str, float]],
    probabilities: list[float],
    candidates: list[WikidataCandidate],
    weights: Mapping[str, float],
) -> str:
    """Summarize the strongest positive signal differences for the top pair."""
    if len(candidates) < 2:
        return ""
    best, second = sorted(
        range(len(probabilities)), key=probabilities.__getitem__, reverse=True
    )[:2]
    best_contributions = contributions(signals[best], best, weights)
    second_contributions = contributions(signals[second], second, weights)
    differences = sorted(
        (
            (
                name,
                best_contributions.get(name, 0.0) - second_contributions.get(name, 0.0),
            )
            for name in best_contributions.keys() | second_contributions.keys()
        ),
        key=lambda item: -item[1],
    )
    strongest = [
        f"{name} {difference:+.1f}"
        for name, difference in differences[:2]
        if difference > 0.05
    ]
    if not strongest:
        return ""
    return f"; frente a {_candidate_name(candidates[second])}: {', '.join(strongest)}"


def build_linked_entity(
    span: EntitySpan,
    candidates: list[WikidataCandidate],
    probabilities: list[float],
    chosen: int | None,
    llm_outcome: LLMOutcome,
    explanation: str,
) -> LinkedEntity:
    """Assemble the API/domain result for a chosen candidate or NIL.

    ``confidence`` is the softmax probability of the chosen candidate: a
    ranking score relative to the retrieved candidates, not a calibrated
    probability that the link is correct.
    """
    selected = candidates[chosen] if chosen is not None else None
    p_chosen = round(probabilities[chosen], 3) if chosen is not None else 0.0
    confidence = p_chosen
    top = candidates[argmax(probabilities)] if candidates else None

    if top is None:
        reasoning = "Sin candidatos en Wikidata"
    elif llm_outcome == "nil":
        reasoning = (
            f"LLM: ningún candidato encaja (sugerido: {_candidate_name(top)}, "
            f"p={max(probabilities):.2f})"
        )
    elif llm_outcome == "elige":
        assert selected is not None
        reasoning = (
            f"LLM elige {_candidate_name(selected)}, p={p_chosen:.2f}, frente "
            f"al de las señales, {_candidate_name(top)}, "
            f"p={max(probabilities):.2f}: desacuerdo, para revisión"
        )
    elif llm_outcome == "confirma":
        reasoning = f"LLM confirma a las señales: p={p_chosen:.2f}{explanation}"
    else:
        reasoning = f"p={p_chosen:.2f} sobre {len(candidates)} candidatos{explanation}"

    return LinkedEntity(
        **span.model_dump(),
        qid=selected.qid if selected else "NIL",
        wikidata_label=selected.label if selected else None,
        wikidata_description=selected.description if selected else None,
        confidence=confidence,
        is_nil=selected is None,
        reasoning=reasoning,
        llm_confirmed=None if llm_outcome is None else llm_outcome == "confirma",
        candidates=[
            candidate
            for _, candidate in sorted(
                zip(probabilities, candidates), key=lambda item: -item[0]
            )
        ],
    )


def _candidate_name(candidate: WikidataCandidate) -> str:
    return f"{candidate.qid} ({candidate.label})"
