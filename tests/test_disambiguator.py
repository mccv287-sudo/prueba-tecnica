"""Desambiguador sin red: sin cliente de Wikidata (sin type ni relation) y
con el LLM simulado."""

from unittest.mock import AsyncMock

import pytest

from entity_linker.config import REVIEW_THRESHOLD
from entity_linker.disambiguation import disambiguator as module
from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.schemas import EntitySpan, LinkedEntity, WikidataCandidate

TEXT = "Kuwait national football team beat Japan 2-0 in the Asian Cup."
SPAN = EntitySpan(text="Kuwait", start_char=0, end_char=6, label="LOC")


def candidate(qid: str, description: str | None = None) -> WikidataCandidate:
    return WikidataCandidate(qid=qid, label="Kuwait", description=description)


COUNTRY = candidate("Q817", "country in Western Asia")
TEAM = candidate("Q195851", "national association football team")
CANDIDATES = [COUNTRY, candidate("Q2")]


async def link_one(
    dis: Disambiguator, text: str, cands: list[WikidataCandidate]
) -> LinkedEntity:
    """Desambigua ``SPAN`` como única mención del documento."""
    return (await dis.disambiguate_document(text, [SPAN], [cands]))[0]


def needs_review(entity: LinkedEntity) -> bool:
    return entity.review_needed(REVIEW_THRESHOLD)


async def test_context_overrides_search_order():
    linked = await link_one(Disambiguator(use_llm=False), TEXT, [COUNTRY, TEAM])
    assert linked.qid == TEAM.qid
    assert linked.candidates[0].qid == TEAM.qid  # ordenados por probabilidad
    assert linked.llm_confirmed is None  # no se consultó al LLM


async def test_no_candidates_is_nil_and_needs_review():
    linked = await link_one(Disambiguator(use_llm=False), TEXT, [])
    assert linked.is_nil and linked.qid == "NIL"
    assert needs_review(linked)


@pytest.fixture
def always_uncertain(monkeypatch):
    """Todas las menciones inciertas, para que siempre se consulte al LLM."""
    monkeypatch.setattr(module, "CONFIDENT_PROB", 1.01)


async def signals_only_confidence() -> float:
    return (
        await link_one(Disambiguator(use_llm=False), "Kuwait", CANDIDATES)
    ).confidence


def with_llm_answer(**mock) -> Disambiguator:
    """Desambiguador con el LLM simulado (``return_value`` o ``side_effect``)."""
    dis = Disambiguator()
    dis.llm_chooser.choose_candidate = AsyncMock(**mock)
    return dis


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_choice_wins_but_disagreement_is_reviewed():
    linked = await link_one(with_llm_answer(return_value=1), "Kuwait", CANDIDATES)
    # Su elección (el 2.º más probable) es la respuesta, con la confianza que le
    # daban las señales (menor que la del mejor) y, por el desacuerdo, para
    # revisión
    assert linked.qid == "Q2"
    assert linked.confidence < await signals_only_confidence()
    assert linked.llm_confirmed is False and needs_review(linked)


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_agreeing_with_signals_is_accepted_without_review():
    linked = await link_one(with_llm_answer(return_value=0), "Kuwait", CANDIDATES)
    assert linked.qid == COUNTRY.qid
    assert linked.confidence == await signals_only_confidence()  # no la infla
    assert linked.llm_confirmed is True and not needs_review(linked)


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_none_fits_is_nil():
    linked = await link_one(with_llm_answer(return_value=None), TEXT, [COUNTRY, TEAM])
    assert linked.is_nil and needs_review(linked)
    # El más probable queda como sugerido y primero en los candidatos
    assert linked.reasoning and TEAM.qid in linked.reasoning
    assert linked.candidates[0].qid == TEAM.qid


@pytest.mark.usefixtures("always_uncertain")
async def test_llm_failure_keeps_most_probable():
    dis = with_llm_answer(side_effect=ValueError("respuesta no JSON"))
    linked = await link_one(dis, TEXT, [COUNTRY, TEAM])
    assert linked.qid == TEAM.qid
