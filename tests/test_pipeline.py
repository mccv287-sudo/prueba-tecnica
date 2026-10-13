from unittest.mock import AsyncMock

import pytest

from entity_linker import pipeline as module
from entity_linker.pipeline import EntityLinkingPipeline
from entity_linker.schemas import EntitySpan, WikidataCandidate


@pytest.mark.asyncio
async def test_link_retrieves_repeated_mentions_once(monkeypatch):
    spans = [
        EntitySpan(text="Paris", start_char=0, end_char=5, label="LOC"),
        EntitySpan(text="Paris", start_char=14, end_char=19, label="LOC"),
        EntitySpan(text="France", start_char=24, end_char=30, label="LOC"),
    ]
    paris = [WikidataCandidate(qid="Q90", label="Paris")]
    france = [WikidataCandidate(qid="Q142", label="France")]

    ner = type("FakeNER", (), {"extract_entities": lambda self, text: spans})()
    wikidata = type(
        "FakeWikidata",
        (),
        {"get_candidates": AsyncMock(side_effect=[paris, france])},
    )()
    disambiguator = type(
        "FakeDisambiguator",
        (),
        {"disambiguate_document": AsyncMock(return_value=[])},
    )()
    monkeypatch.setattr(module, "NERService", lambda: ner)
    monkeypatch.setattr(module, "WikidataService", lambda: wikidata)
    monkeypatch.setattr(module, "Disambiguator", lambda *args, **kwargs: disambiguator)

    pipeline = EntityLinkingPipeline()
    await pipeline.link("Paris was lovely. Paris is in France.", language="en")

    assert [call.args[0] for call in wikidata.get_candidates.await_args_list] == [
        "Paris",
        "France",
    ]
    disambiguator.disambiguate_document.assert_awaited_once_with(
        "Paris was lovely. Paris is in France.",
        spans,
        [paris, paris, france],
        use_llm=None,
    )
