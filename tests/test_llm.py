from unittest.mock import AsyncMock

import pytest

from entity_linker.disambiguation.llm import LLMChooser
from entity_linker.schemas import EntitySpan, WikidataCandidate

SPAN = EntitySpan(
    text="Paris",
    start_char=10,
    end_char=15,
    label="LOC",
    fine_label="city",
)
CANDIDATES = [
    WikidataCandidate(qid="Q90", label="Paris", description="capital of France"),
    WikidataCandidate(qid="Q1612", label="Paris", description="city in Texas"),
]


async def test_chooser_maps_ranked_response_to_original_candidate():
    chooser = LLMChooser()
    chooser._ask = AsyncMock(return_value=1)

    chosen = await chooser.choose_candidate(
        "Text mentions Paris and France.",
        SPAN,
        CANDIDATES,
        [0.2, 0.8],
        ["France"],
    )

    assert chosen == 1
    chooser._ask.assert_awaited_once()
    await_args = chooser._ask.await_args
    assert await_args is not None
    prompt = await_args.args[0]
    assert "1. Paris: city in Texas" in prompt
    assert "2. Paris: capital of France" in prompt
    assert "Other entities in the document: France" in prompt
    assert "(type: city)" in prompt


async def test_chooser_returns_none_when_no_candidate_fits():
    chooser = LLMChooser()
    chooser._ask = AsyncMock(return_value=0)

    chosen = await chooser.choose_candidate(
        "Text mentions Paris.", SPAN, CANDIDATES, [0.7, 0.3], []
    )

    assert chosen is None


async def test_chooser_rejects_out_of_range_response():
    chooser = LLMChooser()
    chooser._ask = AsyncMock(return_value=3)

    with pytest.raises(ValueError, match="out of range"):
        await chooser.choose_candidate(
            "Text mentions Paris.", SPAN, CANDIDATES, [0.7, 0.3], []
        )
