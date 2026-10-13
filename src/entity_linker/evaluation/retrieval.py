"""Candidate-retrieval evaluation."""

import asyncio
from typing import Any

from entity_linker.evaluation.datasets import Document
from entity_linker.evaluation.metrics import retrieval_metrics
from entity_linker.wikidata import WikidataService


async def evaluate_retrieval(
    docs: list[Document], wikidata: WikidataService, limit: int
) -> dict[str, Any]:
    """Measure whether the gold QID appears in candidates from Wikidata."""
    mentions = [
        (doc.text[mention.start : mention.end], mention.qid)
        for doc in docs
        for mention in doc.mentions
        if mention.qid
    ]
    unique_texts = list(dict.fromkeys(text for text, _ in mentions))
    results = await asyncio.gather(
        *(wikidata.get_candidates(text, limit=limit) for text in unique_texts)
    )
    found = {
        text: [candidate.qid for candidate in candidates]
        for text, candidates in zip(unique_texts, results)
    }
    candidates_by_mention = [found[text] for text, _ in mentions]
    return {
        "metrics": retrieval_metrics(
            [qid for _, qid in mentions], candidates_by_mention
        ),
        "errors": [
            {"mention": text, "gold": qid, "candidates": candidates[:5]}
            for (text, qid), candidates in zip(mentions, candidates_by_mention)
            if qid not in candidates
        ],
    }
