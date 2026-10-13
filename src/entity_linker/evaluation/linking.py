"""End-to-end entity-linking evaluation."""

from time import perf_counter
from typing import Any, Protocol

from entity_linker.evaluation.datasets import Document
from entity_linker.evaluation.metrics import linking_metrics
from entity_linker.evaluation.reports import Spans, span_report


class LinkingPipeline(Protocol):
    disambiguator: Any

    async def link(self, text: str) -> list[Any]: ...


async def evaluate_linking(
    docs: list[Document], pipeline: LinkingPipeline
) -> dict[str, Any]:
    """Measure the complete NER, retrieval, and disambiguation pipeline."""
    calls_before = pipeline.disambiguator.llm_calls
    started_at = perf_counter()
    gold: Spans = [
        {(mention.start, mention.end, mention.qid or "NIL") for mention in doc.mentions}
        for doc in docs
    ]
    predicted: Spans = []
    for doc in docs:
        entities = await pipeline.link(doc.text)
        predicted.append(
            {(entity.start_char, entity.end_char, entity.qid) for entity in entities}
        )
    report = span_report(
        docs, gold, predicted, linking_metrics(gold, predicted), started_at
    )
    report["llm_calls"] = pipeline.disambiguator.llm_calls - calls_before
    return report
