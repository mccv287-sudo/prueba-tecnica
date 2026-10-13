"""NER-stage evaluation."""

from time import perf_counter
from typing import Any, Protocol

from entity_linker.evaluation.datasets import Document
from entity_linker.evaluation.metrics import ner_metrics
from entity_linker.evaluation.reports import Spans, span_report


class NER(Protocol):
    def extract_entities(
        self, text: str, labels: dict[str, str] | None = None
    ) -> list[Any]: ...


def evaluate_ner(
    ner: NER,
    docs: list[Document],
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure span and type quality for an NER implementation."""
    started_at = perf_counter()
    gold: Spans = [
        {
            (mention.start, mention.end, mention.label)
            for mention in doc.mentions
            if mention.label
        }
        for doc in docs
    ]
    predicted: Spans = [
        {
            (entity.start_char, entity.end_char, entity.label)
            for entity in ner.extract_entities(doc.text, labels)
        }
        for doc in docs
    ]
    return span_report(docs, gold, predicted, ner_metrics(gold, predicted), started_at)
