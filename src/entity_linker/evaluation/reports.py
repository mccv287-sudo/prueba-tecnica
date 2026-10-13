"""Shared report helpers for stage evaluators."""

from time import perf_counter
from typing import Any

from entity_linker.evaluation.datasets import Document

Span = tuple[int, int, str]
Spans = list[set[Span]]


def span_report(
    docs: list[Document],
    gold: Spans,
    predicted: Spans,
    metrics: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    """Return metrics, average document latency, and span-level errors."""
    errors = [
        {
            "doc_id": doc.doc_id,
            "missed": describe_spans(doc.text, gold_spans - predicted_spans),
            "spurious": describe_spans(doc.text, predicted_spans - gold_spans),
        }
        for doc, gold_spans, predicted_spans in zip(docs, gold, predicted)
        if gold_spans != predicted_spans
    ]
    elapsed_ms = (perf_counter() - started_at) * 1000
    ms_per_doc = round(elapsed_ms / len(docs), 1) if docs else 0.0
    return {"metrics": metrics, "ms_per_doc": ms_per_doc, "errors": errors}


def describe_spans(text: str, spans: set[Span]) -> list[str]:
    return [f"{text[start:end]} [{label}]" for start, end, label in sorted(spans)]


def without_internal_data(value: Any) -> Any:
    """Remove evaluator-only data before writing a user-facing report."""
    if isinstance(value, dict):
        return {
            key: without_internal_data(item)
            for key, item in value.items()
            if key != "hits_by_doc"
        }
    if isinstance(value, list):
        return [without_internal_data(item) for item in value]
    return value
