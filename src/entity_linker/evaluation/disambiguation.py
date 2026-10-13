"""Disambiguation evaluation and configuration comparisons."""

from itertools import pairwise
from time import perf_counter
from typing import Any

from entity_linker.config import REVIEW_THRESHOLD
from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.evaluation.datasets import group_by_doc, sample_span
from entity_linker.evaluation.metrics import Hits, accuracy_ci, compare


def rank_bucket(gold_rank: int) -> str:
    return "rank 1" if gold_rank == 1 else "rank 2-10" if gold_rank <= 10 else ">10"


async def evaluate_disambiguation(
    rows: list[dict[str, Any]],
    disambiguator: Disambiguator,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Measure accuracy by candidate rank and review-confidence bucket."""
    if not rows:
        raise ValueError("Disambiguation evaluation requires at least one mention")

    calls_before = disambiguator.llm_calls
    buckets: dict[str, list[bool]] = {}
    hits_by_doc: Hits = {}
    started_at = perf_counter()
    for doc_id, doc_rows in group_by_doc(rows).items():
        linked = await disambiguator.disambiguate_document(
            doc_rows[0]["text"],
            [sample_span(row) for row in doc_rows],
            [row["candidates"] for row in doc_rows],
            use_llm=use_llm,
        )
        doc_hits = [entity.qid == row["gold"] for row, entity in zip(doc_rows, linked)]
        hits_by_doc[doc_id] = doc_hits
        for row, entity, hit in zip(doc_rows, linked, doc_hits):
            buckets.setdefault(rank_bucket(row["gold_rank"]), []).append(hit)
            bucket = (
                "uncertain" if entity.review_needed(REVIEW_THRESHOLD) else "confident"
            )
            buckets.setdefault(bucket, []).append(hit)

    all_hits = [hit for doc_hits in hits_by_doc.values() for hit in doc_hits]
    return {
        "accuracy": {
            "all": round(sum(all_hits) / len(all_hits), 3),
            **{
                bucket: round(sum(hits) / len(hits), 3)
                for bucket, hits in sorted(buckets.items())
            },
        },
        "ci95": accuracy_ci(hits_by_doc),
        "mentions": {bucket: len(hits) for bucket, hits in sorted(buckets.items())},
        "llm_calls": disambiguator.llm_calls - calls_before,
        "seconds": round(perf_counter() - started_at, 1),
        "hits_by_doc": hits_by_doc,
    }


async def compare_disambiguation(
    rows: list[dict[str, Any]], disambiguator: Disambiguator, use_llm: bool = False
) -> dict[str, Any]:
    """Compare first candidate, signal ranking, and optionally LLM decisions."""
    if not rows:
        raise ValueError("Disambiguation comparison requires at least one mention")

    first: Hits = {
        doc_id: [row["gold_rank"] == 1 for row in doc_rows]
        for doc_id, doc_rows in group_by_doc(rows).items()
    }
    first_all = sum(hit for hits in first.values() for hit in hits) / len(rows)
    report: dict[str, Any] = {
        "1er candidato": {
            "accuracy": {"all": round(first_all, 3)},
            "ci95": accuracy_ci(first),
            "hits_by_doc": first,
        }
    }
    configs = [("sin LLM", False)] + ([("con LLM", True)] if use_llm else [])
    for name, call_llm in configs:
        report[name] = await evaluate_disambiguation(rows, disambiguator, call_llm)

    for before, after in pairwise(list(report)):
        report[f"{after} vs {before}"] = compare(
            report[before]["hits_by_doc"], report[after]["hits_by_doc"]
        )
    return report
