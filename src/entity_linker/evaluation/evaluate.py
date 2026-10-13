"""Run the standard AIDA test evaluation across the main pipeline stages.

Run with ``uv run python -m entity_linker.evaluation.evaluate``. Adjust the
constants below to change the sample size or enable the local LLM.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from entity_linker.config import RESULTS_DIR
from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.evaluation.datasets import (
    limit_rows_to_documents,
    load_disambiguation_sample,
    load_documents,
)
from entity_linker.evaluation.disambiguation import (
    compare_disambiguation,
    evaluate_disambiguation,
)
from entity_linker.evaluation.linking import evaluate_linking
from entity_linker.evaluation.ner import evaluate_ner
from entity_linker.evaluation.reports import without_internal_data
from entity_linker.evaluation.retrieval import evaluate_retrieval
from entity_linker.ner import NERService
from entity_linker.pipeline import EntityLinkingPipeline
from entity_linker.wikidata import WikidataService

__all__ = [
    "evaluate",
    "evaluate_disambiguation",
    "evaluate_linking",
    "evaluate_ner",
    "evaluate_retrieval",
    "main",
]

DOCUMENT_LIMIT = 10
USE_LLM = False
MAX_CANDIDATES = 20
NER_THRESHOLD = 0.5


def _result_path(dataset: str, split: str, use_llm: bool) -> Path:
    suffix = "_llm" if use_llm else ""
    return RESULTS_DIR / f"evaluation{suffix}_{dataset}_{split}.json"


async def evaluate(
    document_limit: int | None = DOCUMENT_LIMIT,
    use_llm: bool = USE_LLM,
) -> dict[str, Any]:
    """Evaluate NER, retrieval, disambiguation, and end-to-end linking.

    The default run is intentionally small and reproducible: the first ten
    documents of the AIDA test split, with the LLM disabled.
    """
    dataset, split = "aida", "test"
    docs = load_documents(dataset, split)
    if document_limit is not None:
        docs = docs[:document_limit]
    if not docs:
        raise ValueError(f"No documents available for {dataset!r} split {split!r}")

    report: dict[str, Any] = {
        "settings": {
            "dataset": dataset,
            "split": split,
            "document_limit": document_limit,
            "use_llm": use_llm,
            "max_candidates": MAX_CANDIDATES,
            "ner_threshold": NER_THRESHOLD,
        },
        "documents": len(docs),
    }
    report["ner"] = evaluate_ner(NERService(threshold=NER_THRESHOLD), docs)

    wikidata = WikidataService()
    try:
        report["retrieval"] = await evaluate_retrieval(docs, wikidata, MAX_CANDIDATES)
        sample_rows = load_disambiguation_sample(split, dataset)
        sample_rows = limit_rows_to_documents(sample_rows, document_limit)
        disambiguator = Disambiguator(wikidata, use_llm=use_llm)
        report["disambiguation"] = await compare_disambiguation(
            sample_rows, disambiguator, use_llm
        )
    finally:
        await wikidata.aclose()

    pipeline = EntityLinkingPipeline(use_llm=use_llm)
    try:
        report["linking"] = await evaluate_linking(docs, pipeline)
    finally:
        await pipeline.wikidata.aclose()
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"{report['settings']['dataset']} [{report['settings']['split']}], "
        f"{report['documents']} documents"
    )
    for stage in ("ner", "retrieval", "disambiguation", "linking"):
        result = report[stage]
        if stage == "disambiguation":  # un resultado por configuración y comparación
            for name, config in result.items():
                print(f"{stage} [{name}]: {config.get('accuracy', config)}")
        else:
            print(f"{stage}: {result['metrics']}")


def main() -> None:
    report = asyncio.run(evaluate())
    _print_summary(report)
    path = _result_path("aida", "test", USE_LLM)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(without_internal_data(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Full report: {path}")


if __name__ == "__main__":
    main()
