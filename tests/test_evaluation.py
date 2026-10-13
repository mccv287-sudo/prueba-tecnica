import pytest

from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.evaluation.datasets import (
    limit_rows_to_documents,
    load_documents,
    load_wikianc,
)
from entity_linker.evaluation.disambiguation import evaluate_disambiguation
from entity_linker.evaluation.reports import without_internal_data
from entity_linker.schemas import WikidataCandidate


def sample_row(doc_id: str, start: int, mention: str, qid: str) -> dict:
    return {
        "doc_id": doc_id,
        "text": f"{mention} appeared in this article.",
        "start": start,
        "end": start + len(mention),
        "mention": mention,
        "label": "PER",
        "fine_label": "person",
        "gold": qid,
        "gold_rank": 1,
        "candidates": [WikidataCandidate(qid=qid, label=mention)],
    }


def test_disambiguation_limit_keeps_whole_documents():
    rows = [
        sample_row("doc-a", 0, "Alice", "Q1"),
        sample_row("doc-a", 0, "Alice", "Q1"),
        sample_row("doc-b", 0, "Bob", "Q2"),
    ]

    limited = limit_rows_to_documents(rows, 1)

    assert len(limited) == 2
    assert {row["doc_id"] for row in limited} == {"doc-a"}


def test_report_serialization_removes_nested_internal_hits():
    report = {
        "metrics": {"accuracy": 1.0},
        "hits_by_doc": {"doc-a": [True]},
        "comparison": {"hits_by_doc": {"doc-b": [False]}, "diff": -1.0},
    }

    assert without_internal_data(report) == {
        "metrics": {"accuracy": 1.0},
        "comparison": {"diff": -1.0},
    }


def test_document_loader_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_documents("not-a-dataset")


def test_aida_loader_rejects_unknown_split():
    with pytest.raises(ValueError, match="Unsupported AIDA split"):
        load_documents("aida", "development")


def test_document_loader_rejects_unsupported_wikianc_split():
    with pytest.raises(ValueError, match="only for validation"):
        load_documents("wikianc", "test")


def test_wikianc_sampler_handles_small_corpus_and_uses_unique_offsets(
    monkeypatch, tmp_path
):
    from entity_linker.evaluation import datasets

    monkeypatch.setattr(datasets, "PAGE_SIZE", 2)
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    fetched_offsets = []

    def fake_fetch(client, dataset, config, split, offset, length):
        fetched_offsets.append(offset)
        if offset == 0 and length == 1:
            return {"num_rows_total": 3}
        return {
            "rows": [
                {
                    "row": {
                        "article_title": f"Article {offset}",
                        "paragraph_text": (
                            "Alice wrote a book in London. ".join(
                                ["This paragraph contains enough text. "] * 12
                            )
                        ),
                        "paragraph_anchors": [{"start": 0, "end": 5, "qid": "Q1"}],
                    }
                }
            ]
        }

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(datasets, "_fetch", fake_fetch)
    monkeypatch.setattr(datasets.httpx, "Client", FakeClient)

    docs = load_wikianc(n_docs=2, seed=0)

    assert len(docs) == 2
    assert len(set(fetched_offsets[1:])) == len(fetched_offsets[1:])


def test_disambiguation_document_limit_preserves_document_order():
    rows = [
        sample_row("doc-b", 0, "Bob", "Q2"),
        sample_row("doc-a", 0, "Alice", "Q1"),
        sample_row("doc-b", 0, "Bob", "Q2"),
    ]

    limited = limit_rows_to_documents(rows, 1)

    assert len(limited) == 2
    assert {row["doc_id"] for row in limited} == {"doc-b"}


async def test_disambiguation_evaluation_groups_hits_by_document():
    rows = [
        sample_row("doc-a", 0, "Alice", "Q1"),
        sample_row("doc-a", 0, "Alice", "Q1"),
    ]

    report = await evaluate_disambiguation(
        rows, Disambiguator(use_llm=False), use_llm=False
    )

    assert report["accuracy"]["all"] == 1.0
    assert report["hits_by_doc"] == {"doc-a": [True, True]}


async def test_disambiguation_evaluation_rejects_empty_sample():
    with pytest.raises(ValueError, match="at least one mention"):
        await evaluate_disambiguation([], Disambiguator(use_llm=False))


async def test_standard_runner_evaluates_each_stage(monkeypatch):
    from entity_linker.evaluation import evaluate as runner

    doc = type("Doc", (), {"doc_id": "doc-a", "text": "Alice", "mentions": []})()

    class FakeWikidata:
        async def aclose(self):
            return None

    class FakePipeline:
        def __init__(self, use_llm):
            self.wikidata = FakeWikidata()

    monkeypatch.setattr(runner, "load_documents", lambda dataset, split: [doc])
    monkeypatch.setattr(runner, "NERService", lambda threshold: object())
    monkeypatch.setattr(runner, "evaluate_ner", lambda ner, docs: {"metrics": {}})
    monkeypatch.setattr(runner, "WikidataService", FakeWikidata)
    monkeypatch.setattr(
        runner,
        "evaluate_retrieval",
        lambda *args: _async_result({"metrics": {}}),
    )
    monkeypatch.setattr(runner, "load_disambiguation_sample", lambda split, dataset: [])
    monkeypatch.setattr(runner, "Disambiguator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "compare_disambiguation",
        lambda *args: _async_result({"accuracy": {"all": 1.0}}),
    )
    monkeypatch.setattr(runner, "EntityLinkingPipeline", FakePipeline)
    monkeypatch.setattr(
        runner,
        "evaluate_linking",
        lambda *args: _async_result({"metrics": {}}),
    )

    report = await runner.evaluate(document_limit=1, use_llm=False)

    assert set(report) == {
        "settings",
        "documents",
        "ner",
        "retrieval",
        "disambiguation",
        "linking",
    }
    assert report["documents"] == 1


async def _async_result(value):
    return value
