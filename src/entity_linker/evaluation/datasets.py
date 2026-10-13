"""Datasets anotados (inglés) con QIDs de Wikidata, descargados de Hugging Face.

- AIDA CoNLL-YAGO: noticias de Reuters con *todas* las entidades anotadas
  (PER, ORG, LOC, MISC) y su QID (None si no está en Wikidata). Sirve para NER
  y retrieval. https://huggingface.co/datasets/cyanic-selkie/aida-conll-yago-wikidata
- WikiAnc: párrafos de Wikipedia con sus enlaces internos y el QID al que
  apuntan. Solo sirve para retrieval: Wikipedia no enlaza todas las entidades
  y no hay etiqueta de tipo. https://huggingface.co/datasets/cyanic-selkie/wikianc
- Benchmarks fuera de AIDA (``BENCHMARKS``: KORE50, MSNBC, AQUAINT, ACE2004),
  publicados por naist-nlp: textos de otras fuentes, etiquetados con páginas
  de Wikipedia que se convierten a QIDs. Solo para evaluar.
- Muestras de desambiguación: menciones con NER correcto y QID entre los
  candidatos, generadas con ``scripts/build_disambiguation_sample.py`` (AIDA
  validation para aprender, AIDA test y los benchmarks para medir).

Cada dataset se descarga una vez y se guarda en ``data/``.
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from entity_linker.config import DATA_DIR, USER_AGENT, WIKIPEDIA_API_URL
from entity_linker.schemas import EntitySpan, WikidataCandidate

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # máximo que permite la API de Hugging Face
DOCS_PER_BLOCK = 10
# Benchmarks de evaluación fuera de AIDA (solo test): ambigüedad difícil
# (KORE50) y noticias de otras fuentes (MSNBC, AQUAINT, ACE2004)
BENCHMARKS = ("kore50", "msnbc", "aquaint", "ace2004")
NIL_PAGE = "-1"  # etiqueta de NIL en esos benchmarks


@dataclass(frozen=True)
class GoldMention:
    start: int
    end: int
    label: str | None  # PER, ORG, LOC, MISC; None si el dataset no lo anota
    qid: str | None  # None: NIL o sin enlace


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    mentions: list[GoldMention]


def load_documents(dataset: str, split: str = "test") -> list[Document]:
    """Load a supported annotated dataset and split."""
    if dataset == "aida":
        return load_aida(split)
    if dataset == "wikianc":
        if split != "validation":
            raise ValueError("WikiAnc is available here only for validation")
        return load_wikianc(split=split)
    if dataset in BENCHMARKS:
        if split != "test":
            raise ValueError(f"Benchmark {dataset!r} is available only for test")
        return load_benchmark(dataset)
    raise ValueError(f"Unknown dataset: {dataset!r}")


def load_aida(split: str = "test") -> list[Document]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported AIDA split: {split!r}")
    path = DATA_DIR / f"aida_{split}.jsonl"
    if not path.exists():
        dataset = ("cyanic-selkie/aida-conll-yago-wikidata", "default", split)
        rows: list[dict[str, Any]] = []
        with httpx.Client(timeout=60) as client:
            while True:
                page = _fetch(client, *dataset, offset=len(rows), length=PAGE_SIZE)
                rows += [item["row"] for item in page["rows"]]
                if len(rows) >= page["num_rows_total"]:
                    break
        _write(path, [_aida_document(row) for row in rows])
    return _read(path)


def load_wikianc(
    n_docs: int = 50, split: str = "validation", seed: int = 42
) -> list[Document]:
    """Muestra aleatoria (reproducible por ``seed``) de párrafos de Wikipedia.

    Se leen bloques de filas en posiciones aleatorias y de cada uno se toman
    hasta ``DOCS_PER_BLOCK`` párrafos de artículos distintos. Solo se
    conservan los enlaces a una entidad con QID cuyo texto empieza por
    mayúscula, como aproximación a "entidad nombrada" (WikiAnc también enlaza
    conceptos comunes como "software").
    """
    if n_docs < 1:
        raise ValueError("n_docs must be at least 1")
    if split != "validation":
        raise ValueError("WikiAnc is available here only for validation")

    path = DATA_DIR / f"wikianc_en_{split}_n{n_docs}_seed{seed}.jsonl"
    if not path.exists():
        dataset = ("cyanic-selkie/wikianc", "en", split)
        rng = random.Random(seed)
        docs: list[Document] = []
        with httpx.Client(timeout=120) as client:
            total = _fetch(client, *dataset, offset=0, length=1)["num_rows_total"]
            offsets = list(range(0, total, PAGE_SIZE))
            rng.shuffle(offsets)
            seen_articles: set[str] = set()
            for offset in offsets:
                length = min(PAGE_SIZE, total - offset)
                page = _fetch(client, *dataset, offset=offset, length=length)
                block = []
                for item in page["rows"]:
                    doc = _wikianc_document(item)
                    if doc is not None and doc.doc_id not in seen_articles:
                        seen_articles.add(doc.doc_id)
                        block.append(doc)
                docs.extend(block[:DOCS_PER_BLOCK])
                if len(docs) >= n_docs:
                    break
        if len(docs) < n_docs:
            raise ValueError(
                f"WikiAnc contains only {len(docs)} eligible unique documents; "
                f"requested {n_docs}"
            )
        _write(path, docs[:n_docs])
    docs = _read(path)
    if len(docs) < n_docs:
        raise ValueError(
            f"Cached WikiAnc sample has {len(docs)} documents, requested {n_docs}"
        )
    return docs[:n_docs]


def disambiguation_sample_path(split: str = "test", dataset: str = "aida") -> Path:
    return DATA_DIR / f"disambiguation_{dataset}_{split}.jsonl"


def load_disambiguation_sample(
    split: str = "test", dataset: str = "aida"
) -> list[dict[str, Any]]:
    """Menciones con NER correcto y QID entre los candidatos, con sus candidatos
    de Wikidata guardados (``scripts/build_disambiguation_sample.py``).

    AIDA ``validation`` sirve para aprender los pesos; AIDA ``test`` y los
    benchmarks, para medirlos.
    """
    if dataset not in {"aida", *BENCHMARKS}:
        raise ValueError(f"Unknown disambiguation dataset: {dataset!r}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported disambiguation split: {split!r}")
    with disambiguation_sample_path(split, dataset).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        r["candidates"] = [WikidataCandidate(**c) for c in r["candidates"]]
    return rows


def limit_rows_to_documents(
    rows: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    """Limit disambiguation rows by whole documents, preserving row order."""
    if limit is None:
        return rows
    selected = set(list(group_by_doc(rows))[:limit])
    return [row for row in rows if row["doc_id"] in selected]


def group_by_doc(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Menciones de la muestra agrupadas por documento (en su orden), como las
    procesa el desambiguador."""
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    return by_doc


def sample_span(row: dict[str, Any]) -> EntitySpan:
    """La mención de una fila de la muestra, como la entrega el NER."""
    return EntitySpan(
        text=row["mention"],
        start_char=row["start"],
        end_char=row["end"],
        label=row["label"],
        fine_label=row.get("fine_label"),
    )


def load_benchmark(name: str) -> list[Document]:
    """Benchmark clásico de *entity linking* fuera del dominio de AIDA
    (``BENCHMARKS``, publicados por naist-nlp en Hugging Face).

    Cada mención está etiquetada con el identificador de su página en la
    Wikipedia inglesa (``-1``: NIL), que se convierte a QID con la API de
    Wikipedia. Las menciones cuya página ya no existe se descartan: no se
    sabe qué QID les corresponde.
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name!r}")
    path = DATA_DIR / f"{name}_test.jsonl"
    if not path.exists():
        rows: list[dict[str, Any]] = []
        with httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}) as client:
            while True:
                page = _fetch(
                    client, f"naist-nlp/{name}", "dataset", "test", len(rows), PAGE_SIZE
                )
                rows += [item["row"] for item in page["rows"]]
                if len(rows) >= page["num_rows_total"]:
                    break
            page_ids = {
                e["label"][0] for r in rows for e in r["entities"] if e["label"]
            } - {NIL_PAGE}
            qids = _wikipedia_qids(client, sorted(page_ids))
        known = {NIL_PAGE, *qids}  # páginas con QID, y NIL
        docs = [
            Document(
                str(r["id"]),
                r["text"],
                [
                    GoldMention(e["start"], e["end"], None, qids.get(e["label"][0]))
                    for e in r["entities"]
                    if e["label"] and e["label"][0] in known
                ],
            )
            for r in rows
        ]
        _write(path, docs)
    return _read(path)


def _wikipedia_qids(client: httpx.Client, page_ids: list[str]) -> dict[str, str]:
    """Identificador de página de la Wikipedia inglesa -> QID (de 50 en 50)."""
    qids: dict[str, str] = {}
    for i in range(0, len(page_ids), 50):
        response = client.get(
            WIKIPEDIA_API_URL.format(language="en"),
            params={
                "action": "query",
                "pageids": "|".join(page_ids[i : i + 50]),
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "format": "json",
            },
        )
        response.raise_for_status()
        for page in response.json()["query"]["pages"].values():
            qid = page.get("pageprops", {}).get("wikibase_item")
            if qid:
                qids[str(page["pageid"])] = qid
    return qids


def _aida_document(row: dict[str, Any]) -> Document:
    mentions = [
        GoldMention(
            e["start"], e["end"], e["tag"], f"Q{e['qid']}" if e["qid"] else None
        )
        for e in row["entities"]
    ]
    return Document(str(row["document_id"]), row["text"], mentions)


def _wikianc_document(item: dict[str, Any]) -> Document | None:
    row = item["row"]
    text = row["paragraph_text"]
    mentions = [
        GoldMention(a["start"], a["end"], None, f"Q{a['qid']}")
        for a in row["paragraph_anchors"]
        if a["qid"] and text[a["start"] : a["end"]][:1].isupper()
    ]
    if not mentions or not 200 <= len(text) <= 1500:
        return None
    return Document(row["article_title"], text, mentions)


def _fetch(
    client: httpx.Client,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> dict[str, Any]:
    params: dict[str, str | int] = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    # Reintentos ante el límite de peticiones (429) y ante timeouts: la primera
    # lectura de una zona del dataset puede tardar más de un minuto
    for attempt in range(6):
        try:
            response = client.get(ROWS_URL, params=params)
            if response.status_code != 429:
                response.raise_for_status()
                return dict(response.json())
        except httpx.TimeoutException:
            pass
        time.sleep(5 * 2**attempt)
    raise RuntimeError(f"Hugging Face no responde para {dataset} offset={offset}")


def _write(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def _read(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [
            Document(d["doc_id"], d["text"], [GoldMention(**m) for m in d["mentions"]])
            for d in map(json.loads, f)
        ]
