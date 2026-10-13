"""Shared Wikidata result types and parsing helpers."""

import re
from typing import Any

from entity_linker.schemas import WikidataCandidate

Classes = dict[str, tuple[set[str], set[str]]]
DISAMBIGUATION_PAGES = {
    "Wikimedia disambiguation page",
    "página de desambiguación de Wikimedia",
}


def normalize_mention(text: str) -> str:
    """Normalize punctuation and possessives around a mention."""
    text = " ".join(text.split()).replace("’", "'").strip(" \"'“”‘.,;:()")
    return re.sub(r"\s*'s$", "", text)


def candidate(
    qid: str, entity: dict[str, Any], language: str
) -> WikidataCandidate | None:
    label = entity.get("labels", {}).get(language, {}).get("value")
    description = entity.get("descriptions", {}).get(language, {}).get("value")
    if not label or description in DISAMBIGUATION_PAGES:
        return None
    return WikidataCandidate(qid=qid, label=label, description=description)


def batches(items: list[str], size: int = 50) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def values(qids: list[str]) -> str:
    return " ".join(f"wd:{qid}" for qid in qids)


def qid(binding: dict[str, str]) -> str:
    return binding["value"].rsplit("/", 1)[-1]
