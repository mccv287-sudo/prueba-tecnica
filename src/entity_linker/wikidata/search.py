"""Candidate retrieval from Wikidata and Wikipedia."""

import asyncio
from typing import Any

from rapidfuzz import fuzz, utils

from entity_linker.config import SEARCH_LIMIT, WIKIPEDIA_API_URL
from entity_linker.schemas import WikidataCandidate

from .helpers import DISAMBIGUATION_PAGES, candidate, normalize_mention
from .transport import WikidataTransport

FUZZY_MIN = 80


class WikidataSearch:
    def __init__(self, transport: WikidataTransport):
        self._transport = transport

    async def get_candidates(
        self, search_term: str, language: str = "en", limit: int = SEARCH_LIMIT
    ) -> list[WikidataCandidate]:
        term = normalize_mention(search_term)
        candidates, redirect = await self._search_and_redirect(term, language, limit)
        if not candidates and term.lower().startswith("the "):
            candidates = await self._search(term[4:], language, limit)
        if not candidates:
            candidates = await self._fuzzy_search(term, language, limit)
        if redirect:
            candidates = await self._put_first(redirect, candidates, language)
        return candidates[:limit]

    async def _search_and_redirect(
        self, term: str, language: str, limit: int
    ) -> tuple[list[WikidataCandidate], str | None]:
        return await asyncio.gather(
            self._search(term, language, limit),
            self._wikipedia_qid(term, language),
        )

    async def _search(
        self, term: str, language: str, limit: int
    ) -> list[WikidataCandidate]:
        data = await self._transport.get(
            {
                "action": "wbsearchentities",
                "search": term,
                "language": language,
                "uselang": language,
                "type": "item",
                "limit": limit,
                "format": "json",
            }
        )
        return [
            WikidataCandidate(
                qid=item["id"],
                label=item.get("label", term),
                description=item.get("description"),
            )
            for item in (data or {}).get("search", [])
            if item.get("description") not in DISAMBIGUATION_PAGES
        ]

    async def _wikipedia_qid(self, term: str, language: str) -> str | None:
        if not language.isalpha():
            return None
        data = await self._transport.get(
            {
                "action": "query",
                "titles": term,
                "redirects": 1,
                "prop": "pageprops",
                "ppprop": "wikibase_item|disambiguation",
                "format": "json",
            },
            url=WIKIPEDIA_API_URL.format(language=language),
        )
        for page in (data or {}).get("query", {}).get("pages", {}).values():
            props = page.get("pageprops", {})
            if "disambiguation" not in props:
                return props.get("wikibase_item")
        return None

    async def _put_first(
        self, qid: str, candidates: list[WikidataCandidate], language: str
    ) -> list[WikidataCandidate]:
        found = next((item for item in candidates if item.qid == qid), None)
        if found is None:
            entities = await self._entities([qid], language, "labels|descriptions")
            found = candidate(qid, entities.get(qid, {}), language)
            if found is None:
                return candidates
        return [found, *(item for item in candidates if item.qid != qid)]

    async def _fuzzy_search(
        self, term: str, language: str, limit: int
    ) -> list[WikidataCandidate]:
        data = await self._transport.get(
            {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srnamespace": 0,
                "srlimit": limit,
                "format": "json",
            }
        )
        qids = [hit["title"] for hit in (data or {}).get("query", {}).get("search", [])]
        if not qids:
            return []
        entities = await self._entities(qids, language, "labels|descriptions|aliases")
        scored = []
        for item_qid in qids:
            entity = entities.get(item_qid, {})
            item = candidate(item_qid, entity, language)
            if item is None:
                continue
            aliases = [
                alias["value"] for alias in entity.get("aliases", {}).get(language, [])
            ]
            similarity = max(
                fuzz.token_sort_ratio(term, name, processor=utils.default_process)
                for name in [item.label, *aliases]
            )
            if similarity >= FUZZY_MIN:
                scored.append((similarity, item))
        return [item for _, item in sorted(scored, key=lambda score: -score[0])]

    async def _entities(
        self, qids: list[str], language: str, props: str
    ) -> dict[str, Any]:
        data = await self._transport.get(
            {
                "action": "wbgetentities",
                "ids": "|".join(qids),
                "props": props,
                "languages": language,
                "format": "json",
            }
        )
        return (data or {}).get("entities", {})
