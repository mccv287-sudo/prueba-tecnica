"""Popularity, class, and relation data used by disambiguation signals."""

from typing import Any

from entity_linker.config import WIKIDATA_SPARQL_URL

from .cache import WikidataCache
from .helpers import Classes, batches, qid, values
from .transport import WikidataTransport


class WikidataEnrichment:
    def __init__(self, transport: WikidataTransport, cache: WikidataCache | None):
        self._transport = transport
        self._cache = cache

    async def get_sitelinks(self, qids: list[str]) -> dict[str, int]:
        cached = {
            item: self._cache_get(f"sitelinks:{item}") for item in dict.fromkeys(qids)
        }
        counts = {item: value["count"] for item, value in cached.items() if value}
        for batch in batches([item for item in cached if item not in counts]):
            data = await self._transport.get(
                {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "sitelinks",
                    "format": "json",
                },
                cache=False,
            )
            if data is None:
                continue
            entities = data.get("entities", {})
            for item in batch:
                counts[item] = len(entities.get(item, {}).get("sitelinks", {}))
                self._store(f"sitelinks:{item}", {"count": counts[item]})
        return {item: counts.get(item, 0) for item in cached}

    async def get_classes(self, qids: list[str], roots: set[str]) -> Classes:
        prefix = ",".join(sorted(roots)) + ":"
        cached = {item: self._cache_get(prefix + item) for item in dict.fromkeys(qids)}
        classes: Classes = {
            item: (set(value["direct"]), set(value["roots"]))
            for item, value in cached.items()
            if value
        }
        wanted = ", ".join(f"wd:{root}" for root in sorted(roots))
        for batch in batches([item for item in cached if item not in classes]):
            rows = await self._sparql(
                "SELECT DISTINCT ?item ?cls ?root WHERE { "
                f"VALUES ?item {{ {values(batch)} }} "
                "?item wdt:P31 ?cls . OPTIONAL { ?cls wdt:P279* ?root . "
                f"FILTER(?root IN ({wanted})) }} }}",
                cache=False,
            )
            if rows is None:
                continue
            found: Classes = {item: (set(), set()) for item in batch}
            for row in rows:
                direct, root_classes = found[qid(row["item"])]
                direct.add(qid(row["cls"]))
                if "root" in row:
                    root_classes.add(qid(row["root"]))
            for item, (direct, root_classes) in found.items():
                self._store(
                    prefix + item,
                    {"direct": sorted(direct), "roots": sorted(root_classes)},
                )
            classes.update(found)
        return {item: classes.get(item, (set(), set())) for item in cached}

    async def get_links(
        self, qids: list[str], targets: list[str]
    ) -> set[tuple[str, str]]:
        if not qids or not targets:
            return set()
        links = set()
        unique_targets = list(dict.fromkeys(targets))
        for batch in batches(list(dict.fromkeys(qids))):
            rows = await self._sparql(
                "SELECT DISTINCT ?a ?b WHERE { "
                f"VALUES ?a {{ {values(batch)} }} "
                f"VALUES ?b {{ {values(unique_targets)} }} "
                "{ ?a ?p ?b } UNION { ?b ?p ?a } "
                "FILTER(STRSTARTS(STR(?p), "
                '"http://www.wikidata.org/prop/direct/")) }'
            )
            links |= {(qid(row["a"]), qid(row["b"])) for row in rows or []}
        return links

    async def _sparql(
        self, query: str, cache: bool = True
    ) -> list[dict[str, Any]] | None:
        data = await self._transport.get(
            {"query": query, "format": "json"},
            url=WIKIDATA_SPARQL_URL,
            timeout=60,
            cache=cache,
        )
        return None if data is None else data.get("results", {}).get("bindings", [])

    def _cache_get(self, key: str) -> Any:
        return self._cache.get(key) if self._cache else None

    def _store(self, key: str, body: Any) -> None:
        if self._cache:
            self._cache.store(key, body)
