"""Public Wikidata service facade used by the pipeline and disambiguator."""

from pathlib import Path

from entity_linker.config import (
    CACHE_PATH,
    SEARCH_LIMIT,
    WIKIDATA_CONCURRENCY,
    WIKIDATA_TIMEOUT,
)
from entity_linker.schemas import WikidataCandidate

from .cache import WikidataCache
from .enrichment import WikidataEnrichment
from .helpers import Classes, normalize_mention
from .search import WikidataSearch
from .transport import WikidataTransport

__all__ = ["WikidataService", "normalize_mention"]


class WikidataService:
    """Candidate search and enrichment operations backed by Wikidata."""

    def __init__(
        self,
        timeout: float = WIKIDATA_TIMEOUT,
        max_concurrency: int = WIKIDATA_CONCURRENCY,
        cache_path: Path | None = CACHE_PATH,
    ):
        self._cache = WikidataCache(cache_path) if cache_path else None
        self._transport = WikidataTransport(self._cache, timeout, max_concurrency)
        self._search = WikidataSearch(self._transport)
        self._enrichment = WikidataEnrichment(self._transport, self._cache)

    @property
    def _client(self):
        return self._transport.client

    @_client.setter
    def _client(self, client):
        self._transport.client = client

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def get_candidates(
        self, search_term: str, language: str = "en", limit: int = SEARCH_LIMIT
    ) -> list[WikidataCandidate]:
        return await self._search.get_candidates(search_term, language, limit)

    async def get_sitelinks(self, qids: list[str]) -> dict[str, int]:
        return await self._enrichment.get_sitelinks(qids)

    async def get_classes(self, qids: list[str], roots: set[str]) -> Classes:
        return await self._enrichment.get_classes(qids, roots)

    async def get_links(
        self, qids: list[str], targets: list[str]
    ) -> set[tuple[str, str]]:
        return await self._enrichment.get_links(qids, targets)
