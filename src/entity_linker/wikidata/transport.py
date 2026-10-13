"""HTTP transport, retry policy, and request-level caching for Wikidata."""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from entity_linker.config import (
    USER_AGENT,
    WIKIDATA_API_URL,
    WIKIDATA_CONCURRENCY,
    WIKIDATA_TIMEOUT,
)

from .cache import WikidataCache

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class WikidataTransport:
    def __init__(
        self,
        cache: WikidataCache | None,
        timeout: float = WIKIDATA_TIMEOUT,
        max_concurrency: int = WIKIDATA_CONCURRENCY,
    ):
        self.client = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        self._cache = cache
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._resume_at = 0.0

    async def aclose(self) -> None:
        await self.client.aclose()
        if self._cache:
            self._cache.close()

    async def get(
        self,
        params: dict[str, Any],
        url: str = WIKIDATA_API_URL,
        timeout: float | None = None,
        cache: bool = True,
    ) -> dict[str, Any] | None:
        key = url + "?" + urlencode(sorted(params.items()))
        if cache and self._cache and (cached := self._cache.get(key)) is not None:
            return cached

        for attempt in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    await asyncio.sleep(self._resume_at - time.monotonic())
                    response = await self.client.get(
                        url, params=params, timeout=timeout or self.client.timeout
                    )
                    if response.status_code == 429:
                        wait = float(response.headers.get("Retry-After", 2**attempt))
                        self._resume_at = time.monotonic() + wait
                        continue
                if response.status_code not in RETRYABLE_STATUS:
                    response.raise_for_status()
                    data = response.json()
                    if cache and self._cache:
                        self._cache.store(key, data)
                    return data
            except httpx.TransportError:
                pass
            except httpx.HTTPError as exc:
                logger.warning("Wikidata: %s", exc)
                return None
            await asyncio.sleep(2**attempt)
        logger.warning("Wikidata sin respuesta: %s", url)
        return None
