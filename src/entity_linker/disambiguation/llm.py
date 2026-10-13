"""LLM local con Ollama: selección de candidato, prompt y protocolo JSON.

La URL y el modelo están en ``config`` (``OLLAMA_URL``, ``LLM_MODEL``).

Requisitos: ``curl -fsSL https://ollama.com/install.sh | sh`` y
``ollama pull qwen2.5:3b``.
"""

import asyncio
import json
from typing import Any

import httpx

from entity_linker.config import (
    CONTEXT_CHARS,
    LLM_MODEL,
    LLM_TIMEOUT,
    MAX_CANDIDATES,
    OLLAMA_URL,
)
from entity_linker.schemas import EntitySpan, WikidataCandidate

MAX_DESCRIPTION = 100
PROMPT = """Which Wikidata entity does the mention refer to in this text?

Text: "...{context}..."
Other entities in the document: {document_entities}
Mention: "{mention}" (type: {label})

Candidates:
{candidates}

Answer only with JSON: {{"choice": <candidate number, or 0 if none fits>}}"""


class LLMChooser:
    """Ask the configured local LLM to choose among Wikidata candidates."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT,
        url: str = OLLAMA_URL,
    ):
        self.model = model
        self.timeout = timeout
        self.url = url
        self._semaphore = asyncio.Semaphore(1)
        self.calls = 0

    async def choose_candidate(
        self,
        text: str,
        span: EntitySpan,
        candidates: list[WikidataCandidate],
        probabilities: list[float],
        resolved_names: list[str],
    ) -> int | None:
        """Return the original candidate index, or ``None`` when none fits.

        Raises:
            httpx.HTTPError, KeyError, TypeError, ValueError: If the request or
                response is invalid.
        """
        start = max(0, span.start_char - CONTEXT_CHARS)
        context = text[start : span.end_char + CONTEXT_CHARS]
        order = sorted(range(len(candidates)), key=lambda index: -probabilities[index])
        listing = "\n".join(
            f"{rank}. {candidates[index].label}: "
            f"{(candidates[index].description or 'no description')[:MAX_DESCRIPTION]}"
            for rank, index in enumerate(order, 1)
        )
        prompt = PROMPT.format(
            context=" ".join(context.split()),
            document_entities=", ".join(resolved_names[:MAX_CANDIDATES]) or "none",
            mention=span.text,
            label=span.fine_label or span.label,
            candidates=listing,
        )
        choice = await self._ask(prompt)
        if choice == 0:
            return None
        if not 1 <= choice <= len(order):
            raise ValueError(f"LLM candidate choice is out of range: {choice}")
        return order[choice - 1]

    async def _ask(self, prompt: str) -> int:
        async with self._semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url,
                json=json_request(self.model, prompt, max_tokens=20),
            )
        response.raise_for_status()
        self.calls += 1
        return int(parse_json_reply(response.json())["choice"])


def json_request(
    model: str, prompt: str, max_tokens: int | None = None
) -> dict[str, Any]:
    """Cuerpo de la petición: respuesta en JSON y determinista."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    return body


def parse_json_reply(data: dict[str, Any]) -> dict[str, Any]:
    """JSON de la respuesta del modelo.

    Raises:
        KeyError, ValueError: Si la respuesta no tiene el formato esperado.
    """
    return json.loads(data["choices"][0]["message"]["content"])
