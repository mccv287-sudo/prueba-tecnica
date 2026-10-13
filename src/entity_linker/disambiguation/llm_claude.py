"""Prueba: el mismo selector de candidatos con Claude (API de Anthropic) en vez
del LLM local.

Reutiliza el prompt y la validación de ``LLMChooser``; solo cambia la llamada.
Se activa pasándolo al desambiguador:
``Disambiguator(wikidata, llm_chooser=ClaudeChooser())``.

Requisitos: ``ANTHROPIC_API_KEY``, como variable de entorno o en el ``.env`` de
la raíz del proyecto (no se versiona). El modelo se puede cambiar con
``ENTITY_LINKER_CLAUDE_MODEL``.
"""

import json
import os
import re

import httpx

from entity_linker.config import ROOT
from entity_linker.disambiguation.llm import LLMChooser

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_MODEL = os.getenv("ENTITY_LINKER_CLAUDE_MODEL", "claude-sonnet-5-5")
_JSON = re.compile(r"\{.*?\}", re.DOTALL)
SYSTEM = "Respond with only the JSON object requested, with no explanation."


class ClaudeChooser(LLMChooser):
    """``LLMChooser`` que pregunta a Claude en lugar de a Ollama."""

    def __init__(self, model: str = CLAUDE_MODEL, timeout: float = 60.0):
        super().__init__(model=model, timeout=timeout, url=ANTHROPIC_URL)
        self.api_key = _api_key()

    async def _ask(self, prompt: str) -> int:
        async with self._semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                # Sin temperature: los modelos actuales no la admiten. Margen en
                # max_tokens por si razona o explica antes del JSON
                json={
                    "model": self.model,
                    "max_tokens": 400,
                    "system": SYSTEM,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        response.raise_for_status()
        self.calls += 1
        blocks = response.json()["content"]  # puede empezar por un bloque "thinking"
        text = "".join(b["text"] for b in blocks if b["type"] == "text")
        match = _JSON.search(text)  # por si añade texto alrededor del JSON
        if match is None:
            raise ValueError(f"Respuesta sin JSON: {text!r}")
        return int(json.loads(match.group())["choice"])


def _api_key() -> str:
    """``ANTHROPIC_API_KEY`` del entorno o, si no está, del ``.env`` del
    proyecto (líneas ``CLAVE=valor``)."""
    if key := os.getenv("ANTHROPIC_API_KEY"):
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    raise RuntimeError("Falta ANTHROPIC_API_KEY (variable de entorno o .env)")
