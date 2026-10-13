"""Pipeline: 1. NER -> 2. candidatos de Wikidata -> 3. desambiguación."""

import asyncio
import time

from entity_linker.config import OUTPUT_CANDIDATES, REVIEW_THRESHOLD
from entity_linker.disambiguation.disambiguator import Disambiguator
from entity_linker.ner import NERService
from entity_linker.schemas import LinkedEntity, TextLinkingResponse
from entity_linker.wikidata import WikidataService


class EntityLinkingPipeline:
    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: El LLM resuelve las menciones que siguen inciertas.
        """
        self.ner = NERService()
        self.wikidata = WikidataService()
        self.disambiguator = Disambiguator(self.wikidata, use_llm=use_llm)

    async def link(
        self, text: str, language: str = "en", use_llm: bool | None = None
    ) -> list[LinkedEntity]:
        # 1. NER con tipos finos (fine_label) agrupados en PER/ORG/LOC/MISC
        spans = self.ner.extract_entities(text)

        # 2. Recupera cada forma superficial una sola vez y conserva el orden
        # original de las menciones para la desambiguación.
        mentions = list(dict.fromkeys(s.text for s in spans))
        retrieved = await asyncio.gather(
            *(
                self.wikidata.get_candidates(mention, language=language)
                for mention in mentions
            )
        )
        by_mention = dict(zip(mentions, retrieved))
        candidates_list = [by_mention[span.text] for span in spans]

        # 3. Desambiguación probabilística (pide las clases solo si hacen falta)
        return await self.disambiguator.disambiguate_document(
            text, spans, list(candidates_list), use_llm=use_llm
        )

    async def process_text(
        self,
        text: str,
        language: str = "en",
        use_llm: bool | None = None,
        review_threshold: float = REVIEW_THRESHOLD,
        max_candidates: int = OUTPUT_CANDIDATES,
    ) -> TextLinkingResponse:
        """Respuesta de la API: entidades enlazadas con la marca de revisión
        (``needs_review``) según ``review_threshold`` y, como mucho,
        ``max_candidates`` candidatos por entidad."""
        start_time = time.perf_counter()
        entities = [
            e.model_copy(
                update={
                    "needs_review": e.review_needed(review_threshold),
                    "candidates": e.candidates[:max_candidates],
                }
            )
            for e in await self.link(text, language, use_llm)
        ]
        return TextLinkingResponse(
            original_text=text,
            entities=entities,
            processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
