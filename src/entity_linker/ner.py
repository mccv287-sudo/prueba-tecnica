"""Reconocimiento de entidades con GLiNER (zero-shot).

GLiNER detecta tipos finos ("sports team", "city"...), que se agrupan en el
esquema CoNLL (PER, ORG, LOC, MISC) para la salida y la evaluación del NER,
pero se conservan en ``EntitySpan.fine_label`` como señal de desambiguación
(``disambiguation.signals.type_score``).
"""

import re

from entity_linker.config import GLINER_MODEL, NER_THRESHOLD
from entity_linker.schemas import EntitySpan

# Tipo fino pedido a GLiNER -> etiqueta CoNLL (el valor por defecto del NER)
FINE_LABELS = {
    "person": "PER",
    "sports team": "ORG",
    "company": "ORG",
    "political party": "ORG",
    "news agency": "ORG",
    "international organization": "ORG",
    "government agency": "ORG",
    "university": "ORG",
    "organization": "ORG",
    "country": "LOC",
    "city": "LOC",
    "region": "LOC",
    "location": "LOC",
    "nationality": "MISC",
    "sports competition": "MISC",
    "event": "MISC",
    "award": "MISC",
    "work of art": "MISC",  # libros, películas, canciones, cuadros...
    "miscellaneous entity": "MISC",
}

# GLiNER procesa ~384 palabras por pasada y descarta el resto: los textos
# largos se trocean por frases
MAX_CHUNK_CHARS = 1500
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class NERService:
    def __init__(
        self, model_name: str = GLINER_MODEL, threshold: float = NER_THRESHOLD
    ):
        # Import diferido: PyTorch solo se carga si se usa GLiNER
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold

    def extract_entities(
        self, text: str, labels: dict[str, str] | None = None
    ) -> list[EntitySpan]:
        """Detecta entidades; ``labels`` (tipo pedido -> etiqueta CoNLL) permite
        probar otros prompts. Por defecto, los tipos finos (``FINE_LABELS``)."""
        labels = labels or FINE_LABELS
        entities = []
        for offset, chunk in _chunks(text):
            for pred in self.model.predict_entities(
                chunk, list(labels), threshold=self.threshold
            ):
                start, end = offset + pred["start"], offset + pred["end"]
                entities.append(
                    EntitySpan(
                        text=text[start:end],
                        start_char=start,
                        end_char=end,
                        label=labels[pred["label"]],
                        fine_label=pred["label"],
                    )
                )
        return entities


def _chunks(text: str) -> list[tuple[int, str]]:
    """Trozos de frases completas, con su posición en el texto original."""
    chunks = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() - start > MAX_CHUNK_CHARS:
            chunks.append((start, text[start : match.start()]))
            start = match.end()
    chunks.append((start, text[start:]))
    return chunks
