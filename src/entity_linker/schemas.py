from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field

from entity_linker.config import MAX_CANDIDATES, OUTPUT_CANDIDATES, REVIEW_THRESHOLD

# Tipos reutilizables con sus restricciones
Qid = Annotated[str, Field(pattern=r"^Q\d+$", description="Identificador de Wikidata")]
QidOrNil = Annotated[
    str, Field(pattern=r"^(Q\d+|NIL)$", description="QID de Wikidata o NIL")
]
Confidence = Annotated[
    float, Field(ge=0.0, le=1.0, description="Puntuación de confianza (0.0 a 1.0)")
]


class StrictModel(BaseModel):
    """Base de los modelos internos: sin conversiones implícitas de tipos."""

    model_config = ConfigDict(strict=True)


class EntitySpan(StrictModel):
    text: str = Field(..., description="Texto de la entidad extraída")
    start_char: int = Field(
        ..., ge=0, description="Índice de inicio en el texto original"
    )
    end_char: int = Field(..., ge=0, description="Índice de fin en el texto original")
    label: str = Field(..., description="Categoría NER (PER, ORG, LOC, MISC)")
    fine_label: str | None = Field(
        default=None, description="Tipo fino detectado (p. ej. 'sports team')"
    )


class WikidataCandidate(StrictModel):
    """Candidato recuperado de la API de Wikidata."""

    qid: Qid
    label: str = Field(..., description="Etiqueta u nombre oficial en Wikidata")
    description: str | None = Field(
        default=None, description="Descripción corta de la entidad"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def concept_uri(self) -> str:
        """URI completa en Wikidata."""
        return f"http://www.wikidata.org/entity/{self.qid}"


class LinkedEntity(EntitySpan):
    qid: QidOrNil = "NIL"
    wikidata_label: str | None = Field(
        default=None, description="Nombre oficial en Wikidata"
    )
    wikidata_description: str | None = Field(
        default=None, description="Descripción oficial en Wikidata"
    )
    confidence: Confidence
    is_nil: bool = Field(
        default=False, description="Indica si la entidad no pudo ser vinculada"
    )
    reasoning: str | None = Field(
        default=None, description="Motivo de la decisión (LLM o regla aplicada)"
    )
    llm_confirmed: bool | None = Field(
        default=None,
        description="true: el LLM eligió el mismo candidato que las señales (se "
        "acepta aunque la confianza sea baja); false: eligió otro o ninguno; "
        "null: no se le consultó (use_llm=false, mención ya resuelta por las "
        "señales o fallo del LLM)",
    )
    needs_review: bool = Field(
        default=False,
        description="NIL, desacuerdo del LLM o confianza baja sin confirmar: "
        "conviene que una persona elija entre los candidatos",
    )
    candidates: list[WikidataCandidate] = Field(
        default_factory=list,
        description="Principales candidatos, en el orden usado para decidir",
    )

    def review_needed(self, threshold: float) -> bool:
        """NIL, o confianza por debajo de ``threshold`` que el LLM no confirmó
        (si el LLM eligió otro candidato que las señales, la confianza es la de
        las señales para ese candidato, baja: queda para revisión)."""
        return self.is_nil or (self.confidence < threshold and not self.llm_confirmed)


class LinkingOptions(BaseModel):
    """Opciones de una petición, como parámetros de la URL de
    ``/api/v1/link/text`` (el texto va en el cuerpo)."""

    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2,3}$",
        description="Idioma del texto (ISO 639, p. ej. 'en', 'es'): idioma de "
        "búsqueda y de etiquetas/descripciones en Wikidata",
    )
    use_llm: bool = Field(
        default=True,
        description="Resolver las menciones dudosas con el LLM local. False: "
        "respuesta rápida (~2 s/doc) quedándose con el mejor candidato",
    )
    review_threshold: Confidence = Field(
        default=REVIEW_THRESHOLD,
        description="Confianza mínima: por debajo (o si es NIL) la entidad se "
        "marca con needs_review para revisión humana",
    )
    max_candidates: int = Field(
        default=OUTPUT_CANDIDATES,
        ge=0,
        le=MAX_CANDIDATES,
        description="Candidatos devueltos por entidad (0: respuesta compacta)",
    )


class TextLinkingResponse(BaseModel):
    original_text: str
    entities: list[LinkedEntity]
    processing_time_ms: float
