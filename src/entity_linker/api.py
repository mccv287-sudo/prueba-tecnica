from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.openapi.models import Example

from entity_linker.pipeline import EntityLinkingPipeline
from entity_linker.schemas import LinkingOptions, TextLinkingResponse

# Textos de prueba, seleccionables en el desplegable "Examples" de /docs. El
# idioma va en la URL: los ejemplos en español necesitan language=es
EXAMPLES: dict[str, Example] = {
    "homonimos": {
        "summary": "Homónimos: Columbia, el sello discográfico (relación con "
        "sus artistas)",
        "value": "Despite featuring some of the most prominent musicians of "
        "their decade, like Sinatra, Dylan, Joel, and Santana, Columbia was "
        "acquired by Sony in the 1980s.",
    },
    "contexto": {
        "summary": "Contexto frente al orden de búsqueda (Paris, Texas)",
        "value": "Paris Hilton arrived in Paris, Texas, for the opening of her "
        "new hotel.",
    },
    "biografia": {
        "summary": "Biografía: personas, lugares, premios",
        "value": "Marie Curie was born in Warsaw and studied at the University "
        "of Paris. In 1903 she shared the Nobel Prize in Physics with Pierre "
        "Curie and Henri Becquerel.",
    },
    "biografia_marie_curie": {
        "summary": "Wikipedia-style biography: Marie Curie (use_llm=false)",
        "value": "Marie Curie was a physicist and chemist whose research on "
        "radioactivity influenced the development of modern science. She was "
        "born Maria Skłodowska in Warsaw on 7 November 1867, when the city was "
        "part of the Russian Empire. Her parents were teachers, and she "
        "received much of her early education at home. Because women had "
        "limited access to higher education in the Russian-controlled "
        "universities, she attended informal classes in Warsaw and worked as a "
        "tutor before leaving Poland.\n\n"
        "In 1891, Curie moved to Paris to study at the Sorbonne. She earned "
        "degrees in physics and mathematics and met Pierre Curie, a physicist "
        "working at the School of Industrial Physics and Chemistry. They "
        "married in 1895 and began a scientific collaboration. Their studies "
        "of uranium rays led them to investigate minerals that appeared more "
        "active than uranium alone. With the chemist Gustave Bémont, they "
        "identified polonium, named after Curie's homeland, and radium. The "
        "work required the processing of large quantities of pitchblende and "
        "was carried out with limited laboratory resources.\n\n"
        "Curie shared the 1903 Nobel Prize in Physics with Pierre Curie and "
        "Henri Becquerel for research into radiation. In 1906, after Pierre's "
        "death in Paris, she continued their work and became the first woman "
        "to teach at the Sorbonne. She received the 1911 Nobel Prize in "
        "Chemistry for the discovery and study of radium and polonium. During "
        "the First World War, she helped organize mobile radiography units for "
        "French military hospitals. Curie later directed the Radium Institute "
        "in Paris. She died in 1934 in Passy, France. Her daughter Irène Joliot-"
        "Curie also became a scientist and received the Nobel Prize in "
        "Chemistry with Frédéric Joliot-Curie in 1935.",
    },
    "biografia_garcia_marquez": {
        "summary": "Biografía estilo Wikipedia: García Márquez "
        "(language=es, use_llm=false)",
        "value": "Gabriel García Márquez fue un escritor y periodista colombiano, "
        "considerado una de las figuras centrales de la literatura en español "
        "del siglo XX. Nació el 6 de marzo de 1927 en Aracataca, en el "
        "departamento de Magdalena. Pasó parte de su infancia con sus abuelos, "
        "cuyas historias y recuerdos de la región influyeron en su imaginación "
        "literaria. Más tarde estudió en Bogotá y comenzó estudios de Derecho "
        "en la Universidad Nacional de Colombia, aunque pronto orientó su "
        "actividad hacia el periodismo y la escritura.\n\n"
        "Durante las décadas de 1950 y 1960 trabajó como periodista en "
        "Colombia, Europa y México. Publicó reportajes, cuentos y novelas, "
        "entre ellas La hojarasca y El coronel no tiene quien le escriba. En "
        "1967 apareció Cien años de soledad, novela que narra varias "
        "generaciones de la familia Buendía en el pueblo ficticio de Macondo. "
        "La obra alcanzó lectores internacionales y se convirtió en una de "
        "las novelas más conocidas de la literatura latinoamericana. García "
        "Márquez también publicó El otoño del patriarca, Crónica de una muerte "
        "anunciada y El amor en los tiempos del cólera.\n\n"
        "En 1982 recibió el Premio Nobel de Literatura. En su discurso de "
        "aceptación, habló de la historia y las condiciones políticas de "
        "América Latina. Su obra suele asociarse con el realismo mágico, "
        "aunque incluye distintos estilos y géneros. Mantuvo vínculos con "
        "Cuba y con Fidel Castro, una relación que generó debate entre "
        "escritores y comentaristas. En sus últimos años vivió principalmente "
        "en Ciudad de México, donde murió el 17 de abril de 2014. Su archivo "
        "personal se conserva en la Universidad de Texas en Austin, y sus "
        "libros continúan siendo publicados y estudiados en distintos países.",
    },
    "nil": {
        "summary": "NIL: persona que no está en Wikidata",
        "value": "Local baker Jonathan Pemberley-Smythe won the village pie "
        "contest in Little Snoring last Sunday.",
    },
    "espanol": {
        "summary": "Español: literatura (language=es)",
        "value": "Gabriel García Márquez nació en Aracataca y escribió Cien "
        "años de soledad, novela con la que ganó el Premio Nobel de "
        "Literatura en 1982.",
    },
    "espanol_deporte": {
        "summary": "Español: crónica deportiva extensa (language=es, use_llm=false)",
        "value": "La selección argentina llegó a Miami antes de disputar un "
        "amistoso en el Chase Stadium. Lionel Messi entrenó con sus compañeros "
        "del Inter Miami, mientras el cuerpo técnico revisaba vídeos de los "
        "últimos partidos. En la rueda de prensa, el entrenador destacó la "
        "importancia de cuidar el balón y evitar pérdidas cerca del área.\n\n"
        "Al día siguiente, el equipo se enfrentó al Orlando City en un partido "
        "que atrajo a aficionados de Florida y de otros estados. Messi participó "
        "en la jugada del primer gol y asistió a un compañero en la segunda "
        "parte. El resultado fue ajustado, pero el Inter Miami mantuvo la "
        "ventaja hasta el final. Varios jugadores jóvenes también tuvieron "
        "minutos, una oportunidad que el club considera importante para la "
        "temporada.\n\n"
        "En España, el Barcelona siguió la jornada desde la ciudad deportiva. "
        "El club confirmó conversaciones con Pedri sobre su futuro y recordó "
        "que la planificación depende del calendario de La Liga y de la "
        "Champions League. La prensa comparó el ambiente de Miami con el de "
        "Liverpool, donde el equipo local se preparaba para recibir al Arsenal. "
        "Los dos encuentros mostraron cómo una misma crónica puede mencionar "
        "clubes, ciudades, competiciones y personas con nombres potencialmente "
        "ambiguos.",
    },
    "ciencia_extensa": {
        "summary": "English: science and institutions in a long report (use_llm=false)",
        "value": "Researchers at the University of Cambridge and the European "
        "Space Agency announced a joint study of samples collected by the "
        "Hayabusa2 mission. The team compared laboratory measurements with "
        "observations made by the spacecraft near the asteroid Ryugu. NASA "
        "scientist Melissa Hayes said that the results help explain how water "
        "and organic compounds may have reached the early Earth. The paper, "
        "published in Nature, does not claim to have found evidence of life.\n\n"
        "The analysis brought together geologists, chemists, and engineers "
        "from Japan, Germany, and the United Kingdom. JAXA provided mission "
        "data, while the Natural History Museum in London helped compare the "
        "material with meteorites in its collection. Several measurements "
        "were repeated independently because the grains are extremely small "
        "and can be altered by handling. The researchers say that future work "
        "will compare the findings with material returned by NASA's OSIRIS-REx "
        "mission from asteroid Bennu.\n\n"
        "The project also illustrates a challenge for science reporting: names "
        "such as Cambridge, London, and Japan can refer to places or to the "
        "institutions associated with them. In this article, Cambridge refers "
        "to the university, London to the museum, and Japan to the country. "
        "The distinction matters when building a searchable record of the "
        "paper, its authors, partner organizations, locations, and the space "
        "missions discussed in the research.",
    },
}

# Cuerpo en texto plano, con los ejemplos seleccionables en /docs
PLAIN_TEXT_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {"text/plain": {"schema": {"type": "string"}, "examples": EXAMPLES}},
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Crea los recursos al arrancar (en el bucle de eventos que los usará) y los
    cierra al apagar: modelo, cliente HTTP y caché de Wikidata."""
    app.state.pipeline = EntityLinkingPipeline()
    yield
    await app.state.pipeline.wikidata.aclose()


app = FastAPI(
    title="Entity Linker API",
    description=(
        "Servicio REST de Reconocimiento y Enlace de Entidades a Wikidata "
        "(NER + Entity Linking)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "Argos Entity Linker"}


@app.post(
    "/api/v1/link/text",
    response_model=TextLinkingResponse,
    openapi_extra=PLAIN_TEXT_BODY,
)
async def link_plain_text(
    options: Annotated[LinkingOptions, Query()], http_request: Request
) -> TextLinkingResponse:
    """Detecta las entidades del texto y las enlaza con Wikidata.

    el cuerpo es el texto y las opciones van como parámetros de la URL 
    (``language``, ``use_llm``, ``review_threshold``, ``max_candidates``) Las entidades con
    ``needs_review`` (NIL o confianza baja) incluyen los principales candidatos
    para que una persona elija.

    Raises:
        HTTPException: 400 si el texto está vacío o no es UTF-8.
    """
    try:
        text = (await http_request.body()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="El texto debe ser UTF-8.") from exc
    if not text.strip():
        raise HTTPException(
            status_code=400, detail="El texto proporcionado no puede estar vacío."
        )
    pipeline: EntityLinkingPipeline = http_request.app.state.pipeline
    return await pipeline.process_text(
        text,
        language=options.language,
        use_llm=options.use_llm,
        review_threshold=options.review_threshold,
        max_candidates=options.max_candidates,
    )
