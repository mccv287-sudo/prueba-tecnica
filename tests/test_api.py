"""Endpoints con un pipeline simulado (sin GLiNER ni Wikidata)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from entity_linker import api
from entity_linker.schemas import TextLinkingResponse

URL = "/api/v1/link/text"


def test_openapi_offers_the_examples_as_plain_text():
    body = api.app.openapi()["paths"][URL]["post"]["requestBody"]["content"]
    examples = body["text/plain"]["examples"]
    assert set(examples) == set(api.EXAMPLES)
    # Ejemplos largos en inglés y español (biografías y crónicas)
    for name in (
        "biografia_marie_curie",
        "biografia_garcia_marquez",
        "espanol_deporte",
        "ciencia_extensa",
    ):
        assert len(examples[name]["value"]) > 1000


@pytest.fixture
def pipeline(monkeypatch):
    fake = MagicMock()
    fake.wikidata.aclose = AsyncMock()
    fake.process_text = AsyncMock(
        return_value=TextLinkingResponse(
            original_text="Paris", entities=[], processing_time_ms=1.0
        )
    )
    monkeypatch.setattr(api, "EntityLinkingPipeline", lambda: fake)
    return fake


@pytest.fixture
def client(pipeline):
    with TestClient(api.app) as client:
        yield client


def test_health_check(client):
    assert client.get("/").json()["status"] == "ok"


def test_plain_text_body_with_options_as_query_params(client, pipeline):
    response = client.post(
        f"{URL}?language=es&use_llm=false",
        content="Messi marcó dos goles.".encode(),
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 200
    pipeline.process_text.assert_awaited_once_with(
        "Messi marcó dos goles.",
        language="es",
        use_llm=False,
        review_threshold=0.7,
        max_candidates=2,
    )


def test_blank_text_is_400(client):
    assert client.post(URL, content=b"  ").status_code == 400


def test_invalid_options_are_422(client):
    assert client.post(f"{URL}?review_threshold=2", content=b"Paris").status_code == 422
