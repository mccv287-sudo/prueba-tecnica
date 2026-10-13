"""Cliente de Wikidata con la red simulada (``httpx.MockTransport``)."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from entity_linker.wikidata import WikidataService, normalize_mention

Handler = Callable[[httpx.Request], httpx.Response]


NO_ARTICLE = {"query": {"pages": {"-1": {"title": "x", "missing": ""}}}}


def service(
    handler: Handler, wikipedia: dict[str, Any] = NO_ARTICLE
) -> WikidataService:
    """Servicio sin caché en disco cuyas peticiones a Wikidata responde
    ``handler``; a Wikipedia (redirecciones), ``wikipedia``."""

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host.endswith("wikipedia.org"):
            return httpx.Response(200, json=wikipedia)
        return handler(request)

    wikidata = WikidataService(cache_path=None)
    wikidata._client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    return wikidata


def search_result(*qids: str) -> dict[str, Any]:
    return {
        "search": [
            {
                "id": q,
                "label": f"label {q}",
                "description": f"description {q}",
                "concepturi": f"http://www.wikidata.org/entity/{q}",
            }
            for q in qids
        ]
    }


@pytest.mark.parametrize(
    ("mention", "expected"),
    [("Obama's", "Obama"), (' "Reuters". ', "Reuters"), ("New  York", "New York")],
)
def test_normalize_mention(mention, expected):
    assert normalize_mention(mention) == expected


async def test_get_candidates_parses_search_in_requested_language():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=search_result("Q90", "Q167646"))

    wikidata = service(handler)
    candidates = await wikidata.get_candidates("Paris", language="es")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q90", "Q167646"]
    assert candidates[0].description == "description Q90"
    assert requests[0].url.params["uselang"] == "es"


async def test_get_candidates_skips_disambiguation_pages():
    result = search_result("Q1", "Q2")
    result["search"][0]["description"] = "Wikimedia disambiguation page"
    wikidata = service(lambda request: httpx.Response(200, json=result))
    candidates = await wikidata.get_candidates("Hadley")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q2"]


async def test_get_candidates_retries_without_leading_article():
    def handler(request: httpx.Request) -> httpx.Response:
        found = request.url.params["search"] == "Beatles"
        return httpx.Response(200, json=search_result("Q1299") if found else {})

    wikidata = service(handler)
    candidates = await wikidata.get_candidates("the Beatles")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q1299"]


async def test_wikipedia_redirect_puts_its_item_first():
    redirect = {
        "query": {
            "redirects": [{"from": "Asian Cup", "to": "AFC Asian Cup"}],
            "pages": {"1": {"pageprops": {"wikibase_item": "Q157894"}}},
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["action"] == "wbsearchentities":
            return httpx.Response(200, json=search_result("Q727359", "Q157894"))
        raise AssertionError("ya estaba entre los candidatos: no hace falta pedirlo")

    wikidata = service(handler, wikipedia=redirect)
    candidates = await wikidata.get_candidates("Asian Cup")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q157894", "Q727359"]


async def test_wikipedia_disambiguation_page_is_ignored():
    disambiguation = {
        "query": {
            "pages": {
                "1": {"pageprops": {"wikibase_item": "Q1", "disambiguation": ""}}
            }
        }
    }
    wikidata = service(
        lambda request: httpx.Response(200, json=search_result("Q90")),
        wikipedia=disambiguation,
    )
    candidates = await wikidata.get_candidates("Paris")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q90"]


async def test_fuzzy_search_when_prefix_search_finds_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params["action"] == "wbsearchentities":  # el prefijo no encuentra nada
            return httpx.Response(200, json={"search": []})
        if params["action"] == "query":  # texto completo: dos resultados
            hits = [{"title": "Q1"}, {"title": "Q2"}]
            return httpx.Response(200, json={"query": {"search": hits}})
        entities = {
            "Q1": {"labels": {"en": {"value": "Sam Philpott"}}, "aliases": {}},
            "Q2": {"labels": {"en": {"value": "Philpott Racing"}}, "aliases": {}},
        }
        return httpx.Response(200, json={"entities": entities})

    wikidata = service(handler)
    candidates = await wikidata.get_candidates("S.Philpott")
    await wikidata.aclose()

    # RapidFuzz deja fuera el resultado que no se parece a la mención
    assert [c.qid for c in candidates] == ["Q1"]


async def test_get_candidates_waits_and_retries_after_429():
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json=search_result("Q90")),
        ]
    )
    wikidata = service(lambda request: next(responses))
    candidates = await wikidata.get_candidates("Paris")
    await wikidata.aclose()

    assert [c.qid for c in candidates] == ["Q90"]


async def test_get_sitelinks_counts_wikipedia_articles():
    entities = {
        "Q90": {"sitelinks": {"enwiki": {}, "eswiki": {}, "frwiki": {}}},
        "Q2": {"sitelinks": {}},
    }
    wikidata = service(
        lambda request: httpx.Response(200, json={"entities": entities})
    )
    counts = await wikidata.get_sitelinks(["Q90", "Q2"])
    await wikidata.aclose()

    assert counts == {"Q90": 3, "Q2": 0}


async def test_get_links_returns_linked_pairs():
    entity = "http://www.wikidata.org/entity/"
    row = {"a": {"value": entity + "Q3308285"}, "b": {"value": entity + "Q168756"}}
    wikidata = service(
        lambda request: httpx.Response(200, json={"results": {"bindings": [row]}})
    )
    links = await wikidata.get_links(["Q3308285", "Q41421"], ["Q168756"])
    await wikidata.aclose()

    assert links == {("Q3308285", "Q168756")}


async def test_get_classes_parses_sparql_bindings():
    entity = "http://www.wikidata.org/entity/"

    def handler(request: httpx.Request) -> httpx.Response:
        row = {
            "item": {"value": entity + "Q90"},
            "cls": {"value": entity + "Q515"},
            "root": {"value": entity + "Q515"},
        }
        return httpx.Response(200, json={"results": {"bindings": [row]}})

    wikidata = service(handler)
    classes = await wikidata.get_classes(["Q90", "Q2"], roots={"Q515"})
    await wikidata.aclose()

    assert classes["Q90"] == ({"Q515"}, {"Q515"})
    assert classes["Q2"] == (set(), set())  # sin P31 en la respuesta
