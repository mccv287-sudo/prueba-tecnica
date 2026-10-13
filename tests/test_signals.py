from entity_linker.disambiguation.signals import (
    FINE_TYPE_CLASSES,
    argmax,
    context_score,
    popularity_scores,
    probabilities,
    relation_score,
    surroundings,
    type_score,
)
from entity_linker.schemas import WikidataCandidate


def candidate(qid: str, description: str | None = None) -> WikidataCandidate:
    return WikidataCandidate(qid=qid, label=qid, description=description)


def test_context_score_counts_description_words_in_context():
    cand = candidate("Q1", "French football club")
    assert context_score(cand, "the football club won in Paris", "PSG") == 2 / 3
    assert context_score(candidate("Q2"), "any text", "PSG") == 0.0


def test_context_ignores_the_mention():
    text = "Curie won the Nobel Prize in Physics in 1903."
    mention = "Nobel Prize in Physics"
    start = text.index(mention)
    context = surroundings(text, start, start + len(mention))
    assert context == "Curie won the  in 1903."
    # La descripción que repite el nombre no gana contexto (ni aunque el
    # nombre aparezca otra vez cerca)
    repeats_name = candidate("Q1", "controversies around the Nobel Prize in Physics")
    assert context_score(repeats_name, text, mention) == 0.0


def test_popularity_is_relative_to_most_popular_candidate():
    cands = [candidate("Q1"), candidate("Q2"), candidate("Q3")]
    scores = popularity_scores(cands, {"Q1": 3, "Q2": 300})
    assert scores[1] == 1.0  # el más popular
    assert 0 < scores[0] < 1
    assert scores[2] == 0.0  # sin artículos
    assert popularity_scores(cands, {}) == [0.0, 0.0, 0.0]


def test_type_score():
    person = FINE_TYPE_CLASSES["person"]
    classes = {"Q1": ({"Q5"}, {person}), "Q2": ({"Q515"}, set())}
    assert type_score("person", "Q1", classes) == 1.0
    assert type_score("person", "Q2", classes) == 0.0
    # Tipo genérico o candidato sin clases consultadas: no se sabe
    assert type_score(None, "Q1", classes) == 0.5
    assert type_score("person", "Q3", classes) == 0.5


def test_relation_score_counts_linked_or_same_resolved_entities():
    links = {("Q3308285", "Q168756")}  # Michael I. Jordan -> empleador -> UC Berkeley
    assert relation_score("Q3308285", ["Q168756", "Q90"], links) == 0.5
    assert relation_score("Q41421", ["Q168756", "Q90"], links) == 0.0
    assert relation_score("Q90", ["Q90"], set()) == 1.0  # la misma entidad
    assert relation_score("Q90", [], links) == 0.0


def test_probabilities_sum_to_one_and_favor_first_rank():
    probs = probabilities([{"context": 0.0}, {"context": 0.0}, {"context": 0.0}])
    assert abs(sum(probs) - 1) < 1e-9
    assert argmax(probs) == 0
    assert probabilities([]) == []


def test_probabilities_signal_can_beat_rank():
    probs = probabilities([{"context": 0.0}, {"context": 1.0}])
    assert argmax(probs) == 1


def test_probabilities_accept_explicit_weights():
    weights = {"prior": 0.0, "context": 10.0}
    probs = probabilities([{"context": 0.0}, {"context": 1.0}], weights)
    assert argmax(probs) == 1
    assert probs[1] > 0.99
