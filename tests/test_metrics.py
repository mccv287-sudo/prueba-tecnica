"""Intervalos de confianza y comparación de configuraciones."""

from entity_linker.evaluation.metrics import accuracy_ci, compare

# 20 documentos de 5 menciones: A acierta 3 de 5 en todos
A = {f"d{i}": [True, True, True, False, False] for i in range(20)}


def test_accuracy_ci_contains_the_accuracy():
    low, high = accuracy_ci(A)
    assert low <= 0.6 <= high


def test_same_configuration_is_not_significant():
    result = compare(A, A)
    assert result["diff"] == 0.0
    assert result["ci95"] == [0.0, 0.0]


def test_consistent_improvement_is_significant():
    # B acierta además la cuarta mención de cada documento: +0,2
    b = {doc: [True, True, True, True, False] for doc in A}
    result = compare(A, b)
    assert result["diff"] == 0.2
    assert result["ci95"][0] > 0  # el intervalo no incluye el 0
