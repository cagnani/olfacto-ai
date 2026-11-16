from pathlib import Path

import pytest

from olfacto.agent.retrieval import SensoryRetriever, load_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "sensory_corpus.jsonl"


def make_retriever() -> SensoryRetriever:
    return SensoryRetriever(load_corpus(CORPUS_PATH))


def test_retrieves_exact_dish():
    results = make_retriever().retrieve("feijoada", k=1)

    assert results[0]["dish"] == "Feijoada"
    assert results[0]["score"] > 0


def test_retrieves_ingredient_context():
    results = make_retriever().retrieve("camarão coco dendê", k=5)

    dishes = {entry["dish"] for entry in results}
    assert dishes & {"Bobó de camarão", "Vatapá", "Acarajé", "Moqueca de peixe"}


def test_empty_query_returns_nothing():
    assert make_retriever().retrieve("   ", k=3) == []


def test_results_are_deterministic_and_sorted():
    retriever = make_retriever()
    first = retriever.retrieve("pizza queijo tomate", k=4)
    second = retriever.retrieve("pizza queijo tomate", k=4)

    assert [entry["id"] for entry in first] == [entry["id"] for entry in second]
    assert [entry["score"] for entry in first] == sorted(
        [entry["score"] for entry in first], reverse=True
    )


def test_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must"):
        make_retriever().retrieve("feijoada", k=0)


def test_rejects_empty_corpus():
    with pytest.raises(ValueError, match="corpus"):
        SensoryRetriever([])
