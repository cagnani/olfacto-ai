import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from olfacto.agent.pipeline import (
    SommelierAgent,
    make_inference_llm,
    parse_json_response,
)

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / "data" / "odor_labels_ptbr.json"

AROMAS = [
    {"label": "smoky", "label_pt_br": "defumado", "probability": 0.8},
    {"label": "sweet", "label_pt_br": "doce", "probability": 0.6},
    {"label": "woody", "label_pt_br": "amadeirado", "probability": 0.4},
]

CONTEXT = [
    {
        "id": "feijoada",
        "dish": "Feijoada",
        "description": "Feijoada densa e defumada.",
        "pairings": ["arroz", "laranja", "caipirinha"],
        "score": 0.5,
    }
]


class FakeTools:
    def __init__(self, aromas: list[dict]) -> None:
        self.aromas = aromas

    def profile(self, dish, extra_ingredients=None, top_k=8):
        return {
            "dish": dish,
            "category": "Brasileira",
            "ingredients": [{"name": "Feijão", "smiles": "C", "predictions": []}],
            "missing": [],
            "aromas": self.aromas,
        }


class FakeRetriever:
    def retrieve(self, query, k=3):
        return CONTEXT


class FakeLLM:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def valid_payload(**overrides) -> str:
    payload = {
        "descricao": "Feijoada defumada, com toque doce e final amadeirado.",
        "perfil_sensorial": ["defumado", "doce", "amadeirado"],
        "harmonizacoes": ["arroz", "laranja", "caipirinha"],
        "confianca": "alta",
    }
    payload.update(overrides)
    return json.dumps(payload)


def make_agent(llm: FakeLLM, aromas: list[dict] | None = None) -> SommelierAgent:
    return SommelierAgent(
        FakeTools(AROMAS if aromas is None else aromas), FakeRetriever(), llm, LEXICON
    )


def test_generate_with_valid_response():
    llm = FakeLLM([valid_payload()])

    result = make_agent(llm).generate("Feijoada")

    assert result.source == "llm"
    assert result.description.startswith("Feijoada defumada")
    assert result.sensory_profile == ["defumado", "doce", "amadeirado"]
    assert result.pairings == ["arroz", "laranja", "caipirinha"]
    assert result.confidence == "alta"
    assert llm.calls == 1


def test_retries_after_invalid_json():
    llm = FakeLLM(["não é json", valid_payload()])

    result = make_agent(llm).generate("Feijoada")

    assert result.source == "llm"
    assert llm.calls == 2


def test_falls_back_on_forbidden_profile_aroma():
    payload = valid_payload(
        perfil_sensorial=["defumado", "baunilha", "amadeirado"],
        descricao="Feijoada com baunilha e toque defumado.",
    )
    llm = FakeLLM([payload, payload])

    result = make_agent(llm).generate("Feijoada")

    assert result.source == "fallback"
    assert result.confidence == "baixa"
    assert result.pairings == ["arroz", "laranja", "caipirinha"]


def test_falls_back_on_forbidden_description_term():
    payload = valid_payload(descricao="Feijoada com baunilha e final amadeirado.")
    llm = FakeLLM([payload, payload])

    result = make_agent(llm).generate("Feijoada")

    assert result.source == "fallback"


def test_falls_back_when_llm_raises():
    llm = FakeLLM([RuntimeError("provider offline")])

    result = make_agent(llm).generate("Feijoada")

    assert result.source == "fallback"
    assert llm.calls == 1


def test_falls_back_on_short_profile():
    payload = valid_payload(perfil_sensorial=["defumado", "doce"])
    llm = FakeLLM([payload, payload])

    result = make_agent(llm).generate("Feijoada")

    assert result.source == "fallback"


def test_rejects_dish_without_aromas():
    llm = FakeLLM([valid_payload()])

    with pytest.raises(ValueError, match="no odor predictions"):
        make_agent(llm, aromas=[]).generate("Feijoada")


def test_to_dict_contains_source():
    result = make_agent(FakeLLM([valid_payload()])).generate("Feijoada")

    data = result.to_dict()

    assert data["dish"] == "Feijoada"
    assert data["source"] == "llm"
    assert data["aromas"] == AROMAS


def test_parse_json_response_handles_code_fence():
    text = '```json\n{"descricao": "ok"}\n```'

    assert parse_json_response(text) == {"descricao": "ok"}


def test_make_inference_llm():
    class FakeClient:
        def chat_completion(self, **kwargs):
            message = SimpleNamespace(content='{"ok": 1}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    call = make_inference_llm(FakeClient(), "fake-model")

    assert call([]) == '{"ok": 1}'
