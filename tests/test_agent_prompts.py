from olfacto.agent.prompts import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    fallback_response,
)

AROMAS = [
    {"label": "smoky", "label_pt_br": "defumado", "probability": 0.8},
    {"label": "sweet", "label_pt_br": "doce", "probability": 0.6},
    {"label": "woody", "label_pt_br": "amadeirado", "probability": 0.4},
]
CONTEXTS = [
    {
        "dish": "Feijoada",
        "description": "Feijoada densa e defumada.",
        "pairings": ["arroz", "laranja"],
        "score": 0.5,
    }
]


def test_system_prompt_has_guardrails():
    text = SYSTEM_PROMPT.lower()

    assert "não invente aromas" in text
    assert "alergênicos" in text
    assert "json" in text


def test_prompt_contains_dish_aromas_and_context():
    prompt = build_user_prompt(
        "Feijoada", "Brasileira", ["Feijão", "Bacon"], AROMAS, CONTEXTS
    )

    assert "Feijoada" in prompt
    assert "defumado (80%)" in prompt
    assert "Feijoada: Feijoada densa" in prompt
    assert "SOMENTE" in prompt
    assert "JSON" in prompt


def test_prompt_handles_missing_context():
    prompt = build_user_prompt("Salada", "Saudável", ["Alface"], AROMAS, [])

    assert "Nenhum contexto recuperado" in prompt


def test_fallback_response():
    result = fallback_response("Feijoada", AROMAS, ["arroz", "laranja", "caipirinha"])

    assert result["descricao"].startswith("Feijoada")
    assert result["perfil_sensorial"] == ["defumado", "doce", "amadeirado"]
    assert result["harmonizacoes"] == ["arroz", "laranja", "caipirinha"]
    assert result["confianca"] == "baixa"


def test_fallback_handles_single_aroma():
    result = fallback_response("Suco", AROMAS[:1], [])

    assert result["perfil_sensorial"] == ["defumado"]
    assert len(result["harmonizacoes"]) == 3


def test_schema_required_keys():
    assert set(RESPONSE_SCHEMA["required"]) == {
        "descricao",
        "perfil_sensorial",
        "harmonizacoes",
        "confianca",
    }
