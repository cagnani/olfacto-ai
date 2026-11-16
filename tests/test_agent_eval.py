from olfacto.agent.evaluate import parse_preference, render_eval

METRICS = {
    "n": 20,
    "llm_coverage": 0.9,
    "profile_faithfulness": 1.0,
    "description_faithfulness": 0.85,
    "pairing_diversity": 0.95,
    "latency_p50": 2.1,
    "latency_p95": 3.4,
    "llm_wins": 14,
    "template_wins": 4,
    "ties": 2,
    "llm_win_rate": 14 / 18,
    "judge_pairs": 20,
    "fallback_count": 2,
}


def test_parse_preference_accepts_valid_values():
    assert parse_preference('{"vencedor": "A"}') == "A"
    assert parse_preference('{"vencedor": "b"}') == "B"
    assert parse_preference('{"vencedor": "empate"}') == "empate"


def test_parse_preference_rejects_invalid_values():
    assert parse_preference("não é json") is None
    assert parse_preference('{"vencedor": "C"}') is None


def test_render_eval_contains_metrics():
    text = render_eval(METRICS)

    assert "20" in text
    assert "90.0%" in text
    assert "77.8%" in text
    assert "14" in text
    assert "TBD" not in text
