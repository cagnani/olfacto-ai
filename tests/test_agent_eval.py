from olfacto.agent.evaluate import parse_judge, render_eval

METRICS = {
    "n": 20,
    "llm_coverage": 0.9,
    "profile_faithfulness": 1.0,
    "description_faithfulness": 0.85,
    "pairing_diversity": 0.95,
    "latency_p50": 2.1,
    "latency_p95": 3.4,
    "judge_llm_mean": 4.2,
    "judge_llm_std": 0.6,
    "judge_template_mean": 3.1,
    "judge_template_std": 0.5,
    "judge_pairs": 18,
    "fallback_count": 2,
}


def test_parse_judge_accepts_valid_score():
    assert parse_judge('{"nota": 4, "justificativa": "boa"}') == 4


def test_parse_judge_rejects_invalid_values():
    assert parse_judge("não é json") is None
    assert parse_judge('{"nota": 9}') is None
    assert parse_judge('{"nota": "quatro"}') is None


def test_render_eval_contains_metrics():
    text = render_eval(METRICS)

    assert "20" in text
    assert "90.0%" in text
    assert "4.20" in text
    assert "+1.10" in text
    assert "TBD" not in text
