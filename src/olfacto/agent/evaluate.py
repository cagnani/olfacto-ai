"""Evaluation helpers for the Olfacto Sommelier."""

from __future__ import annotations

from olfacto.agent.pipeline import parse_json_response


def parse_judge(text: str) -> int | None:
    """Extract a 1-5 quality score from a judge response."""
    data = parse_json_response(text)
    if not isinstance(data, dict):
        return None

    try:
        score = int(data.get("nota"))
    except (TypeError, ValueError):
        return None

    return score if 1 <= score <= 5 else None


def render_eval(metrics: dict) -> str:
    """Render the evaluation summary as Markdown."""
    delta = metrics["judge_llm_mean"] - metrics["judge_template_mean"]
    llm_mean = metrics["judge_llm_mean"]
    llm_std = metrics["judge_llm_std"]
    template_mean = metrics["judge_template_mean"]
    template_std = metrics["judge_template_std"]

    return f"""# Olfacto Sommelier — avaliação

Benchmark de {metrics["n"]} pratos em 8 categorias. As notas de qualidade são de um
juiz LLM (1–5), não de um painel humano.

| Métrica | Valor |
|---|---|
| Pratos avaliados | {metrics["n"]} |
| Cobertura do LLM | {metrics["llm_coverage"]:.1%} |
| Faithfulness do perfil | {metrics["profile_faithfulness"]:.1%} |
| Faithfulness da descrição | {metrics["description_faithfulness"]:.1%} |
| Harmonizações com 3 itens únicos | {metrics["pairing_diversity"]:.1%} |
| Latência p50 / p95 | {metrics["latency_p50"]:.2f}s / {metrics["latency_p95"]:.2f}s |
| Nota do juiz — LLM | {llm_mean:.2f} ± {llm_std:.2f} |
| Nota do juiz — template | {template_mean:.2f} ± {template_std:.2f} |
| Ganho do LLM | {delta:+.2f} |

## Como interpretar

- **Faithfulness do perfil:** todos os aromas do perfil estão na lista prevista pelo modelo.
- **Faithfulness da descrição:** nenhum termo olfativo fora do perfil (exceto ingredientes do
  próprio prato) aparece no texto. É uma checagem heurística por termos do léxico PT-BR.
- **Cobertura do LLM:** proporção de pratos em que o LLM respondeu e passou na validação;
  o restante caiu no template determinístico.
- **Nota do juiz:** avaliação automática de clareza, linguagem sensorial e utilidade.

## Limitações

- O juiz é um LLM, não um painel humano; as notas servem para comparação relativa.
- O benchmark tem 20 pratos e não cobre todo o cardápio.
- O modelo de odor foi treinado em catálogos de fragrância; ingredientes salgados podem ter
  previsões fracas.

## Como reproduzir

```bash
uv run --extra train --extra app --extra chem python scripts/evaluate_agent.py
```
"""
