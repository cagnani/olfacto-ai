"""Evaluation helpers for the Olfacto Sommelier."""

from __future__ import annotations

from olfacto.agent.pipeline import parse_json_response


def parse_preference(text: str) -> str | None:
    """Extract a pairwise preference from a judge response."""
    data = parse_json_response(text)
    if not isinstance(data, dict):
        return None

    value = str(data.get("vencedor", "")).strip().lower()
    if value in {"a", "opção a", "opcao a"}:
        return "A"
    if value in {"b", "opção b", "opcao b"}:
        return "B"
    if value in {"empate", "tie"}:
        return "empate"
    return None


def render_eval(metrics: dict) -> str:
    """Render the evaluation summary as Markdown."""
    win_rate = metrics["llm_win_rate"]

    return f"""# Olfacto Sommelier — avaliação

Benchmark de {metrics["n"]} pratos em 8 categorias. A preferência é julgada por um
LLM, não por um painel humano.

| Métrica | Valor |
|---|---|
| Pratos avaliados | {metrics["n"]} |
| Cobertura do LLM | {metrics["llm_coverage"]:.1%} |
| Faithfulness do perfil | {metrics["profile_faithfulness"]:.1%} |
| Faithfulness da descrição | {metrics["description_faithfulness"]:.1%} |
| Harmonizações com 3 itens únicos | {metrics["pairing_diversity"]:.1%} |
| Latência p50 / p95 | {metrics["latency_p50"]:.2f}s / {metrics["latency_p95"]:.2f}s |
| Vitórias do LLM | {metrics["llm_wins"]} |
| Vitórias do template | {metrics["template_wins"]} |
| Empates | {metrics["ties"]} |
| Taxa de preferência do LLM | {win_rate:.1%} |

## Como interpretar

- **Faithfulness do perfil:** todos os aromas do perfil estão na lista prevista pelo modelo.
- **Faithfulness da descrição:** nenhum termo olfativo fora do perfil (exceto ingredientes do
  próprio prato) aparece no texto. É uma checagem heurística por termos do léxico PT-BR.
- **Cobertura do LLM:** proporção de pratos em que o LLM respondeu e passou na validação;
  o restante caiu no template determinístico.
- **Preferência:** o juiz compara as duas descrições lado a lado e escolhe a melhor
  (clareza, linguagem sensorial, coerência e utilidade). A ordem A/B é randomizada.

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
