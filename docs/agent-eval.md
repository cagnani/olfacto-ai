# Olfacto Sommelier — avaliação

Benchmark de 20 pratos em 8 categorias. A preferência é julgada por um
LLM, não por um painel humano.

| Métrica | Valor |
|---|---|
| Pratos avaliados | 20 |
| Cobertura do LLM | 95.0% |
| Faithfulness do perfil | 100.0% |
| Faithfulness da descrição | 75.0% |
| Harmonizações com 3 itens únicos | 100.0% |
| Latência p50 / p95 | 2.92s / 4.89s |
| Vitórias do LLM | 15 |
| Vitórias do template | 5 |
| Empates | 0 |
| Taxa de preferência do LLM | 75.0% |

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
