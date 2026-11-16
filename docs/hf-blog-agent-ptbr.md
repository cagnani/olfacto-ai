**TL;DR**

- 45 pratos brasileiros em 8 categorias, 78 ingredientes com SMILES e um corpus sensorial curado.
- O agente usa o modelo de odor como ferramenta: o LLM **não inventa aromas**, apenas redige o texto.
- 95% de cobertura do LLM, 100% de fidelidade do perfil e **75% de preferência** sobre o template determinístico.
- Latência p50 de 2,9s por prato.

**Links:** [Modelo](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138) · [Dataset](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr) · [Demo](https://huggingface.co/spaces/SamuelCagnani/olfacto-agent) · [Código](https://github.com/cagnani/olfacto-ai)

![Arquitetura do Olfacto Sommelier](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/sommelier-architecture.png)

## Do modelo ao produto

No [artigo anterior](https://huggingface.co/blog/SamuelCagnani/olfacto-ai-br), treinei um modelo que prevê 138 descritores olfativos a partir da estrutura de uma molécula. O resultado foi honesto: o ensemble chegou a 0,8402 de ROC-AUC macro out-of-fold, e um baseline de fingerprints venceu o transformer sozinho.

Agora o problema muda. Uma predição de aroma não é um produto. Um restaurante não quer uma lista de probabilidades; quer uma descrição de menu que ajude o cliente a escolher. O **Olfacto Sommelier** é a camada que transforma aroma em texto — com o modelo de odor no centro, não o LLM.

![Demo do Olfacto Sommelier](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/agent-demo.gif)

## Como funciona

1. **Prato → ingredientes.** Um catálogo curado liga 45 pratos a ingredientes. O usuário também pode informar ingredientes extras.
2. **Ingrediente → SMILES.** Dois catálogos (40 ingredientes doces e 38 salgados) guardam o SMILES de uma molécula marcante de cada ingrediente.
3. **SMILES → aromas.** O ensemble do artigo anterior prevê o perfil olfativo de cada molécula. As probabilidades são agregadas por máximo.
4. **Contexto sensorial.** Um corpus de 45 descrições originais é consultado com TF-IDF, sem vector database.
5. **LLM.** O Qwen 7B recebe o prato, os aromas previstos e o contexto recuperado. Ele devolve JSON com descrição, perfil e harmonizações.
6. **Guardrails.** O perfil precisa usar apenas aromas da lista prevista. Se o JSON falhar ou citar outro aroma, há uma nova tentativa e, depois, um fallback determinístico.

O LLM nunca decide quais aromas o prato tem. Ele recebe essa decisão do modelo e escreve a partir dela. É a diferença entre um gerador de texto e um agente com uma ferramenta.

## O corpus

São 45 pratos em 8 categorias: brasileira, lanches, pizza, japonesa, doces, bebidas, saudável e massas. Cada entrada tem ingredientes, aromas de referência, uma descrição curta e harmonizações.

O corpus não é ground truth; é contexto. A função dele é dar ao LLM vocabulário sensorial e evitar que ele invente referências gastronômicas. O retrieval é TF-IDF com unigramas e bigramas, top-3 por similaridade de cosseno.

## O LLM com guardrails

O prompt tem três regras que importam:

- Use somente os aromas da lista prevista.
- Não faça alegações de saúde, alérgenos ou segurança alimentar.
- Responda apenas com JSON válido.

O pipeline valida o schema, exige 3 a 5 aromas no perfil, 3 harmonizações e uma confiança declarada. Se o perfil citar um aroma fora da lista, a resposta é rejeitada. Na segunda falha, entra o template:

> “Feijoada com aroma de carne, notas de torrado, sulfuroso e final equilibrado.”

O fallback é mais pobre, mas é previsível. Em produção, previsibilidade vale mais do que uma frase bonita.

## Resultados

Benchmark de 20 pratos, com um juiz LLM comparando as duas descrições lado a lado (LLM vs. template), em ordem A/B randomizada:

| Métrica | Valor |
|---|---|
| Pratos avaliados | 20 |
| Cobertura do LLM | 95% |
| Fidelidade do perfil | 100% |
| Fidelidade da descrição | 75% |
| Harmonizações com 3 itens únicos | 100% |
| Latência p50 / p95 | 2,9s / 4,9s |
| Preferência do LLM | 75% (15 de 20) |
| Fallback | 1 de 20 |

O número mais importante é o primeiro: **100% do perfil estruturado usa apenas aromas previstos**. A fidelidade da descrição é menor (75%) porque o LLM às vezes adiciona termos plausíveis — “defumado” numa feijoada, “amanteigado” num pão de queijo — que não estão no top-k do modelo. É uma limitação real, não um detalhe.

O juiz LLM preferiu as descrições do agente em 15 dos 20 pares. O template é sempre fiel, mas genérico. O agente é mais útil, com o custo de latência e de uma API externa.

## O que eu aprendi

**O modelo é a ferramenta; o agente é o produto.** A parte difícil não foi chamar o LLM. Foi desenhar o contrato: o que ele pode dizer, o que ele não pode, e o que acontece quando ele falha.

**Guardrails são a parte cara.** Validar o perfil é fácil. Validar a descrição é difícil, porque linguagem natural não tem schema. A checagem automática de termos ainda é heurística; a avaliação humana continua necessária.

**Juiz LLM é ruidoso.** Comparar duas descrições lado a lado é mais confiável do que pedir uma nota de 1 a 5. A preferência pairwise deu um sinal claro; as notas absolutas variavam demais.

**Fallback é uma feature.** Um agente que sempre responde algo previsível é mais útil do que um que falha com uma frase brilhante.

## Limitações

- O juiz é um LLM, não um painel humano; as notas servem para comparação relativa.
- O benchmark tem 20 pratos e não cobre todo o cardápio.
- O modelo de odor foi treinado em catálogos de fragrância; ingredientes salgados podem ter previsões fracas.
- As harmonizações são sugestões, não recomendações gastronômicas validadas.
- Não há alegação de alérgenos, saúde ou segurança alimentar.

## Reprodutibilidade

- Código: [github.com/cagnani/olfacto-ai](https://github.com/cagnani/olfacto-ai)
- Modelo: [SamuelCagnani/olfacto-chemberta-138](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138)
- Dataset: [SamuelCagnani/olfacto-odor-138-ptbr](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr)
- Demo: [SamuelCagnani/olfacto-agent](https://huggingface.co/spaces/SamuelCagnani/olfacto-agent)

O benchmark roda com `python scripts/evaluate_agent.py`; os testes, com `make test`.
