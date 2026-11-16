**TL;DR**

- 45 Brazilian dishes across 8 categories, 78 ingredients with SMILES and a curated sensory corpus.
- The agent uses the odor model as a tool: the LLM **does not invent aromas**, it only writes the copy.
- 95% LLM coverage, 100% profile faithfulness and **75% preference** over the deterministic template.
- p50 latency of 2.9s per dish.

**Links:** [Model](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138) · [Dataset](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr) · [Demo](https://huggingface.co/spaces/SamuelCagnani/olfacto-agent) · [Code](https://github.com/cagnani/olfacto-ai)

![Olfacto Sommelier architecture](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/sommelier-architecture.png)

## From model to product

In the [previous article](https://huggingface.co/blog/SamuelCagnani/olfacto-ai-br), I trained a model that predicts 138 odor descriptors from a molecule's structure. The result was honest: the ensemble reached 0.8402 macro ROC-AUC out-of-fold, and a fingerprint baseline beat the transformer alone.

Now the problem changes. An odor prediction is not a product. A restaurant does not want a list of probabilities; it wants a menu description that helps a customer choose. **Olfacto Sommelier** is the layer that turns aroma into text — with the odor model at the center, not the LLM.

![Olfacto Sommelier demo](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/agent-demo.gif)

## How it works

1. **Dish → ingredients.** A curated catalog maps 45 dishes to ingredients. Users can also add extra ingredients.
2. **Ingredient → SMILES.** Two catalogs (40 sweet and 38 savory ingredients) store the SMILES of a marker molecule for each ingredient.
3. **SMILES → aromas.** The ensemble from the previous article predicts each molecule's odor profile. Probabilities are aggregated by maximum.
4. **Sensory context.** A corpus of 45 original descriptions is queried with TF-IDF, no vector database.
5. **LLM.** Qwen 7B receives the dish, the predicted aromas and the retrieved context. It returns JSON with description, profile and pairings.
6. **Guardrails.** The profile must use only aromas from the predicted list. If the JSON fails or cites another aroma, there is one retry and then a deterministic fallback.

The LLM never decides which aromas a dish has. It receives that decision from the model and writes from it. That is the difference between a text generator and a tool-using agent.

## The corpus

There are 45 dishes across 8 categories: Brazilian, sandwiches, pizza, Japanese, desserts, drinks, healthy and pasta. Each entry has ingredients, reference aromas, a short description and pairings.

The corpus is not ground truth; it is context. Its job is to give the LLM sensory vocabulary and prevent invented gastronomic references. Retrieval is TF-IDF with unigrams and bigrams, top-3 by cosine similarity.

## The LLM with guardrails

The prompt has three rules that matter:

- Use only the aromas from the predicted list.
- Make no health, allergen or food-safety claims.
- Answer only with valid JSON.

The pipeline validates the schema, requires 3 to 5 aromas in the profile, 3 pairings and a declared confidence. If the profile cites an aroma outside the list, the response is rejected. On the second failure, the template takes over:

> "Feijoada with a meat aroma, notes of toasted, sulfurous and a balanced finish."

The fallback is poorer, but predictable. In production, predictability beats a beautiful sentence.

## Results

A 20-dish benchmark, with an LLM judge comparing the two descriptions side by side (LLM vs. template), in randomized A/B order:

| Metric | Value |
|---|---|
| Dishes evaluated | 20 |
| LLM coverage | 95% |
| Profile faithfulness | 100% |
| Description faithfulness | 75% |
| Pairings with 3 unique items | 100% |
| Latency p50 / p95 | 2.9s / 4.9s |
| LLM preference | 75% (15 of 20) |
| Fallback | 1 of 20 |

The most important number is the first: **100% of the structured profile uses only predicted aromas**. Description faithfulness is lower (75%) because the LLM sometimes adds plausible terms — "smoky" in a feijoada, "buttery" in a cheese bread — that are not in the model's top-k. That is a real limitation, not a detail.

The LLM judge preferred the agent's descriptions in 15 of 20 pairs. The template is always faithful, but generic. The agent is more useful, at the cost of latency and an external API.

## What I learned

**The model is the tool; the agent is the product.** The hard part was not calling the LLM. It was designing the contract: what it may say, what it may not, and what happens when it fails.

**Guardrails are the expensive part.** Validating the profile is easy. Validating the description is hard, because natural language has no schema. Automatic term checking is still heuristic; human evaluation remains necessary.

**LLM judges are noisy.** Comparing two descriptions side by side is more reliable than asking for a 1–5 score. Pairwise preference gave a clear signal; absolute scores varied too much.

**Fallback is a feature.** An agent that always returns something predictable is more useful than one that fails with a brilliant sentence.

## Limitations

- The judge is an LLM, not a human panel; scores are for relative comparison.
- The benchmark has 20 dishes and does not cover a full menu.
- The odor model was trained on fragrance catalogues; savory ingredients may have weak predictions.
- Pairings are suggestions, not validated gastronomic recommendations.
- No allergen, health or food-safety claims.

## Reproducibility

- Code: [github.com/cagnani/olfacto-ai](https://github.com/cagnani/olfacto-ai)
- Model: [SamuelCagnani/olfacto-chemberta-138](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138)
- Dataset: [SamuelCagnani/olfacto-odor-138-ptbr](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr)
- Demo: [SamuelCagnani/olfacto-agent](https://huggingface.co/spaces/SamuelCagnani/olfacto-agent)

Run the benchmark with `python scripts/evaluate_agent.py`; tests with `make test`.
