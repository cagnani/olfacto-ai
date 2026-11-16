"""Evaluate the Olfacto Sommelier on a 20-dish benchmark."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient, get_token

from olfacto.agent.evaluate import parse_preference, render_eval
from olfacto.agent.pipeline import SommelierAgent, make_inference_llm
from olfacto.agent.prompts import fallback_response
from olfacto.agent.retrieval import SensoryRetriever, load_corpus
from olfacto.agent.tools import OdorTools
from olfacto.ensemble import EnsemblePredictor
from olfacto.labels import load_lexicon

GENERIC_TERMS = {
    "fresco", "doce", "amargo", "ácido", "salgado", "verde", "floral", "frutado",
    "amadeirado", "especiado", "herbal", "oleoso", "cremoso", "quente", "seco",
    "limpo", "natural", "suave", "intenso", "equilibrado", "leve", "refrescante",
    "delicado", "marcante", "pronunciado", "agradável",
}

DISHES = [
    "Feijoada",
    "Moqueca de peixe",
    "Acarajé",
    "Pão de queijo",
    "Churrasco",
    "Hambúrguer",
    "Batata frita",
    "Pastel",
    "Pizza margherita",
    "Pizza calabresa",
    "Sushi de salmão",
    "Yakisoba",
    "Brigadeiro",
    "Açaí",
    "Café",
    "Caipirinha",
    "Salada",
    "Frango grelhado",
    "Lasanha",
    "Nhoque",
]

MODEL_ID = "SamuelCagnani/olfacto-chemberta-138"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_PROVIDER = "featherless-ai"
PAIRWISE_PROMPT = """Qual descrição de menu é melhor para o prato abaixo? Considere
clareza, linguagem sensorial, coerência com os aromas e utilidade para um restaurante.

Prato: {dish}
Aromas previstos: {aromas}

Opção A: {option_a}
Opção B: {option_b}

Responda somente com JSON:
{{"vencedor": "A" ou "B" ou "empate", "justificativa": "até 20 palavras"}}
"""


def judge_pairwise(
    client, dish: str, aromas: list[dict], option_a: str, option_b: str
) -> str | None:
    prompt = PAIRWISE_PROMPT.format(
        dish=dish,
        aromas=", ".join(item["label_pt_br"] for item in aromas),
        option_a=option_a,
        option_b=option_b,
    )
    for attempt in range(3):
        try:
            response = client.chat_completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            preference = parse_preference(response.choices[0].message.content or "")
            if preference is not None:
                return preference
        except Exception:
            pass
        time.sleep(2**attempt)
    return None


def unexpected_terms(
    text: str, allowed: set[str], ingredients: list[str], lexicon: dict
) -> set[str]:
    known = {entry["pt_br"].lower() for entry in lexicon.values() if entry["pt_br"]}
    mentioned = {term for term in known if len(term) >= 5 and term in text.lower()}
    return mentioned - allowed - {item.lower() for item in ingredients} - GENERIC_TERMS


def main() -> int:
    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face token configured")

    predictor = EnsemblePredictor.from_pretrained(
        "outputs/cv/final",
        "data/odor_labels_ptbr.json",
        "outputs/fingerprint/model.joblib",
        weight_fingerprint=0.8,
    )
    tools = OdorTools(
        predictor,
        [
            "data/brazilian_ingredients.csv",
            "data/sommelier_ingredients.csv",
        ],
        "data/dish_ingredients.json",
    )
    retriever = SensoryRetriever(load_corpus("data/sensory_corpus.jsonl"))
    lexicon = load_lexicon("data/odor_labels_ptbr.json")

    client = InferenceClient(model=LLM_MODEL, provider=LLM_PROVIDER, token=token)
    agent = SommelierAgent(tools, retriever, make_inference_llm(client, LLM_MODEL))

    rows = []
    rng = random.Random(42)
    for dish in DISHES:
        start = time.perf_counter()
        result = agent.generate(dish)
        latency = time.perf_counter() - start

        allowed = {item["label_pt_br"].lower() for item in result.aromas}
        profile_faithful = all(
            item.lower() in allowed for item in result.sensory_profile
        )
        description_faithful = not unexpected_terms(
            result.description,
            allowed,
            tools.dishes[dish]["ingredients"],
            lexicon,
        )
        pairing_unique = len(set(result.pairings)) == 3

        context_pairings = result.context[0]["pairings"] if result.context else []
        template = fallback_response(dish, result.aromas, context_pairings)

        if rng.random() < 0.5:
            option_a, option_b = result.description, template["descricao"]
            mapping = {"A": "llm", "B": "template"}
        else:
            option_a, option_b = template["descricao"], result.description
            mapping = {"A": "template", "B": "llm"}

        winner = judge_pairwise(client, dish, result.aromas, option_a, option_b)
        preference = mapping.get(winner, "empate") if winner in {"A", "B"} else winner
        time.sleep(1.0)

        rows.append(
            {
                "dish": dish,
                "category": result.category,
                "source": result.source,
                "latency_s": latency,
                "profile_faithful": profile_faithful,
                "description_faithful": description_faithful,
                "pairing_unique": pairing_unique,
                "preference": preference,
                "aromas": [item["label_pt_br"] for item in result.aromas],
                "description": result.description,
                "template_description": template["descricao"],
            }
        )
        print(
            f"{dish:<22} source={result.source:<8} "
            f"faithful={description_faithful} "
            f"preference={preference} "
            f"{latency:.2f}s",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    llm_wins = int((frame["preference"] == "llm").sum())
    template_wins = int((frame["preference"] == "template").sum())
    ties = int((frame["preference"] == "empate").sum())
    decisive = llm_wins + template_wins
    latencies = frame["latency_s"].to_numpy()
    p50, p95 = np.percentile(latencies, [50, 95])

    metrics = {
        "n": len(frame),
        "llm_coverage": float((frame["source"] == "llm").mean()),
        "profile_faithfulness": float(frame["profile_faithful"].mean()),
        "description_faithfulness": float(frame["description_faithful"].mean()),
        "pairing_diversity": float(frame["pairing_unique"].mean()),
        "latency_p50": float(p50),
        "latency_p95": float(p95),
        "llm_wins": llm_wins,
        "template_wins": template_wins,
        "ties": ties,
        "llm_win_rate": float(llm_wins / decisive) if decisive else 0.0,
        "judge_pairs": decisive + ties,
        "fallback_count": int((frame["source"] == "fallback").sum()),
    }

    output_dir = Path("outputs/agent_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path("docs/agent-eval.md").write_text(render_eval(metrics), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
