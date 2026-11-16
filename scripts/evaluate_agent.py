"""Evaluate the Olfacto Sommelier on a 20-dish benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient, get_token

from olfacto.agent.evaluate import parse_judge, render_eval
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
JUDGE_PROMPT = """Avalie a descrição de menu abaixo de 1 a 5, considerando clareza,
linguagem sensorial e utilidade para um restaurante.

Escala:
1 = ruim: genérica ou sem relação com os aromas
2 = fraca: clara, mas pouco sensorial
3 = aceitável: clara e coerente, mas básica
4 = boa: sensorial, específica e útil
5 = excelente: apetitosa, precisa e pronta para menu

Prato: {dish}
Aromas previstos: {aromas}

Descrição: {description}

Responda somente com JSON: {{"nota": 1, "justificativa": "..."}}
"""


def judge(client, dish: str, aromas: list[dict], description: str) -> int | None:
    prompt = JUDGE_PROMPT.format(
        dish=dish,
        aromas=", ".join(item["label_pt_br"] for item in aromas),
        description=description,
    )
    for _ in range(2):
        try:
            response = client.chat_completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            score = parse_judge(response.choices[0].message.content or "")
            if score is not None:
                return score
        except Exception:
            time.sleep(1.0)
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
        llm_score = judge(client, dish, result.aromas, result.description)
        time.sleep(0.4)
        template_score = judge(client, dish, result.aromas, template["descricao"])
        time.sleep(0.4)

        rows.append(
            {
                "dish": dish,
                "category": result.category,
                "source": result.source,
                "latency_s": latency,
                "profile_faithful": profile_faithful,
                "description_faithful": description_faithful,
                "pairing_unique": pairing_unique,
                "judge_llm": llm_score,
                "judge_template": template_score,
                "aromas": [item["label_pt_br"] for item in result.aromas],
                "description": result.description,
                "template_description": template["descricao"],
            }
        )
        print(
            f"{dish:<22} source={result.source:<8} "
            f"faithful={description_faithful} "
            f"judge={llm_score}/{template_score} "
            f"{latency:.2f}s",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    llm_scores = frame["judge_llm"].dropna().astype(float)
    template_scores = frame["judge_template"].dropna().astype(float)
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
        "judge_llm_mean": float(llm_scores.mean()) if len(llm_scores) else 0.0,
        "judge_llm_std": float(llm_scores.std()) if len(llm_scores) > 1 else 0.0,
        "judge_template_mean": float(template_scores.mean())
        if len(template_scores)
        else 0.0,
        "judge_template_std": float(template_scores.std())
        if len(template_scores) > 1
        else 0.0,
        "judge_pairs": int(
            ((frame["judge_llm"].notna()) & (frame["judge_template"].notna())).sum()
        ),
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
