"""Olfacto Sommelier: menu descriptions grounded on odor predictions."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import InferenceClient, get_token, hf_hub_download

from olfacto.agent.pipeline import SommelierAgent, make_inference_llm
from olfacto.agent.retrieval import SensoryRetriever, load_corpus
from olfacto.agent.tools import OdorTools
from olfacto.ensemble import EnsemblePredictor

MODEL_ID = os.environ.get("OLFACTO_MODEL_ID", "SamuelCagnani/olfacto-chemberta-138")
DATASET_ID = os.environ.get("OLFACTO_DATASET_ID", "SamuelCagnani/olfacto-odor-138-ptbr")
LLM_MODEL = os.environ.get("OLFACTO_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_PROVIDER = os.environ.get("OLFACTO_LLM_PROVIDER", "featherless-ai")


def data_asset(name: str) -> str:
    local = Path("data") / name
    if local.is_file():
        return str(local)
    return hf_hub_download(DATASET_ID, f"agent/{name}", repo_type="dataset")


def root_asset(name: str) -> str:
    local = Path("data") / name
    if local.is_file():
        return str(local)
    return hf_hub_download(DATASET_ID, name, repo_type="dataset")


def model_asset(name: str) -> str:
    candidates = [Path("outputs/fingerprint") / name, Path("data") / name]
    if name == "fingerprint.joblib":
        candidates.insert(0, Path("outputs/fingerprint/model.joblib"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return hf_hub_download(MODEL_ID, name)


LEXICON_PATH = root_asset("odor_labels_ptbr.json")
PREDICTOR = EnsemblePredictor.from_pretrained(
    MODEL_ID,
    LEXICON_PATH,
    model_asset("fingerprint.joblib"),
    weight_fingerprint=0.8,
)
TOOLS = OdorTools(
    PREDICTOR,
    [
        root_asset("brazilian_ingredients.csv"),
        data_asset("sommelier_ingredients.csv"),
    ],
    data_asset("dish_ingredients.json"),
)
RETRIEVER = SensoryRetriever(load_corpus(data_asset("sensory_corpus.jsonl")))
DISHES = sorted(TOOLS.dishes)

TOKEN = os.environ.get("HF_TOKEN") or get_token()
CLIENT = (
    InferenceClient(model=LLM_MODEL, provider=LLM_PROVIDER, token=TOKEN)
    if TOKEN
    else None
)
LLM_CALL = make_inference_llm(CLIENT, LLM_MODEL) if CLIENT else None


def call_llm(messages: list[dict]) -> str:
    if LLM_CALL is None:
        raise RuntimeError("HF_TOKEN is not configured")
    return LLM_CALL(messages)


AGENT = SommelierAgent(TOOLS, RETRIEVER, call_llm)


def describe(dish: str, extra: str, top_k: float):
    extras = [item.strip() for item in extra.split(",") if item.strip()] or None
    try:
        result = AGENT.generate(dish, extras, top_k=int(top_k))
    except ValueError as error:
        raise gr.Error(str(error)) from error

    profile = pd.DataFrame(
        [
            {
                "aroma": item["label_pt_br"],
                "probabilidade": round(item["probability"], 3),
            }
            for item in result.aromas
        ]
    )
    pairings = "\n".join(f"- {item}" for item in result.pairings)
    source = (
        f"Fonte: **{result.source}** · confiança: **{result.confidence}**"
        if result.source == "llm"
        else f"Fonte: **{result.source}** · confiança: **{result.confidence}** "
        "(fallback determinístico)"
    )
    ingredients = ", ".join(result.ingredients) or "não informados"
    return result.description, profile, pairings, source, ingredients


with gr.Blocks(title="Olfacto Sommelier") as demo:
    gr.Markdown(
        "# Olfacto Sommelier\n"
        "Descrições sensoriais para menus, a partir de aromas previstos pelo modelo "
        "de odor molecular. O LLM **não inventa aromas**: ele recebe a lista prevista "
        "e apenas redige o texto.\n\n"
        "[Modelo](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138) · "
        "[Dataset](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr) · "
        "[Código](https://github.com/cagnani/olfacto-ai)"
    )

    with gr.Row():
        with gr.Column(scale=4):
            dish = gr.Dropdown(DISHES, value="Feijoada", label="Prato")
            extra = gr.Textbox(
                label="Ingredientes extras (opcional)",
                placeholder="ex.: bacon, cebola",
            )
            top_k = gr.Slider(4, 12, value=8, step=1, label="Aromas no perfil")
            button = gr.Button("Gerar descrição", variant="primary")
            gr.Examples(
                [["Feijoada", "", 8], ["Pizza margherita", "", 8], ["Café", "", 6]],
                inputs=[dish, extra, top_k],
            )

        with gr.Column(scale=6):
            description = gr.Textbox(label="Descrição para o menu", lines=4)
            profile = gr.Dataframe(
                headers=["aroma", "probabilidade"],
                datatype=["str", "number"],
                interactive=False,
                label="Perfil sensorial",
            )
            pairings = gr.Markdown(label="Harmonizações")
            source = gr.Markdown()
            ingredients = gr.Textbox(label="Ingredientes usados", interactive=False)

    button.click(
        describe,
        inputs=[dish, extra, top_k],
        outputs=[description, profile, pairings, source, ingredients],
    )

    gr.Markdown(
        "> Protótipo de pesquisa. Não faz alegações de alérgenos, saúde ou segurança "
        "alimentar. As previsões vêm de um modelo treinado em catálogos de fragrância."
    )

if __name__ == "__main__":
    demo.launch()
