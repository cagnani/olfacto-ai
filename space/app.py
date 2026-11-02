"""Olfacto AI — Hugging Face Space."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download

from olfacto.ensemble import EnsemblePredictor
from olfacto.ingredients import load_ingredients
from olfacto.labels import load_lexicon
from olfacto.predict import OdorPredictor
from olfacto.space_logic import (
    find_ingredient,
    odor_wheel_svg,
    predictions_table,
    scatter_svg,
)

MODEL_ID = os.environ.get("OLFACTO_MODEL_ID", "SamuelCagnani/olfacto-chemberta-138")
DATASET_ID = os.environ.get("OLFACTO_DATASET_ID", "SamuelCagnani/olfacto-odor-138-ptbr")

EXAMPLES = [
    ["CCO", "etanol"],
    ["COC1=C(C=CC(=C1)C=O)O", "vanilina"],
    ["CC1=CCC(CC1)C(=C)C", "limoneno"],
    ["C1=CC=C(C=C1)/C=C/C=O", "cinamaldeído"],
    ["COC1=CC=CC=C1O", "guaiacol"],
]


def asset(filename: str) -> str:
    local = Path("data") / filename
    if local.is_file():
        return str(local)
    return hf_hub_download(repo_id=DATASET_ID, filename=filename, repo_type="dataset")


def model_asset(filename: str) -> str:
    candidates = [Path("outputs/fingerprint") / filename, Path("data") / filename]
    if filename == "fingerprint.joblib":
        candidates.insert(0, Path("outputs/fingerprint/model.joblib"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return hf_hub_download(repo_id=MODEL_ID, filename=filename)


def molecule_image(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return Draw.MolToImage(molecule, size=(360, 260))
    except ImportError:
        return None


def load_map() -> pd.DataFrame | None:
    try:
        return pd.read_csv(asset("map.csv"))
    except Exception:
        return None


LEXICON_PATH = asset("odor_labels_ptbr.json")
LEXICON = load_lexicon(LEXICON_PATH)
INGREDIENTS = load_ingredients(asset("brazilian_ingredients.csv"))
MAP = load_map()

try:
    FINGERPRINT_PATH = model_asset("fingerprint.joblib")
    PREDICTOR = EnsemblePredictor.from_pretrained(
        MODEL_ID, LEXICON_PATH, FINGERPRINT_PATH, weight_fingerprint=0.8
    )
except Exception:
    PREDICTOR = OdorPredictor.from_pretrained(MODEL_ID, LEXICON_PATH)


def predict_molecule(smiles: str, top_k: int):
    if not smiles or not smiles.strip():
        raise gr.Error("Digite um SMILES.")
    try:
        predictions = PREDICTOR.predict(smiles.strip(), int(top_k))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return (
        predictions_table(predictions),
        odor_wheel_svg(predictions, LEXICON),
        molecule_image(smiles),
    )


def predict_ingredient(name: str, top_k: int):
    row = find_ingredient(INGREDIENTS, name)
    if row is None:
        raise gr.Error("Ingrediente não encontrado.")
    smiles = row["smiles"]
    predictions = PREDICTOR.predict(smiles, int(top_k))
    info = (
        f"### {row['name_pt']}\n"
        f"**{row['name_en']}** · molécula marcante: *{row['pubchem_name']}*\n\n"
        f"`{smiles}`\n\n_{row['note']}_"
    )
    return (
        info,
        predictions_table(predictions),
        odor_wheel_svg(predictions, LEXICON),
        molecule_image(smiles),
    )


with gr.Blocks(title="Olfacto AI") as demo:
    gr.Markdown(
        "# Olfacto AI\n"
        "Do SMILES ao aroma, em português. Modelo aberto de predição de odor molecular "
        "treinado sobre o dataset do OpenPOM.\n\n"
        "> Prevê **qualidade** olfativa de moléculas únicas, não intensidade nem misturas."
    )

    with gr.Tab("Molécula"):
        with gr.Row():
            with gr.Column():
                smiles_input = gr.Textbox(label="SMILES", value="CCO")
                top_k = gr.Slider(3, 20, value=10, step=1, label="Quantos aromas?")
                predict_button = gr.Button("Prever aromas", variant="primary")
                structure = gr.Image(label="Estrutura", type="pil")
            with gr.Column():
                wheel = gr.HTML(label="Roda olfativa")
                table = gr.Dataframe(
                    headers=["aroma", "probabilidade", "rótulo"],
                    datatype=["str", "number", "str"],
                    interactive=False,
                    label="Top aromas",
                )
        gr.Examples(EXAMPLES, inputs=[smiles_input])
        predict_button.click(
            predict_molecule,
            inputs=[smiles_input, top_k],
            outputs=[table, wheel, structure],
        )

    with gr.Tab("Ingredientes do Brasil"):
        gr.Markdown("Moléculas marcantes de ingredientes brasileiros, com SMILES do PubChem.")
        with gr.Row():
            with gr.Column():
                ingredient = gr.Dropdown(
                    sorted(INGREDIENTS["name_pt"].tolist()),
                    value="Café",
                    label="Ingrediente",
                )
                ingredient_top_k = gr.Slider(3, 20, value=10, step=1, label="Quantos aromas?")
                ingredient_button = gr.Button("Ver aromas", variant="primary")
                ingredient_info = gr.Markdown()
            with gr.Column():
                ingredient_wheel = gr.HTML()
                ingredient_table = gr.Dataframe(
                    headers=["aroma", "probabilidade", "rótulo"],
                    datatype=["str", "number", "str"],
                    interactive=False,
                )
        ingredient_button.click(
            predict_ingredient,
            inputs=[ingredient, ingredient_top_k],
            outputs=[ingredient_info, ingredient_table, ingredient_wheel],
        )

    with gr.Tab("Mapa de cheiros"):
        gr.Markdown("Projeção UMAP dos embeddings do modelo, colorida por família olfativa.")
        if MAP is not None:
            gr.HTML(scatter_svg(MAP))
        else:
            gr.Markdown("Mapa em geração — será publicado junto com o dataset.")

if __name__ == "__main__":
    demo.launch()
