"""Generate images for the Hugging Face articles."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from olfacto.ensemble import EnsemblePredictor
from olfacto.ingredients import load_ingredients
from olfacto.space_logic import FAMILY_COLORS

FAMILY_PT = {
    "floral": "floral",
    "frutado": "frutado",
    "verde": "verde",
    "amadeirado": "amadeirado",
    "especiado": "especiado",
    "doce": "doce",
    "animal": "animal",
    "químico": "químico",
    "terroso": "terroso",
    "alimentar": "alimentar",
    "outro": "outro",
}

GALLERY = ["Café", "Cachaça", "Guaraná", "Canela", "Alho", "Baunilha"]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    properties = font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"
    )
    return ImageFont.truetype(font_manager.findfont(properties), size)


def metrics_chart(output: Path) -> None:
    summary = json.loads(Path("outputs/cv/cv_summary.json").read_text())
    fingerprint = json.loads(Path("outputs/fingerprint/cv_summary.json").read_text())
    ensemble = json.loads(Path("outputs/ensemble/summary.json").read_text())

    rows = [
        ("OpenPOM\n(ensemble de 10)", 0.8872, "#9aa0a6"),
        ("Olfacto Ensemble\n(80% fingerprint)", ensemble["oof"]["macro_roc_auc"], "#4c9f70"),
        ("Morgan + LogReg", fingerprint["oof"]["macro_roc_auc"], "#e6b566"),
        ("ChemBERTa\n(modelo único)", summary["oof"]["macro_roc_auc"], "#c1440e"),
    ]
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]

    figure, axis = plt.subplots(figsize=(11, 6), dpi=150)
    bars = axis.barh(labels, values, color=colors, height=0.58)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#1f2937",
        )
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("ROC-AUC macro (5-fold out-of-fold)", fontsize=12)
    axis.set_title("Comparação de modelos", fontsize=18, fontweight="bold", loc="left")
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=11)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def odor_map(output: Path) -> None:
    points = pd.read_csv("data/map.csv")
    molecules = points[points["source"] == "molécula"]
    if len(molecules) > 2200:
        molecules = molecules.sample(2200, random_state=42)
    ingredients = points[points["source"] == "ingrediente"]

    figure, axis = plt.subplots(figsize=(11, 9), dpi=150)
    for family, color in FAMILY_COLORS.items():
        subset = molecules[molecules["family"] == family]
        if subset.empty:
            continue
        axis.scatter(
            subset["x"],
            subset["y"],
            s=10,
            color=color,
            alpha=0.55,
            linewidths=0,
            label=FAMILY_PT.get(family, family),
        )
    axis.scatter(
        ingredients["x"],
        ingredients["y"],
        s=70,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.1,
        label="ingredientes",
    )
    axis.set_title("Espaço olfativo (UMAP)", fontsize=18, fontweight="bold", loc="left")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        markerscale=1.4,
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def ingredients_grid(predictor: EnsemblePredictor, output: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw

    catalog = load_ingredients("data/brazilian_ingredients.csv")
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=150)
    for axis, name in zip(axes.ravel(), GALLERY, strict=True):
        row = catalog[catalog["name_pt"] == name].iloc[0]
        molecule = Chem.MolFromSmiles(row["smiles"])
        image = Draw.MolToImage(molecule, size=(420, 300))
        axis.imshow(image)
        axis.axis("off")
        predictions = predictor.predict(row["smiles"], top_k=3)
        odds = "\n".join(
            f"{prediction.label_pt_br} {prediction.probability:.0%}"
            for prediction in predictions
        )
        axis.set_title(
            row["name_pt"],
            fontsize=13,
            fontweight="bold",
            loc="left",
            color="#1f2937",
        )
        axis.text(
            0.01,
            0.01,
            odds,
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            color="#1f2937",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
        )
    figure.suptitle(
        "Ingredientes do Brasil e seus aromas previstos",
        fontsize=18,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def thumbnail(output: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw

    width, height = 1300, 650
    image = Image.new("RGB", (width, height), "#f7f9f5")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, height - 14, width, height), fill="#4c9f70")
    draw.rounded_rectangle((820, 90, 1230, 560), radius=28, fill="white")

    molecule = Chem.MolFromSmiles("COC1=C(C=CC(=C1)C=O)O")
    structure = Draw.MolToImage(molecule, size=(390, 330))
    image.paste(structure, (830, 130))

    for index, color in enumerate(["#4c9f70", "#e6b566", "#c1440e", "#8b5e3c"]):
        draw.ellipse((860 + index * 26, 490, 878 + index * 26, 508), fill=color)

    draw.text((70, 120), "Olfacto", font=_font(96, bold=True), fill="#1f2937")
    draw.text((490, 120), "AI", font=_font(96, bold=True), fill="#4c9f70")
    draw.text(
        (74, 250),
        "Do SMILES ao aroma, em português.",
        font=_font(34),
        fill="#4b5563",
    )
    draw.text(
        (74, 330),
        "138 descritores olfativos · 4.983 moléculas",
        font=_font(24),
        fill="#6b7280",
    )
    draw.text(
        (74, 372),
        "ChemBERTa + fingerprints · ensemble aberto",
        font=_font(24),
        fill="#6b7280",
    )
    draw.text(
        (74, 470),
        "github.com/cagnani/olfacto-ai",
        font=_font(22, bold=True),
        fill="#1f2937",
    )
    draw.text(
        (860, 520),
        "vanilina",
        font=_font(24, bold=True),
        fill="#374151",
    )
    image.save(output)


def main() -> int:
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_chart(output_dir / "metrics.png")
    odor_map(output_dir / "odor-map.png")
    thumbnail(output_dir / "thumbnail.png")

    predictor = EnsemblePredictor.from_pretrained(
        "outputs/cv/final",
        "data/odor_labels_ptbr.json",
        "outputs/fingerprint/model.joblib",
        weight_fingerprint=0.8,
    )
    ingredients_grid(predictor, output_dir / "ingredients.png")

    for path in sorted(output_dir.glob("*.png")):
        print(path, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
