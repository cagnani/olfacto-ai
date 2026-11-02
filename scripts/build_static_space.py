"""Build a static Hugging Face Space with precomputed predictions."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from olfacto.ensemble import EnsemblePredictor
from olfacto.ingredients import load_ingredients
from olfacto.space_logic import FAMILY_COLORS, scatter_svg

SELECTED_INGREDIENTS = [
    "Café",
    "Cachaça",
    "Guaraná",
    "Cupuaçu",
    "Maracujá",
    "Abacaxi",
    "Manga",
    "Goiaba",
    "Coco",
    "Cacau",
    "Canela",
    "Cravo",
    "Baunilha",
    "Pimenta-do-reino",
    "Alho",
    "Cebola",
    "Hortelã",
    "Camomila",
    "Pequi",
    "Jabuticaba",
    "Pitanga",
]


def molecule_svg(smiles: str, width: int = 220, height: int = 150) -> str:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().split("?>", 1)[-1]


def metric_cards(summary: dict, fingerprint: dict, ensemble: dict) -> str:
    oof = summary["oof"]
    rows = [
        ("ChemBERTa", oof["macro_roc_auc"], "modelo único"),
        ("Morgan + LogReg", fingerprint["oof"]["macro_roc_auc"], "baseline clássico"),
        ("Ensemble", ensemble["oof"]["macro_roc_auc"], "80% fingerprint + 20% ChemBERTa"),
        ("OpenPOM", 0.8872, "ensemble de 10 MPNNs (paper)"),
    ]
    cards = []
    for name, value, note in rows:
        highlight = " highlight" if name == "Ensemble" else ""
        cards.append(
            f'<div class="metric{highlight}"><strong>{value:.4f}</strong>'
            f"<span>{html.escape(name)}</span><small>{html.escape(note)}</small></div>"
        )
    return "".join(cards)


def ingredient_cards(
    predictor: EnsemblePredictor,
    ingredients: pd.DataFrame,
    selected: list[str],
    top_k: int = 5,
) -> str:
    cards = []
    for name in selected:
        match = ingredients[ingredients["name_pt"] == name]
        if match.empty:
            continue
        row = match.iloc[0]
        predictions = predictor.predict(row["smiles"], top_k=top_k)
        items = "".join(
            f'<li><span>{html.escape(prediction.label_pt_br)}</span>'
            f'<i style="--w:{prediction.probability:.0%}"></i>'
            f"<b>{prediction.probability:.0%}</b></li>"
            for prediction in predictions
        )
        cards.append(
            f'<article class="ingredient"><div class="structure">'
            f"{molecule_svg(row['smiles'])}</div>"
            f"<h3>{html.escape(row['name_pt'])}</h3>"
            f'<p class="meta">{html.escape(row["name_en"])} · '
            f"<em>{html.escape(row['pubchem_name'])}</em></p>"
            f'<ul class="odds">{items}</ul></article>'
        )
    return "".join(cards)


def legend() -> str:
    return "".join(
        f'<span><i style="background:{color}"></i>{html.escape(family)}</span>'
        for family, color in FAMILY_COLORS.items()
    )


def build_html(
    metrics: str,
    cards: str,
    map_svg: str,
    legend_html: str,
    n_points: int,
) -> str:
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Olfacto AI — do SMILES ao aroma</title>
<style>
:root {{ --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; --paper:#ffffff;
  --bg:#f8faf7; --green:#4c9f70; --amber:#e6b566; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--bg); color:var(--ink); line-height:1.6; }}
main {{ max-width:1100px; margin:0 auto; padding:48px 20px 72px; }}
header {{ padding:12px 0 28px; }}
h1 {{ font-size:clamp(2rem,5vw,3.4rem); line-height:1.05; margin:0 0 12px;
  letter-spacing:-.03em; }}
h1 span {{ color:var(--green); }}
.lead {{ font-size:1.15rem; color:var(--muted); max-width:720px; }}
.links a {{ color:var(--green); font-weight:600; text-decoration:none; margin-right:16px; }}
.links a:hover {{ text-decoration:underline; }}
section {{ margin-top:52px; }}
h2 {{ font-size:1.5rem; margin:0 0 8px; }}
.note {{ color:var(--muted); font-size:.95rem; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  gap:12px; margin-top:18px; }}
.metric {{ background:var(--paper); border:1px solid var(--line); border-radius:14px;
  padding:16px; display:flex; flex-direction:column; }}
.metric.highlight {{ border-color:var(--green); box-shadow:0 0 0 2px #4c9f7022; }}
.metric strong {{ font-size:1.7rem; letter-spacing:-.02em; }}
.metric span {{ font-weight:600; }}
.metric small {{ color:var(--muted); }}
.map {{ background:var(--paper); border:1px solid var(--line); border-radius:16px;
  padding:12px; overflow:hidden; }}
.map svg {{ width:100%; height:auto; display:block; }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px 18px; margin-top:12px;
  color:var(--muted); font-size:.85rem; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%;
  margin-right:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
  gap:14px; margin-top:18px; }}
.ingredient {{ background:var(--paper); border:1px solid var(--line);
  border-radius:16px; padding:16px; }}
.structure {{ background:#fbfcfb; border-radius:10px; min-height:150px;
  display:flex; align-items:center; justify-content:center; }}
.structure svg {{ width:100%; height:auto; }}
.ingredient h3 {{ margin:12px 0 2px; font-size:1.1rem; }}
.meta {{ margin:0 0 10px; color:var(--muted); font-size:.82rem; }}
.odds {{ list-style:none; margin:0; padding:0; display:grid; gap:6px; }}
.odds li {{ display:grid; grid-template-columns:1fr 90px 38px; align-items:center;
  gap:8px; font-size:.85rem; }}
.odds span {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.odds i {{ display:block; height:7px; border-radius:99px; background:var(--line);
  position:relative; }}
.odds i::after {{ content:""; position:absolute; inset:0; width:var(--w);
  background:linear-gradient(90deg,var(--green),var(--amber)); border-radius:99px; }}
.odds b {{ text-align:right; color:var(--muted); font-weight:600; }}
pre {{ background:#111827; color:#e5e7eb; padding:16px; border-radius:12px;
  overflow:auto; font-size:.85rem; }}
footer {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--line);
  color:var(--muted); font-size:.9rem; }}
@media (max-width:600px) {{ main {{ padding-top:28px; }} }}
</style>
</head>
<body>
<main>
<header>
  <h1>Olfacto <span>AI</span></h1>
  <p class="lead">Do SMILES ao aroma, em português. Um modelo aberto que prevê 138
  descritores olfativos a partir da estrutura de uma molécula — com uma galeria de
  ingredientes brasileiros e um mapa do espaço olfativo.</p>
  <p class="links">
    <a href="https://github.com/cagnani/olfacto-ai">GitHub</a>
    <a href="https://huggingface.co/SamuelCagnani/olfacto-chemberta-138">Modelo</a>
    <a href="https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr">Dataset</a>
  </p>
</header>

<section>
  <h2>Resultados (5-fold out-of-fold)</h2>
  <p class="note">ROC-AUC macro. O ensemble é o que gera as previsões desta página.</p>
  <div class="metrics">{metrics}</div>
</section>

<section>
  <h2>Mapa de cheiros</h2>
  <p class="note">Projeção UMAP dos embeddings, colorida por família olfativa.
  Amostra de {n_points:,} pontos; os ingredientes brasileiros estão incluídos.</p>
  <div class="map">{map_svg}</div>
  <div class="legend">{legend_html}</div>
</section>

<section>
  <h2>Ingredientes do Brasil</h2>
  <p class="note">Moléculas marcantes de ingredientes brasileiros e os aromas previstos
  pelo ensemble. Estruturas renderizadas com RDKit.</p>
  <div class="grid">{cards}</div>
</section>

<section>
  <h2>Como funciona</h2>
  <p>O modelo recebe um SMILES — a representação textual da molécula — e retorna a
  probabilidade de 138 descritores. O ensemble combina um ChemBERTa fine-tunado com um
  baseline de fingerprints Morgan + regressão logística. O peso de 80% para o fingerprint
  foi escolhido por validação out-of-fold.</p>
  <pre><code>from olfacto.ensemble import EnsemblePredictor

predictor = EnsemblePredictor.from_pretrained(
    "SamuelCagnani/olfacto-chemberta-138",
    "data/odor_labels_ptbr.json",
    "fingerprint.joblib",
)
for p in predictor.predict("CCO", top_k=5):
    print(p.label_pt_br, round(p.probability, 3))</code></pre>
</section>

<footer>
  <p><strong>Limitações:</strong> prevê qualidade olfativa, não intensidade; apenas
  moléculas únicas, não misturas; rótulos de catálogo têm ruído e viés para fragrâncias
  comerciais. Esta é uma demo estática com previsões pré-computadas — o app Gradio
  completo está no GitHub.</p>
  <p>Dados: OpenPOM / GoodScents / Leffingwell. Código MIT.</p>
</footer>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/cv/final"))
    parser.add_argument(
        "--fingerprint", type=Path, default=Path("outputs/fingerprint/model.joblib")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/curated_GS_LF_merged_4983.csv")
    )
    parser.add_argument(
        "--ingredients", type=Path, default=Path("data/brazilian_ingredients.csv")
    )
    parser.add_argument(
        "--lexicon", type=Path, default=Path("data/odor_labels_ptbr.json")
    )
    parser.add_argument("--map", type=Path, default=Path("data/map.csv"))
    parser.add_argument(
        "--cv-summary", type=Path, default=Path("outputs/cv/cv_summary.json")
    )
    parser.add_argument(
        "--fingerprint-summary",
        type=Path,
        default=Path("outputs/fingerprint/cv_summary.json"),
    )
    parser.add_argument(
        "--ensemble-summary",
        type=Path,
        default=Path("outputs/ensemble/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("space_static"))
    parser.add_argument("--max-points", type=int, default=2500)
    args = parser.parse_args()

    ingredients = load_ingredients(args.ingredients)
    predictor = EnsemblePredictor.from_pretrained(
        args.model_dir, args.lexicon, args.fingerprint, weight_fingerprint=0.8
    )

    map_df = pd.read_csv(args.map)
    molecules = map_df[map_df["source"] == "molécula"]
    if len(molecules) > args.max_points:
        molecules = molecules.sample(args.max_points, random_state=42)
    map_sample = pd.concat([molecules, map_df[map_df["source"] == "ingrediente"]])
    map_svg = scatter_svg(map_sample, size=760)

    summary = json.loads(args.cv_summary.read_text(encoding="utf-8"))
    fingerprint = json.loads(args.fingerprint_summary.read_text(encoding="utf-8"))
    ensemble = json.loads(args.ensemble_summary.read_text(encoding="utf-8"))

    page = build_html(
        metric_cards(summary, fingerprint, ensemble),
        ingredient_cards(predictor, ingredients, SELECTED_INGREDIENTS),
        map_svg,
        legend(),
        len(map_sample),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        """---
title: Olfacto AI
emoji: 👃
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Olfacto AI

Demo estática do Olfacto: mapa olfativo, ingredientes brasileiros e previsões
pré-computadas. O app Gradio completo está no
[GitHub](https://github.com/cagnani/olfacto-ai).
""",
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
