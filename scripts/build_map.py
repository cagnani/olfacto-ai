"""Build the 2D olfactory map from the final checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from olfacto.data import SMILES_COLUMN, get_label_columns, load_odor_dataset
from olfacto.ingredients import load_ingredients
from olfacto.labels import load_lexicon
from olfacto.map import dominant_family, extract_embeddings, project_umap
from olfacto.train import predict_proba


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/cv/final"))
    parser.add_argument(
        "--data", type=Path, default=Path("data/curated_GS_LF_merged_4983.csv")
    )
    parser.add_argument(
        "--lexicon", type=Path, default=Path("data/odor_labels_ptbr.json")
    )
    parser.add_argument(
        "--ingredients", type=Path, default=Path("data/brazilian_ingredients.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/map"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N molecules")
    args = parser.parse_args()

    df = load_odor_dataset(args.data)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    labels = get_label_columns(df)
    lexicon = load_lexicon(args.lexicon)
    families = [lexicon[label]["family"] for label in labels]
    ingredients = load_ingredients(args.ingredients)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    molecule_smiles = df[SMILES_COLUMN].tolist()
    ingredient_smiles = ingredients["smiles"].tolist()
    all_smiles = molecule_smiles + ingredient_smiles

    print(f"extracting embeddings for {len(all_smiles)} molecules", flush=True)
    embeddings = extract_embeddings(model, tokenizer, all_smiles, max_length=args.max_length)

    print("projecting with UMAP", flush=True)
    coords = project_umap(embeddings)

    molecule_families = [
        dominant_family(row, families) for row in df[labels].to_numpy(dtype=float)
    ]
    ingredient_scores = predict_proba(
        model, tokenizer, ingredient_smiles, args.max_length
    )
    ingredient_families = [
        dominant_family(row, families) for row in ingredient_scores
    ]

    map_df = pd.DataFrame(
        {
            "name": molecule_smiles + ingredients["name_pt"].tolist(),
            "family": molecule_families + ingredient_families,
            "source": ["molécula"] * len(molecule_smiles) + ["ingrediente"] * len(ingredients),
            "smiles": all_smiles,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_df.to_csv(args.output_dir / "map.csv", index=False)
    np.save(args.output_dir / "embeddings.npy", embeddings[: len(molecule_smiles)])

    data_copy = Path("data/map.csv")
    if args.limit is None and data_copy.parent.is_dir():
        map_df.to_csv(data_copy, index=False)

    print(f"saved {len(map_df)} points to {args.output_dir / 'map.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
