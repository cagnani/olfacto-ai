"""Tools that connect dishes and ingredients to the odor model."""

from __future__ import annotations

import json
from pathlib import Path

from olfacto.ingredients import load_ingredients


def load_smiles_map(paths: list[str | Path]) -> dict[str, str]:
    """Map ingredient names to SMILES from one or more catalogs."""
    mapping: dict[str, str] = {}
    for path in paths:
        catalog = load_ingredients(path)
        for row in catalog.itertuples(index=False):
            mapping[row.name_pt] = row.smiles
    return mapping


def load_dishes(path: str | Path) -> dict[str, dict]:
    """Load the dish-to-ingredient mapping, keyed by dish name."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in entries}


def aggregate_predictions(rows: list[dict], top_k: int = 8) -> list[dict]:
    """Aggregate per-ingredient predictions by taking the maximum probability."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    best: dict[str, float] = {}
    names: dict[str, str] = {}
    for row in rows:
        for prediction in row["predictions"]:
            best[prediction.label] = max(
                best.get(prediction.label, 0.0), prediction.probability
            )
            names[prediction.label] = prediction.label_pt_br

    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            "label": label,
            "label_pt_br": names[label],
            "probability": float(probability),
        }
        for label, probability in ranked
    ]


class OdorTools:
    """Resolve a dish to ingredients and an aggregated aroma profile."""

    def __init__(
        self,
        predictor,
        catalog_paths: list[str | Path],
        dishes_path: str | Path,
    ) -> None:
        self.predictor = predictor
        self.smiles_by_name = load_smiles_map(catalog_paths)
        self.dishes = load_dishes(dishes_path)

    def profile(
        self,
        dish: str,
        extra_ingredients: list[str] | None = None,
        top_k: int = 8,
    ) -> dict:
        if extra_ingredients:
            ingredients = list(extra_ingredients)
            category = self.dishes.get(dish, {}).get("category", "Outro")
        elif dish in self.dishes:
            ingredients = self.dishes[dish]["ingredients"]
            category = self.dishes[dish]["category"]
        else:
            raise ValueError(f"unknown dish: {dish}")

        rows = []
        missing = []
        for name in ingredients:
            smiles = self.smiles_by_name.get(name)
            if not smiles:
                missing.append(name)
                continue
            predictions = self.predictor.predict(smiles, top_k=10)
            rows.append({"name": name, "smiles": smiles, "predictions": predictions})

        return {
            "dish": dish,
            "category": category,
            "ingredients": rows,
            "missing": missing,
            "aromas": aggregate_predictions(rows, top_k=top_k),
        }
