"""Curated Brazilian ingredients and their marker aroma molecules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INGREDIENT_COLUMNS = ["name_pt", "name_en", "category", "pubchem_name", "smiles", "note"]


def load_ingredients(path: str | Path) -> pd.DataFrame:
    """Load the curated ingredient catalog as strings."""
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in INGREDIENT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"ingredient catalog is missing columns: {missing}")
    return df[INGREDIENT_COLUMNS]


def validate_ingredients(df: pd.DataFrame) -> None:
    """Validate that the catalog is complete and unambiguous."""
    if df["name_pt"].str.strip().eq("").any():
        raise ValueError("ingredient names must not be empty")
    if df["name_pt"].duplicated().any():
        raise ValueError("ingredient catalog contains duplicate names")
    if df["smiles"].str.strip().eq("").any():
        raise ValueError("all ingredients must have a SMILES string")


def parse_pubchem_response(payload: dict) -> str | None:
    """Extract the isomeric SMILES from a PubChem PUG REST property response."""
    try:
        properties = payload["PropertyTable"]["Properties"]
        if not properties:
            return None
        entry = properties[0]
    except (KeyError, TypeError):
        return None

    return entry.get("IsomericSMILES") or entry.get("SMILES") or None
