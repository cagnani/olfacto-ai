from pathlib import Path

import pandas as pd
import pytest

from olfacto.ingredients import (
    INGREDIENT_COLUMNS,
    load_ingredients,
    parse_pubchem_response,
    validate_ingredients,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "brazilian_ingredients.csv"


def test_ingredient_catalog_schema():
    df = load_ingredients(CATALOG_PATH)

    assert list(df.columns) == INGREDIENT_COLUMNS
    assert len(df) >= 30
    assert not df["name_pt"].duplicated().any()


def test_ingredient_catalog_has_filled_smiles():
    df = load_ingredients(CATALOG_PATH)

    validate_ingredients(df)
    assert df["smiles"].str.strip().ne("").all()


def test_parse_pubchem_response():
    payload = {
        "PropertyTable": {"Properties": [{"CID": 702, "IsomericSMILES": "CCO"}]}
    }

    assert parse_pubchem_response(payload) == "CCO"


def test_parse_empty_or_malformed_response():
    assert parse_pubchem_response({"PropertyTable": {"Properties": []}}) is None
    assert parse_pubchem_response({}) is None
    assert parse_pubchem_response({"PropertyTable": None}) is None


def test_validate_rejects_missing_smiles():
    df = pd.DataFrame(
        [
            {
                "name_pt": "Teste",
                "name_en": "Test",
                "category": "teste",
                "pubchem_name": "Ethanol",
                "smiles": "",
                "note": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="SMILES"):
        validate_ingredients(df)


def test_validate_rejects_duplicate_names():
    row = {
        "name_pt": "Café",
        "name_en": "Coffee",
        "category": "bebida",
        "pubchem_name": "Guaiacol",
        "smiles": "COc1ccccc1O",
        "note": "",
    }

    with pytest.raises(ValueError, match="duplicate"):
        validate_ingredients(pd.DataFrame([row, row]))
