from pathlib import Path

import pytest

from olfacto.agent.tools import OdorTools, aggregate_predictions, load_smiles_map
from olfacto.predict import Prediction

ROOT = Path(__file__).resolve().parents[1]
CATALOGS = [
    ROOT / "data" / "brazilian_ingredients.csv",
    ROOT / "data" / "sommelier_ingredients.csv",
]
DISHES = ROOT / "data" / "dish_ingredients.json"


class FakePredictor:
    def __init__(self, values: dict[str, list[Prediction]] | None = None) -> None:
        self.values = values or {}

    def predict(self, smiles: str, top_k: int = 10):
        return self.values.get(
            smiles,
            [
                Prediction("sweet", "doce", 0.8),
                Prediction("smoky", "defumado", 0.4),
            ],
        )[:top_k]


def make_tools() -> OdorTools:
    return OdorTools(FakePredictor(), CATALOGS, DISHES)


def test_load_smiles_map_covers_both_catalogs():
    mapping = load_smiles_map(CATALOGS)

    assert mapping["Alho"]
    assert mapping["Queijo"]
    assert mapping["Arroz"]


def test_profile_aggregates_by_maximum():
    rows = [
        {
            "predictions": [
                Prediction("sweet", "doce", 0.4),
                Prediction("smoky", "defumado", 0.2),
            ]
        },
        {
            "predictions": [
                Prediction("sweet", "doce", 0.9),
                Prediction("fruity", "frutado", 0.3),
            ]
        },
    ]

    result = aggregate_predictions(rows, top_k=2)

    assert result[0]["label"] == "sweet"
    assert result[0]["probability"] == pytest.approx(0.9)
    assert result[1]["label"] == "fruity"


def test_profile_resolves_known_dish():
    profile = make_tools().profile("Feijoada", top_k=3)

    assert profile["category"] == "Brasileira"
    assert len(profile["ingredients"]) == 6
    assert profile["missing"] == []
    assert profile["aromas"][0]["label"] == "sweet"


def test_profile_uses_extra_ingredients_and_reports_missing():
    profile = make_tools().profile(
        "Prato inventado", extra_ingredients=["Alho", "Ingrediente inexistente"]
    )

    assert len(profile["ingredients"]) == 1
    assert profile["missing"] == ["Ingrediente inexistente"]
    assert profile["category"] == "Outro"


def test_profile_rejects_unknown_dish_without_ingredients():
    with pytest.raises(ValueError, match="unknown dish"):
        make_tools().profile("Prato inventado")


def test_aggregate_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        aggregate_predictions([], top_k=0)
