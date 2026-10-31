import pandas as pd

from olfacto.predict import Prediction
from olfacto.space_logic import (
    find_ingredient,
    odor_wheel_svg,
    predictions_table,
    scatter_svg,
)

LEXICON = {
    "rose": {"pt_br": "rosa", "family": "floral"},
    "sweet": {"pt_br": "doce", "family": "doce"},
}
PREDICTIONS = [
    Prediction("sweet", "doce", 0.91),
    Prediction("rose", "rosa", 0.42),
]


def test_predictions_table_rounds_probabilities():
    rows = predictions_table(PREDICTIONS)

    assert rows[0]["aroma"] == "doce"
    assert rows[0]["probabilidade"] == 0.91
    assert len(rows) == 2


def test_odor_wheel_contains_labels_and_svg():
    svg = odor_wheel_svg(PREDICTIONS, LEXICON)

    assert svg.startswith("<svg")
    assert "doce" in svg
    assert "rosa" in svg


def test_odor_wheel_handles_empty_predictions():
    assert "Sem previsões" in odor_wheel_svg([], LEXICON)


def test_find_ingredient():
    catalog = pd.DataFrame(
        [
            {"name_pt": "Café", "pubchem_name": "Furaneol"},
            {"name_pt": "Canela", "pubchem_name": "Cinnamaldehyde"},
        ]
    )

    assert find_ingredient(catalog, "Canela")["pubchem_name"] == "Cinnamaldehyde"
    assert find_ingredient(catalog, "Nada") is None


def test_scatter_svg_draws_one_circle_per_point():
    points = pd.DataFrame(
        [
            {"x": 0.0, "y": 0.0, "family": "floral", "name": "rosa"},
            {"x": 1.0, "y": 1.0, "family": "frutado", "name": "banana"},
        ]
    )

    svg = scatter_svg(points)

    assert svg.count("<circle") == 2
    assert "rosa" in svg
