import json
from pathlib import Path

from olfacto.ingredients import load_ingredients

ROOT = Path(__file__).resolve().parents[1]
DISHES_PATH = ROOT / "data" / "dish_ingredients.json"
CORPUS_PATH = ROOT / "data" / "sensory_corpus.jsonl"
CATALOG_PATHS = [
    ROOT / "data" / "brazilian_ingredients.csv",
    ROOT / "data" / "sommelier_ingredients.csv",
]


def load_dishes() -> list[dict]:
    return json.loads(DISHES_PATH.read_text(encoding="utf-8"))


def test_dish_catalog_is_complete():
    dishes = load_dishes()

    assert len(dishes) >= 40
    assert len({dish["name"] for dish in dishes}) == len(dishes)
    assert len({dish["category"] for dish in dishes}) == 8
    for dish in dishes:
        assert dish["ingredients"]
        assert all(isinstance(item, str) and item.strip() for item in dish["ingredients"])


def test_every_ingredient_exists_in_a_catalog():
    known = set()
    for path in CATALOG_PATHS:
        known.update(load_ingredients(path)["name_pt"])

    used = {item for dish in load_dishes() for item in dish["ingredients"]}

    assert sorted(used - known) == []


def load_corpus() -> list[dict]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_corpus_covers_every_dish():
    dishes = {dish["name"]: dish for dish in load_dishes()}
    corpus = load_corpus()
    by_dish = {entry["dish"]: entry for entry in corpus}

    assert len(corpus) >= 40
    assert len({entry["id"] for entry in corpus}) == len(corpus)
    assert set(by_dish) == set(dishes)

    for name, dish in dishes.items():
        entry = by_dish[name]
        assert entry["category"] == dish["category"]
        assert entry["ingredients"] == dish["ingredients"]
        assert entry["aromas"]
        assert entry["pairings"]
        assert len(entry["description"]) >= 20
