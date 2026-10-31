from pathlib import Path

import pytest

from olfacto.data import get_label_columns, load_odor_dataset
from olfacto.labels import get_family, load_lexicon, translate_labels

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "curated_GS_LF_merged_4983.csv"
LEXICON_PATH = ROOT / "data" / "odor_labels_ptbr.json"


def test_lexicon_covers_dataset_labels():
    df = load_odor_dataset(DATA_PATH)
    lexicon = load_lexicon(LEXICON_PATH)

    assert set(lexicon) == set(get_label_columns(df))
    assert len(lexicon) == 138


def test_lexicon_entries_are_complete():
    lexicon = load_lexicon(LEXICON_PATH)

    for label, entry in lexicon.items():
        assert set(entry) == {"pt_br", "family"}, label
        assert entry["pt_br"].strip(), label
        assert entry["family"].strip(), label


def test_no_duplicate_translations():
    lexicon = load_lexicon(LEXICON_PATH)
    translations = [entry["pt_br"] for entry in lexicon.values()]

    assert len(translations) == len(set(translations))


def test_translate_labels_preserves_order():
    lexicon = load_lexicon(LEXICON_PATH)

    assert translate_labels(["rose", "sweet"], lexicon) == ["rosa", "doce"]


def test_translate_unknown_label_raises():
    with pytest.raises(KeyError):
        translate_labels(["unknown"], {})


def test_get_family():
    lexicon = load_lexicon(LEXICON_PATH)

    assert get_family("rose", lexicon) == "floral"
    assert get_family("smoky", lexicon) == "alimentar"
