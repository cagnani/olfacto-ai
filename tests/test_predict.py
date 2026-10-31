from types import SimpleNamespace

import numpy as np
import pytest

from olfacto.predict import (
    OdorPredictor,
    Prediction,
    is_valid_smiles,
    rank_predictions,
)

LEXICON = {
    "rose": {"pt_br": "rosa", "family": "floral"},
    "sweet": {"pt_br": "doce", "family": "doce"},
    "smoky": {"pt_br": "defumado", "family": "alimentar"},
}


def test_is_valid_smiles_accepts_ethanol():
    assert is_valid_smiles("CCO") is True


def test_is_valid_smiles_rejects_garbage():
    assert is_valid_smiles("") is False
    assert is_valid_smiles("não é química") is False


def test_rank_predictions_orders_by_score():
    scores = np.array([0.2, 0.9, 0.5])

    result = rank_predictions(scores, ["rose", "sweet", "smoky"], LEXICON, top_k=2)

    assert result == [
        Prediction("sweet", "doce", pytest.approx(0.9)),
        Prediction("smoky", "defumado", pytest.approx(0.5)),
    ]


def test_rank_predictions_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        rank_predictions(np.array([0.1]), ["rose", "sweet"], LEXICON)


def test_rank_predictions_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        rank_predictions(np.array([0.1]), ["rose"], LEXICON, top_k=0)


def test_predictor_ranks_with_fake_model():
    torch = pytest.importorskip("torch")

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            logits = torch.tensor([[0.0, 5.0, -5.0]]).repeat(input_ids.shape[0], 1)
            return SimpleNamespace(logits=logits)

    class FakeTokenizer:
        def __call__(self, smiles_list, **kwargs):
            count = len(smiles_list)
            return {
                "input_ids": torch.ones((count, 4), dtype=torch.long),
                "attention_mask": torch.ones((count, 4), dtype=torch.long),
            }

    predictor = OdorPredictor(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        labels=["rose", "sweet", "smoky"],
        lexicon=LEXICON,
    )

    result = predictor.predict("CCO", top_k=1)

    assert result[0].label == "sweet"
    assert result[0].probability > 0.99


def test_predictor_rejects_invalid_smiles():
    predictor = OdorPredictor(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        labels=["rose"],
        lexicon=LEXICON,
    )

    with pytest.raises(ValueError, match="invalid SMILES"):
        predictor.predict("isto não é um smiles")
