"""Reusable SMILES-to-odor inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import load_lexicon

_ALLOWED_SMILES_CHARS = set(
    "BCNOPSFIbcnopsfi" "[]()=#+-@.0123456789/%\\Hhrl"
)


@dataclass(frozen=True)
class Prediction:
    label: str
    label_pt_br: str
    probability: float


def _looks_like_smiles(smiles: str) -> bool:
    return (
        bool(smiles)
        and set(smiles) <= _ALLOWED_SMILES_CHARS
        and smiles.count("(") == smiles.count(")")
        and smiles.count("[") == smiles.count("]")
    )


def is_valid_smiles(smiles: str) -> bool:
    """Validate a SMILES string, using RDKit when it is installed."""
    if not smiles or not smiles.strip():
        return False

    try:
        from rdkit import Chem
    except ImportError:
        return _looks_like_smiles(smiles.strip())

    return Chem.MolFromSmiles(smiles) is not None


def rank_predictions(
    scores: np.ndarray,
    labels: list[str],
    lexicon: dict[str, dict[str, str]],
    top_k: int = 10,
) -> list[Prediction]:
    """Turn a score vector into the top-k predictions, translated to PT-BR."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    order = np.argsort(-np.asarray(scores))[:top_k]
    return [
        Prediction(
            label=labels[index],
            label_pt_br=lexicon[labels[index]]["pt_br"],
            probability=float(scores[index]),
        )
        for index in order
    ]


class OdorPredictor:
    """Thin wrapper around a fine-tuned sequence classifier."""

    def __init__(
        self,
        model,
        tokenizer,
        labels: list[str],
        lexicon: dict[str, dict[str, str]],
        max_length: int = 128,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.lexicon = lexicon
        self.max_length = max_length
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        lexicon_path: str | Path,
        device: str = "cpu",
        max_length: int = 128,
    ) -> OdorPredictor:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        model_path = Path(model_path)
        lexicon = load_lexicon(lexicon_path)

        labels_file = model_path / "labels.json"
        if labels_file.is_file():
            labels = json.loads(labels_file.read_text(encoding="utf-8"))
        else:
            labels = list(lexicon)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(torch.device(device))
        model.eval()

        return cls(model, tokenizer, labels, lexicon, max_length, device)

    def predict_batch(
        self, smiles_list: list[str], top_k: int = 10
    ) -> list[list[Prediction]]:
        invalid = [smiles for smiles in smiles_list if not is_valid_smiles(smiles)]
        if invalid:
            raise ValueError(f"invalid SMILES: {invalid}")

        import torch

        encoded = self.tokenizer(
            smiles_list,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        scores = torch.sigmoid(logits).cpu().numpy()

        return [
            rank_predictions(row, self.labels, self.lexicon, top_k) for row in scores
        ]

    def predict(self, smiles: str, top_k: int = 10) -> list[Prediction]:
        return self.predict_batch([smiles], top_k)[0]
