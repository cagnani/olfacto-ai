"""Ensemble of the SMILES transformer and the fingerprint baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .fingerprints import morgan_matrix, predict_scores
from .labels import load_lexicon
from .predict import OdorPredictor, Prediction, is_valid_smiles, rank_predictions


def combine_scores(
    transformer_scores: np.ndarray,
    fingerprint_scores: np.ndarray,
    weight_fingerprint: float = 0.65,
) -> np.ndarray:
    """Weighted average of two score matrices."""
    if not 0.0 <= weight_fingerprint <= 1.0:
        raise ValueError("weight_fingerprint must be between 0 and 1")
    if transformer_scores.shape != fingerprint_scores.shape:
        raise ValueError("score matrices must have the same shape")

    return (1.0 - weight_fingerprint) * transformer_scores + (
        weight_fingerprint * fingerprint_scores
    )


class EnsemblePredictor:
    """Blend transformer probabilities with a Morgan-fingerprint model."""

    def __init__(
        self,
        transformer: OdorPredictor,
        fingerprint_bundle: dict,
        lexicon: dict[str, dict[str, str]],
        weight_fingerprint: float = 0.65,
    ) -> None:
        self.transformer = transformer
        self.models = fingerprint_bundle["models"]
        self.labels = fingerprint_bundle["labels"]
        self.radius = fingerprint_bundle["radius"]
        self.n_bits = fingerprint_bundle["n_bits"]
        self.lexicon = lexicon
        self.weight_fingerprint = weight_fingerprint

        if self.labels != transformer.labels:
            raise ValueError("transformer and fingerprint label orders differ")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        lexicon_path: str | Path,
        fingerprint_path: str | Path,
        device: str = "cpu",
        weight_fingerprint: float = 0.65,
        max_length: int = 128,
    ) -> EnsemblePredictor:
        transformer = OdorPredictor.from_pretrained(
            model_path, lexicon_path, device=device, max_length=max_length
        )
        bundle = joblib.load(fingerprint_path)
        return cls(
            transformer,
            bundle,
            load_lexicon(lexicon_path),
            weight_fingerprint=weight_fingerprint,
        )

    def predict_scores(self, smiles: str) -> np.ndarray:
        if not is_valid_smiles(smiles):
            raise ValueError(f"invalid SMILES: {smiles}")
        transformer_scores = self.transformer.predict_scores_batch([smiles])[0]
        fingerprint_scores = predict_scores(
            morgan_matrix([smiles], self.radius, self.n_bits), self.models
        )[0]
        return combine_scores(
            transformer_scores, fingerprint_scores, self.weight_fingerprint
        )

    def predict(self, smiles: str, top_k: int = 10) -> list[Prediction]:
        scores = self.predict_scores(smiles)
        return rank_predictions(scores, self.labels, self.lexicon, top_k)
