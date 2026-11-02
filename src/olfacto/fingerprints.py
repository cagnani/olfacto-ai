"""Morgan fingerprint baseline for molecular odor prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

DEFAULT_RADIUS = 2
DEFAULT_BITS = 2048


def morgan_matrix(
    smiles: list[str],
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_BITS,
) -> np.ndarray:
    """Return a binary Morgan fingerprint matrix for the given SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {value}")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule, radius, nBits=n_bits
        )
        rows.append(np.array(list(fingerprint), dtype=np.uint8))
    return np.vstack(rows)


def fit_predict_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    c: float = 1.0,
) -> np.ndarray:
    """Fit one balanced logistic regression per label and score the test rows."""
    scores = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=np.float32)
    for index in range(y_train.shape[1]):
        target = y_train[:, index]
        if target.sum() == 0 or target.sum() == len(target):
            continue
        classifier = LogisticRegression(
            max_iter=500, class_weight="balanced", C=c
        )
        classifier.fit(x_train, target)
        scores[:, index] = classifier.predict_proba(x_test)[:, 1]
    return scores


def fit_final_model(
    x: np.ndarray, y: np.ndarray, c: float = 1.0
) -> list[LogisticRegression | None]:
    """Fit one classifier per label over the full matrix."""
    models: list[LogisticRegression | None] = []
    for index in range(y.shape[1]):
        target = y[:, index]
        if target.sum() == 0 or target.sum() == len(target):
            models.append(None)
            continue
        classifier = LogisticRegression(
            max_iter=500, class_weight="balanced", C=c
        )
        classifier.fit(x, target)
        models.append(classifier)
    return models


def predict_scores(x: np.ndarray, models: list[LogisticRegression | None]) -> np.ndarray:
    """Score rows with a list of per-label classifiers."""
    scores = np.zeros((x.shape[0], len(models)), dtype=np.float32)
    for index, classifier in enumerate(models):
        if classifier is None:
            continue
        scores[:, index] = classifier.predict_proba(x)[:, 1]
    return scores
