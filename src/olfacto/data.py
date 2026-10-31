"""Loading and validation for molecular odor datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

SMILES_COLUMN = "nonStereoSMILES"
DESCRIPTOR_COLUMN = "descriptors"
METADATA_COLUMNS = (SMILES_COLUMN, DESCRIPTOR_COLUMN)
LABEL_START_COLUMN = len(METADATA_COLUMNS)


def get_label_columns(df: pd.DataFrame) -> list[str]:
    """Return the binary odor descriptor columns, in dataset order."""
    return list(df.columns[LABEL_START_COLUMN:])


def load_odor_dataset(path: str | Path) -> pd.DataFrame:
    """Load a curated odor CSV and validate its schema.

    The expected format is the OpenPOM curated dataset: a SMILES column, a free-text
    descriptor column and 138 binary odor labels.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")

    df = pd.read_csv(path)
    _validate(df)
    return df


def make_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``fold`` column.

    Folds are built with iterative stratification so that rare odor labels appear in every
    fold. The split is deterministic for a given seed.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(df):
        raise ValueError("n_splits cannot exceed the number of molecules")

    labels = get_label_columns(df)
    y = df[labels].to_numpy(dtype=np.int8)
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    folds = np.full(len(df), -1, dtype=int)
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros((len(df), 1)), y)):
        folds[test_index] = fold

    if (folds < 0).any():
        raise RuntimeError("splitter did not assign every molecule to a fold")

    result = df.copy()
    result["fold"] = folds
    return result


def _validate(df: pd.DataFrame) -> None:
    if df.shape[1] <= LABEL_START_COLUMN:
        raise ValueError(
            f"expected at least {LABEL_START_COLUMN + 1} columns, got {df.shape[1]}"
        )

    if tuple(df.columns[:LABEL_START_COLUMN]) != METADATA_COLUMNS:
        raise ValueError(
            f"expected the first columns to be {METADATA_COLUMNS}, "
            f"got {tuple(df.columns[:LABEL_START_COLUMN])}"
        )

    if df.empty:
        raise ValueError("dataset is empty")

    smiles = df[SMILES_COLUMN].astype("string").str.strip()
    if smiles.isna().any() or (smiles == "").any():
        raise ValueError("SMILES column contains empty values")
    if smiles.duplicated().any():
        raise ValueError("dataset contains duplicate SMILES")
    df[SMILES_COLUMN] = smiles

    label_columns = get_label_columns(df)
    values = df[label_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("odor labels must be numeric")
    if not values.isin([0, 1]).all().all():
        raise ValueError("odor labels must be binary (0 or 1)")
    if (values.sum(axis=1) == 0).any():
        raise ValueError("each molecule must have at least one positive label")

    df[label_columns] = values.astype("int8")
