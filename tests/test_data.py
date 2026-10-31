from pathlib import Path

import pandas as pd
import pytest

from olfacto.data import get_label_columns, load_odor_dataset, make_splits

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "curated_GS_LF_merged_4983.csv"


def _write_csv(
    tmp_path: Path,
    rows: list[tuple[str, str, list[int]]],
    labels=("fruity", "floral"),
) -> Path:
    path = tmp_path / "odor.csv"
    lines = ["nonStereoSMILES,descriptors," + ",".join(labels)]
    for smiles, descriptor, values in rows:
        lines.append(f"{smiles},{descriptor}," + ",".join(str(v) for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_dataset_shape():
    df = load_odor_dataset(DATA_PATH)
    assert df.shape == (4983, 140)


def test_label_columns():
    df = load_odor_dataset(DATA_PATH)
    labels = get_label_columns(df)
    assert len(labels) == 138
    assert labels[0] == "alcoholic"
    assert labels[-1] == "woody"
    assert "nonStereoSMILES" not in labels
    assert "descriptors" not in labels


def test_dataset_has_no_duplicate_smiles():
    df = load_odor_dataset(DATA_PATH)
    assert not df["nonStereoSMILES"].duplicated().any()


def test_rejects_non_binary_label(tmp_path):
    path = _write_csv(tmp_path, [("CCO", "fishy", [2, 0])])
    with pytest.raises(ValueError, match="binary"):
        load_odor_dataset(path)


def test_rejects_empty_smiles(tmp_path):
    path = _write_csv(tmp_path, [("  ", "fishy", [1, 0])])
    with pytest.raises(ValueError, match="SMILES"):
        load_odor_dataset(path)


def test_rejects_duplicate_smiles(tmp_path):
    path = _write_csv(tmp_path, [("CCO", "fishy", [1, 0]), ("CCO", "sweet", [1, 0])])
    with pytest.raises(ValueError, match="duplicate"):
        load_odor_dataset(path)


def test_rejects_row_without_positive_label(tmp_path):
    path = _write_csv(tmp_path, [("CCO", "fishy", [0, 0])])
    with pytest.raises(ValueError, match="at least one"):
        load_odor_dataset(path)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_odor_dataset(tmp_path / "nope.csv")


def test_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        load_odor_dataset(path)


def test_make_splits_assigns_every_row_once():
    df = load_odor_dataset(DATA_PATH)
    split = make_splits(df, n_splits=5, seed=42)

    assert len(split) == len(df)
    assert set(split["fold"]) == {0, 1, 2, 3, 4}
    assert not split["fold"].isna().any()
    assert split["fold"].value_counts().min() >= int(0.18 * len(df))


def test_rare_label_present_in_every_fold():
    df = load_odor_dataset(DATA_PATH)
    split = make_splits(df, n_splits=5, seed=42)

    per_fold = split.groupby("fold")["chamomile"].sum()
    assert (per_fold > 0).all(), per_fold.to_dict()


def test_splits_are_deterministic():
    df = load_odor_dataset(DATA_PATH)
    first = make_splits(df, n_splits=5, seed=42)["fold"].tolist()
    second = make_splits(df, n_splits=5, seed=42)["fold"].tolist()
    assert first == second


def test_invalid_n_splits_raises():
    df = load_odor_dataset(DATA_PATH)
    with pytest.raises(ValueError, match="n_splits"):
        make_splits(df, n_splits=1)
