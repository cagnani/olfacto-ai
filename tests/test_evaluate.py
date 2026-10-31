import numpy as np
import pytest

from olfacto.evaluate import compute_metrics, mean_roc_auc, recall_at_k


def _toy(n_rows=200, n_labels=4, seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random((n_rows, n_labels)) > 0.6).astype(int)
    y_true[0, 0] = 1
    y_true[1, 0] = 0
    y_score = rng.random((n_rows, n_labels))
    return y_true, y_score


def test_perfect_predictions_score_one():
    y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]] * 25)
    y_score = y_true.astype(float)

    assert mean_roc_auc(y_true, y_score, min_positives=2) == pytest.approx(1.0)


def test_random_predictions_are_near_chance():
    y_true, y_score = _toy(seed=7)

    value = mean_roc_auc(y_true, y_score, min_positives=2)

    assert 0.3 < value < 0.7


def test_degenerate_label_is_excluded():
    y_true, y_score = _toy()
    y_true[:, 3] = 0

    metrics = compute_metrics(
        y_true, y_score, ["a", "b", "c", "d"], min_positives=2, n_bootstrap=5
    )

    assert metrics["n_labels"] == 3
    assert all(row["label"] != "d" for row in metrics["per_label"])


def test_min_positives_filters_labels():
    y_true = np.zeros((100, 3), dtype=int)
    y_true[:10, 0] = 1
    y_true[:30, 1] = 1
    y_true[:60, 2] = 1
    y_score = np.random.default_rng(0).random((100, 3))

    metrics = compute_metrics(
        y_true, y_score, ["rare", "mid", "common"], min_positives=20, n_bootstrap=5
    )

    assert [row["label"] for row in metrics["per_label"]] == ["mid", "common"]


def test_recall_at_k():
    y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 0]])
    y_score = np.array([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.3]])

    assert recall_at_k(y_true, y_score, k=2) == pytest.approx(1.0)


def test_bootstrap_interval_contains_macro():
    y_true, y_score = _toy(seed=3)

    metrics = compute_metrics(
        y_true, y_score, ["a", "b", "c", "d"], min_positives=2, n_bootstrap=50
    )

    low, high = metrics["macro_roc_auc_ci"]
    assert low <= metrics["macro_roc_auc"] <= high
