"""Multi-label evaluation for odor prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _label_metrics(
    y_true: np.ndarray, y_score: np.ndarray, min_positives: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    n_rows = y_true.shape[0]
    aucs: list[float] = []
    pr_aucs: list[float] = []
    included: list[int] = []

    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        positives = int(truth.sum())
        if positives < min_positives or positives == n_rows:
            continue
        aucs.append(roc_auc_score(truth, y_score[:, index]))
        pr_aucs.append(average_precision_score(truth, y_score[:, index]))
        included.append(index)

    return np.asarray(aucs), np.asarray(pr_aucs), included


def mean_roc_auc(
    y_true: np.ndarray, y_score: np.ndarray, min_positives: int = 20
) -> float:
    """Macro ROC-AUC over labels with enough positives."""
    aucs, _, _ = _label_metrics(y_true, y_score, min_positives)
    return float(aucs.mean()) if aucs.size else float("nan")


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Average fraction of true labels that appear in the top-k predictions."""
    hits = 0.0
    for row in range(y_true.shape[0]):
        truth = set(np.flatnonzero(y_true[row]).tolist())
        if not truth:
            continue
        top = set(np.argsort(-y_score[row])[:k].tolist())
        hits += len(truth & top) / len(truth)
    return hits / y_true.shape[0]


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    min_positives: int = 20,
    n_bootstrap: int = 1000,
    seed: int = 42,
    k: int = 5,
) -> dict:
    """Compute aggregate metrics, bootstrap CI and a per-label table."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    aucs, pr_aucs, included = _label_metrics(y_true, y_score, min_positives)
    macro_auc = float(aucs.mean()) if aucs.size else float("nan")
    micro_auc = float(roc_auc_score(y_true.ravel(), y_score.ravel()))

    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, y_true.shape[0], y_true.shape[0])
        value = mean_roc_auc(y_true[sample], y_score[sample], min_positives)
        if not np.isnan(value):
            bootstrap.append(value)
    if bootstrap:
        low, high = np.percentile(bootstrap, [2.5, 97.5])
    else:
        low = high = float("nan")

    per_label = [
        {
            "label": label_names[index],
            "positives": int(y_true[:, index].sum()),
            "roc_auc": float(aucs[position]),
            "pr_auc": float(pr_aucs[position]),
        }
        for position, index in enumerate(included)
    ]

    return {
        "macro_roc_auc": macro_auc,
        "micro_roc_auc": micro_auc,
        "macro_pr_auc": float(pr_aucs.mean()) if pr_aucs.size else float("nan"),
        "recall_at_k": recall_at_k(y_true, y_score, k),
        "macro_roc_auc_ci": [float(low), float(high)],
        "n_labels": len(included),
        "min_positives": min_positives,
        "k": k,
        "per_label": per_label,
    }
