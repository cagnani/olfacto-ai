import pytest

from olfacto.cards import build_dataset_card, build_model_card, metrics_from_summary

METRICS = {
    "macro_roc_auc": 0.81,
    "micro_roc_auc": 0.86,
    "macro_pr_auc": 0.42,
    "macro_roc_auc_ci": [0.79, 0.83],
    "n_labels": 130,
    "mean_fold_macro_auc": 0.805,
    "std_fold_macro_auc": 0.012,
}


def test_model_card_contains_real_metrics():
    card = build_model_card(METRICS)

    assert "0.8100" in card
    assert "[0.7900, 0.8300]" in card
    assert "0.8050" in card
    assert "0.8872" in card
    assert "license: mit" in card
    assert "TBD" not in card


def test_model_card_requires_metrics():
    broken = dict(METRICS)
    del broken["macro_roc_auc"]

    with pytest.raises(ValueError, match="macro_roc_auc"):
        build_model_card(broken)


def test_dataset_card_mentions_sources():
    card = build_dataset_card()

    assert "4983" in card
    assert "OpenPOM" in card
    assert "GoodScents" in card
    assert "odor_labels_ptbr.json" in card


def test_metrics_from_summary_flattens_oof():
    summary = {
        "oof": {
            "macro_roc_auc": 0.81,
            "micro_roc_auc": 0.86,
            "macro_pr_auc": 0.42,
            "macro_roc_auc_ci": [0.79, 0.83],
            "n_labels": 130,
        },
        "mean_fold_macro_auc": 0.805,
        "std_fold_macro_auc": 0.012,
    }

    metrics = metrics_from_summary(summary)

    assert metrics["macro_roc_auc"] == 0.81
    assert metrics["mean_fold_macro_auc"] == 0.805


def test_metrics_from_summary_rejects_incomplete():
    with pytest.raises(ValueError, match="oof"):
        metrics_from_summary({})
