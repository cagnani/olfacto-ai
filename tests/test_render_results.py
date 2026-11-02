import pandas as pd

from olfacto.report import render_results

SUMMARY = {
    "n_folds": 5,
    "config": {
        "model_name": "DeepChem/ChemBERTa-77M-MLM",
        "max_length": 128,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "epochs": 10,
        "seed": 42,
    },
    "fold_metrics": [
        {"fold": 0, "best_epoch": 8, "best_val_macro_auc": 0.75},
        {"fold": 1, "best_epoch": 9, "best_val_macro_auc": 0.77},
    ],
    "mean_fold_macro_auc": 0.76,
    "std_fold_macro_auc": 0.01,
    "oof": {
        "macro_roc_auc": 0.7654,
        "macro_roc_auc_ci": [0.75, 0.78],
        "micro_roc_auc": 0.81,
        "macro_pr_auc": 0.4,
        "recall_at_k": 0.55,
        "n_labels": 130,
        "min_positives": 20,
    },
}


def test_render_results_contains_real_numbers():
    per_label = pd.DataFrame(
        [
            {"label": "fruity", "positives": 1902, "roc_auc": 0.9, "pr_auc": 0.8},
            {"label": "chamomile", "positives": 31, "roc_auc": 0.6, "pr_auc": 0.2},
        ]
    )

    text = render_results(
        SUMMARY,
        per_label,
        baselines=[
            {
                "name": "Morgan + LogReg",
                "description": "5-fold OOF",
                "macro_roc_auc": 0.8293,
            }
        ],
    )

    assert "0.7654" in text
    assert "0.8293" in text
    assert "0.8872" in text
    assert "fruity" in text
    assert "chamomile" in text
    assert "TBD" not in text
