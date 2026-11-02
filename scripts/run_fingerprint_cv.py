"""Cross-validate a Morgan-fingerprint + logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olfacto.data import SMILES_COLUMN, get_label_columns, load_odor_dataset, make_splits
from olfacto.evaluate import compute_metrics
from olfacto.fingerprints import fit_final_model, fit_predict_fold, morgan_matrix

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/curated_GS_LF_merged_4983.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fingerprint"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_odor_dataset(args.data)
    labels = get_label_columns(df)
    y = df[labels].to_numpy(dtype=np.int8)
    X = morgan_matrix(df[SMILES_COLUMN].tolist(), args.radius, args.bits)
    split = make_splits(df, args.folds, args.seed)["fold"].to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = np.zeros((len(df), len(labels)), dtype=np.float32)
    fold_metrics = []

    for fold in range(args.folds):
        train_mask = split != fold
        test_mask = split == fold
        scores = fit_predict_fold(
            X[train_mask], y[train_mask], X[test_mask], args.c
        )
        oof[test_mask] = scores
        fold_auc = compute_metrics(
            y[test_mask],
            scores,
            labels,
            min_positives=20,
            n_bootstrap=0,
        )["macro_roc_auc"]
        fold_metrics.append(
            {
                "fold": fold,
                "best_val_macro_auc": fold_auc,
                "n_train": int(train_mask.sum()),
                "n_val": int(test_mask.sum()),
            }
        )
        print(f"[fold {fold}] macro_auc={fold_auc:.4f}", flush=True)

    metrics = compute_metrics(
        y, oof, labels, min_positives=20, n_bootstrap=args.bootstrap, seed=args.seed
    )
    fold_aucs = np.array([row["best_val_macro_auc"] for row in fold_metrics])

    summary = {
        "model": "morgan_logreg",
        "config": {
            "radius": args.radius,
            "n_bits": args.bits,
            "C": args.c,
            "seed": args.seed,
        },
        "n_folds": args.folds,
        "fold_metrics": fold_metrics,
        "mean_fold_macro_auc": float(fold_aucs.mean()),
        "std_fold_macro_auc": float(fold_aucs.std()),
        "oof": {key: value for key, value in metrics.items() if key != "per_label"},
    }
    (args.output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(metrics["per_label"]).to_csv(
        args.output_dir / "per_label.csv", index=False
    )
    np.save(args.output_dir / "oof_scores.npy", oof)

    print("[final] fitting full-data fingerprint model", flush=True)
    models = fit_final_model(X, y, args.c)
    joblib.dump(
        {
            "models": models,
            "labels": labels,
            "radius": args.radius,
            "n_bits": args.bits,
        },
        args.output_dir / "model.joblib",
    )

    print(
        f"[oof] macro_roc_auc={metrics['macro_roc_auc']:.4f} "
        f"micro={metrics['micro_roc_auc']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
