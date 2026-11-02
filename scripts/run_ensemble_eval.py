"""Blend out-of-fold transformer and fingerprint scores and evaluate the ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from olfacto.data import get_label_columns, load_odor_dataset
from olfacto.evaluate import compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/curated_GS_LF_merged_4983.csv")
    )
    parser.add_argument("--cv-dir", type=Path, default=Path("outputs/cv"))
    parser.add_argument(
        "--fingerprint-dir", type=Path, default=Path("outputs/fingerprint")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ensemble"))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    df = load_odor_dataset(args.data)
    labels = get_label_columns(df)
    y = df[labels].to_numpy(dtype=np.int8)

    transformer = np.load(args.cv_dir / "oof_scores.npy")
    fingerprint = np.load(args.fingerprint_dir / "oof_scores.npy")
    if transformer.shape != fingerprint.shape:
        raise SystemExit("out-of-fold score matrices have different shapes")

    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    best_weight = 0.0
    best_auc = float("-inf")
    for weight in weights:
        blended = (1.0 - weight) * transformer + weight * fingerprint
        auc = compute_metrics(
            y, blended, labels, min_positives=20, n_bootstrap=0
        )["macro_roc_auc"]
        print(f"weight={weight:.2f} macro_auc={auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_weight = float(weight)

    blended = (1.0 - best_weight) * transformer + best_weight * fingerprint
    metrics = compute_metrics(
        y,
        blended,
        labels,
        min_positives=20,
        n_bootstrap=args.bootstrap,
        seed=42,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "weight_fingerprint": best_weight,
        "oof": {key: value for key, value in metrics.items() if key != "per_label"},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(metrics["per_label"]).to_csv(
        args.output_dir / "per_label.csv", index=False
    )

    print(
        f"best weight={best_weight:.2f} macro_auc={metrics['macro_roc_auc']:.4f} "
        f"micro_auc={metrics['micro_roc_auc']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
