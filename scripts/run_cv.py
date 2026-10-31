"""Run multi-label cross-validation and save out-of-fold metrics."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from olfacto.data import (
    SMILES_COLUMN,
    get_label_columns,
    load_odor_dataset,
    make_splits,
)
from olfacto.evaluate import compute_metrics
from olfacto.train import TrainConfig, fit, predict_proba


def _load_artifacts(fold_dir: Path, max_length: int, smiles: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(fold_dir)
    model = AutoModelForSequenceClassification.from_pretrained(fold_dir)
    return predict_proba(model, tokenizer, smiles, max_length)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_chemberta.yaml"))
    parser.add_argument(
        "--data", type=Path, default=Path("data/curated_GS_LF_merged_4983.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--final", action="store_true", help="also train a final model")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    else:
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    config = TrainConfig.from_yaml(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.max_length is not None:
        config.max_length = args.max_length
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    df = load_odor_dataset(args.data)
    label_columns = get_label_columns(df)
    split = make_splits(df, n_splits=args.folds, seed=config.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_scores = np.zeros((len(df), len(label_columns)), dtype=np.float32)
    oof_true = df[label_columns].to_numpy(dtype=np.int8)
    covered = np.zeros(len(df), dtype=bool)
    fold_metrics: list[dict] = []

    for fold in range(args.folds):
        train_df = split[split["fold"] != fold]
        val_df = split[split["fold"] == fold]
        val_index = val_df.index.to_numpy()
        fold_dir = args.output_dir / f"fold{fold}"
        fold_config = replace(config, fold=fold, output_dir=str(fold_dir))

        metrics_file = fold_dir / "metrics.json"
        if metrics_file.is_file():
            print(f"[fold {fold}] reusing {metrics_file}", flush=True)
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        else:
            print(f"[fold {fold}] training {len(train_df)} / {len(val_df)}", flush=True)
            metrics = fit(
                fold_config,
                train_df.reset_index(drop=True),
                val_df.reset_index(drop=True),
            )

        scores = _load_artifacts(
            fold_dir, fold_config.max_length, val_df[SMILES_COLUMN].tolist()
        )
        oof_scores[val_index] = scores
        covered[val_index] = True
        fold_metrics.append(
            {
                "fold": fold,
                "best_epoch": metrics["best_epoch"],
                "best_val_macro_auc": metrics["best_val_macro_auc"],
                "n_train": metrics["n_train"],
                "n_val": metrics["n_val"],
            }
        )
        print(
            f"[fold {fold}] best_epoch={metrics['best_epoch']} "
            f"val_macro_auc={metrics['best_val_macro_auc']:.4f}",
            flush=True,
        )

    if not covered.all():
        raise RuntimeError("some molecules were not covered by any fold")

    print("[oof] computing metrics", flush=True)
    oof_metrics = compute_metrics(
        oof_true,
        oof_scores,
        label_columns,
        min_positives=20,
        n_bootstrap=args.bootstrap,
        seed=config.seed,
    )

    fold_aucs = np.array([row["best_val_macro_auc"] for row in fold_metrics], dtype=float)
    summary = {
        "config": asdict(config),
        "n_folds": args.folds,
        "fold_metrics": fold_metrics,
        "mean_fold_macro_auc": float(np.nanmean(fold_aucs)),
        "std_fold_macro_auc": float(np.nanstd(fold_aucs)),
        "oof": {key: value for key, value in oof_metrics.items() if key != "per_label"},
    }
    (args.output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(oof_metrics["per_label"]).to_csv(
        args.output_dir / "per_label.csv", index=False
    )
    np.save(args.output_dir / "oof_scores.npy", oof_scores)
    np.save(args.output_dir / "oof_true.npy", oof_true)

    print(
        f"[oof] macro_roc_auc={oof_metrics['macro_roc_auc']:.4f} "
        f"micro={oof_metrics['micro_roc_auc']:.4f} "
        f"labels={oof_metrics['n_labels']}",
        flush=True,
    )

    if args.final:
        print("[final] training on 90% with 10% validation", flush=True)
        final_split = make_splits(df, n_splits=10, seed=config.seed)
        final_train = final_split[final_split["fold"] != 0]
        final_val = final_split[final_split["fold"] == 0]
        final_config = replace(
            config, fold=None, output_dir=str(args.output_dir / "final")
        )
        fit(
            final_config,
            final_train.reset_index(drop=True),
            final_val.reset_index(drop=True),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
