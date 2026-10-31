"""Publish the trained model and curated dataset to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from olfacto.cards import (
    DATASET_ID,
    MODEL_ID,
    build_dataset_card,
    build_model_card,
    metrics_from_summary,
)

DATASET_FILES = {
    "curated_GS_LF_merged_4983.csv": "curated_GS_LF_merged_4983.csv",
    "odor_labels_ptbr.json": "odor_labels_ptbr.json",
    "brazilian_ingredients.csv": "brazilian_ingredients.csv",
    "map.csv": "map.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("outputs/cv/final")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/cv/cv_summary.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--space-id", default="SamuelCagnani/olfacto")
    parser.add_argument("--space-dir", type=Path, default=Path("space"))
    parser.add_argument("--skip-space", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--write-cards",
        type=Path,
        default=None,
        help="write the generated cards to this directory instead of uploading",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    metrics = metrics_from_summary(summary)
    model_card = build_model_card(metrics, dataset_id=args.dataset_id)
    dataset_card = build_dataset_card()

    if args.write_cards is not None:
        args.write_cards.mkdir(parents=True, exist_ok=True)
        (args.write_cards / "model-card.md").write_text(model_card, encoding="utf-8")
        (args.write_cards / "dataset-card.md").write_text(
            dataset_card, encoding="utf-8"
        )
        print(f"cards written to {args.write_cards}")
        return 0

    if args.dry_run:
        print(f"would upload model {args.model_id} and dataset {args.dataset_id}")
        print(f"macro ROC-AUC: {metrics['macro_roc_auc']:.4f}")
        return 0

    if not args.token:
        raise SystemExit("set HF_TOKEN or pass --token")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(args.model_dir),
        repo_id=args.model_id,
        repo_type="model",
        ignore_patterns=["*.pt", "optimizer*", "training_args.bin"],
    )
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.model_id,
        repo_type="model",
    )
    print(f"model published: https://huggingface.co/{args.model_id}")

    api.create_repo(args.dataset_id, repo_type="dataset", exist_ok=True)
    for source, target in DATASET_FILES.items():
        path = args.data_dir / source
        if not path.is_file():
            raise SystemExit(f"missing dataset file: {path}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target,
            repo_id=args.dataset_id,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=dataset_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.dataset_id,
        repo_type="dataset",
    )
    print(f"dataset published: https://huggingface.co/datasets/{args.dataset_id}")

    if not args.skip_space:
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(
            folder_path=str(args.space_dir),
            repo_id=args.space_id,
            repo_type="space",
        )
        print(f"space published: https://huggingface.co/spaces/{args.space_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
