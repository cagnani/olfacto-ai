"""Fine-tune a SMILES transformer for multi-label odor prediction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import SMILES_COLUMN, get_label_columns, load_odor_dataset, make_splits
from .evaluate import mean_roc_auc


@dataclass
class TrainConfig:
    model_name: str = "DeepChem/ChemBERTa-77M-MLM"
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    pos_weight_cap: float = 20.0
    early_stopping_patience: int = 3
    val_min_positives: int = 5
    seed: int = 42
    fold: int | None = None
    output_dir: str = "outputs/chemberta"
    num_workers: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(y: np.ndarray, cap: float = 20.0) -> np.ndarray:
    """Per-label negative/positive ratio, clipped to ``cap`` and floored at 1."""
    positives = y.sum(axis=0).astype(float)
    negatives = y.shape[0] - positives
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(positives > 0, negatives / positives, 1.0)
    return np.clip(weights, 1.0, cap)


def build_model(config: TrainConfig, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )


def tokenize(
    tokenizer, smiles: list[str], max_length: int
) -> dict[str, torch.Tensor]:
    return tokenizer(
        smiles,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def predict_proba(model, tokenizer, smiles: list[str], max_length: int = 128) -> np.ndarray:
    """Return sigmoid probabilities for a list of SMILES."""
    device = next(model.parameters()).device
    model.eval()
    encoded = tokenize(tokenizer, smiles, max_length)
    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).logits
    return torch.sigmoid(logits).cpu().numpy()


def _loader(
    encoded: dict[str, torch.Tensor],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(np.array(labels, dtype=np.float32, copy=True)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def fit(config: TrainConfig, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train one model and return metrics. Saves the best checkpoint."""
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_columns = get_label_columns(train_df)
    (output_dir / "labels.json").write_text(
        json.dumps(label_columns, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    y_train = train_df[label_columns].to_numpy(dtype=np.float32)
    y_val = val_df[label_columns].to_numpy(dtype=np.float32)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_encoded = tokenize(tokenizer, train_df[SMILES_COLUMN].tolist(), config.max_length)
    val_encoded = tokenize(tokenizer, val_df[SMILES_COLUMN].tolist(), config.max_length)

    train_loader = _loader(
        train_encoded, y_train, config.batch_size, True, config.num_workers
    )
    val_loader = _loader(
        val_encoded, y_val, config.batch_size, False, config.num_workers
    )

    model = build_model(config, len(label_columns))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pos_weight = torch.as_tensor(
        compute_pos_weight(y_train, config.pos_weight_cap),
        dtype=torch.float,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: list[dict] = []
    best_auc = float("-inf")
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * input_ids.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_scores = []
        val_labels = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                ).logits
                val_scores.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(labels.numpy())

        y_score = np.concatenate(val_scores)
        y_true = np.concatenate(val_labels)
        val_auc = mean_roc_auc(y_true, y_score, config.val_min_positives)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auc": val_auc})

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    metrics = {
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_macro_auc": best_auc if best_epoch > 0 else float("nan"),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "num_labels": len(label_columns),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/curated_GS_LF_merged_4983.csv"),
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    df = load_odor_dataset(args.data)

    if config.fold is None:
        split = make_splits(df, n_splits=10, seed=config.seed)
        train_df = split[split["fold"] != 0].reset_index(drop=True)
        val_df = split[split["fold"] == 0].reset_index(drop=True)
    else:
        split = make_splits(df, n_splits=5, seed=config.seed)
        train_df = split[split["fold"] != config.fold].reset_index(drop=True)
        val_df = split[split["fold"] == config.fold].reset_index(drop=True)

    metrics = fit(config, train_df, val_df)
    print(
        f"fold={config.fold} best_epoch={metrics['best_epoch']} "
        f"val_macro_auc={metrics['best_val_macro_auc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
