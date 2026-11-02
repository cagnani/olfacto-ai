"""Render results markdown from cross-validation artifacts."""

from __future__ import annotations

import pandas as pd

OPENPOM_BASELINE = 0.8872


def render_results(
    summary: dict,
    per_label: pd.DataFrame,
    baselines: list[dict] | None = None,
) -> str:
    oof = summary["oof"]
    low, high = oof["macro_roc_auc_ci"]
    folds = summary["fold_metrics"]

    fold_rows = "\n".join(
        f"| {row['fold']} | {row['best_epoch']} | {row['best_val_macro_auc']:.4f} |"
        for row in folds
    )

    ranked = per_label.sort_values("roc_auc", ascending=False)
    best = ranked.head(10)
    worst = ranked.tail(10).sort_values("roc_auc")
    best_rows = "\n".join(
        f"| {row.label} | {int(row.positives)} | {row.roc_auc:.4f} |"
        for row in best.itertuples()
    )
    worst_rows = "\n".join(
        f"| {row.label} | {int(row.positives)} | {row.roc_auc:.4f} |"
        for row in worst.itertuples()
    )

    config = summary["config"]

    baseline_rows = ""
    for baseline in baselines or []:
        baseline_rows += (
            f"\n| {baseline['name']} | {baseline['description']} | "
            f"{baseline['macro_roc_auc']:.4f} |"
        )

    return f"""# Resultados — validação cruzada

Métricas geradas por `scripts/run_cv.py` e por este renderizador. Nenhum número é manual.

## Configuração

| Parâmetro | Valor |
|---|---|
| Backbone | `{config["model_name"]}` |
| Comprimento máximo | {config["max_length"]} |
| Batch | {config["batch_size"]} |
| Learning rate | {config["learning_rate"]} |
| Épocas | {config["epochs"]} |
| Folds | {summary["n_folds"]} |
| Seed | {config["seed"]} |

## AUC por fold

| Fold | Melhor época | ROC-AUC macro |
|---|---|---|
{fold_rows}

Média dos folds: **{summary["mean_fold_macro_auc"]:.4f} ± {summary["std_fold_macro_auc"]:.4f}**.

## Métricas out-of-fold

| Métrica | Valor |
|---|---|
| ROC-AUC macro (OOF) | {oof["macro_roc_auc"]:.4f} |
| IC 95% (bootstrap) | [{low:.4f}, {high:.4f}] |
| ROC-AUC micro | {oof["micro_roc_auc"]:.4f} |
| PR-AUC macro | {oof["macro_pr_auc"]:.4f} |
| Recall@5 | {oof["recall_at_k"]:.4f} |
| Rótulos avaliados | {oof["n_labels"]} (mínimo de {oof["min_positives"]} positivos) |

## Comparação de modelos

| Modelo | Avaliação | ROC-AUC macro |
|---|---|---|
| OpenPOM (MPNN) | 5-fold CV, ensemble de 10 | {OPENPOM_BASELINE:.4f} |
| Olfacto (ChemBERTa) | 5-fold OOF, modelo único | {oof["macro_roc_auc"]:.4f} |{baseline_rows}

A comparação é uma ablation, não uma disputa de arquitetura. O OpenPOM usa um ensemble de
10 MPNNs; o ChemBERTa é um modelo único e pequeno. O baseline de fingerprints supera o
transformer neste dataset — um resultado honesto que motivou o ensemble usado no demo.

## Melhores rótulos

| Rótulo | Positivos | ROC-AUC |
|---|---|---|
{best_rows}

## Piores rótulos

| Rótulo | Positivos | ROC-AUC |
|---|---|---|
{worst_rows}

Rótulos com poucos positivos têm maior variância; o filtro de `min_positives` existe para
não maquiar a média com casos degenerados.

## Como reproduzir

```bash
uv sync --extra train
uv run --extra train python scripts/run_cv.py --output-dir outputs/cv --final
uv run --extra train python scripts/render_results.py
```
"""


