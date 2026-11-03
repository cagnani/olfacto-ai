"""Generate Hugging Face model and dataset cards from real metrics."""

from __future__ import annotations

REQUIRED_METRICS = (
    "macro_roc_auc",
    "micro_roc_auc",
    "macro_pr_auc",
    "macro_roc_auc_ci",
    "n_labels",
    "mean_fold_macro_auc",
    "std_fold_macro_auc",
)

OPENPOM_BASELINE = 0.8872
DATASET_ID = "SamuelCagnani/olfacto-odor-138-ptbr"
MODEL_ID = "SamuelCagnani/olfacto-chemberta-138"
SPACE_ID = "SamuelCagnani/olfacto-live"


def _validate(metrics: dict) -> None:
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"metrics are missing required keys: {missing}")


def build_model_card(
    metrics: dict,
    base_model: str = "DeepChem/ChemBERTa-77M-MLM",
    dataset_id: str = DATASET_ID,
    ensemble: dict | None = None,
    baseline: dict | None = None,
) -> str:
    """Return a model card that only states metrics passed in."""
    _validate(metrics)
    low, high = metrics["macro_roc_auc_ci"]
    mean_fold = metrics["mean_fold_macro_auc"]
    std_fold = metrics["std_fold_macro_auc"]

    comparison_rows = (
        f"| ChemBERTa (este modelo) | {metrics['macro_roc_auc']:.4f} | "
        f"{metrics['micro_roc_auc']:.4f} |"
    )
    if baseline is not None:
        comparison_rows += (
            f"\n| Morgan + LogReg | {baseline['macro_roc_auc']:.4f} | "
            f"{baseline['micro_roc_auc']:.4f} |"
        )
    if ensemble is not None:
        weight = ensemble.get("weight_fingerprint", 0.8)
        comparison_rows += (
            f"\n| **Ensemble ({weight:.0%} fingerprint)** | "
            f"{ensemble['macro_roc_auc']:.4f} | {ensemble['micro_roc_auc']:.4f} |"
        )

    return f"""---
license: mit
language:
  - pt
  - en
library_name: transformers
base_model: {base_model}
datasets:
  - {dataset_id}
tags:
  - odor
  - olfaction
  - chemistry
  - smiles
  - chemberta
  - multi-label
metrics:
  - roc_auc
---

# Olfacto — ChemBERTa 138

Modelo aberto de predição de odor molecular: recebe um SMILES e retorna a probabilidade
de 138 descritores olfativos (frutado, floral, defumado, etc.). Parte do projeto
[Olfacto AI](https://github.com/cagnani/olfacto-ai), com interface em português.

Demo: [Olfacto Space](https://huggingface.co/spaces/SamuelCagnani/olfacto-live).
Artigos: [PT-BR no Hugging Face](https://huggingface.co/blog/SamuelCagnani/olfacto-ai-br) ·
[PT-BR](https://github.com/cagnani/olfacto-ai/blob/main/docs/article-ptbr.md) ·
[EN](https://github.com/cagnani/olfacto-ai/blob/main/docs/article-en.md).

## Uso pretendido

- Predição de qualidade olfativa de moléculas únicas a partir da estrutura.
- Pesquisa em olfação computacional e exploração do espaço olfativo.
- Educação e demonstrações em português.

## Fora de escopo

- Intensidade do odor (o dataset não anota intensidade).
- Misturas (o modelo recebe uma molécula por vez).
- Uso regulatório, médico ou de segurança alimentar.
- Substituir painel sensorial humano.

## Dados de treino

4.983 moléculas do dataset curado do OpenPOM, derivado dos catálogos GoodScents e
Leffingwell, com 138 rótulos binários. Rótulo mais comum: `fruity` (1.902); mais raro:
`chamomile` (31). Folds construídos com estratificação iterativa multi-label.

## Resultados

Avaliação out-of-fold (5 folds, um modelo por fold, `min_positives=20`):

| Métrica | Valor |
|---|---|
| ROC-AUC macro (OOF) | {metrics["macro_roc_auc"]:.4f} |
| IC 95% (bootstrap) | [{low:.4f}, {high:.4f}] |
| ROC-AUC micro | {metrics["micro_roc_auc"]:.4f} |
| PR-AUC macro | {metrics["macro_pr_auc"]:.4f} |
| AUC por fold (média±dp) | {mean_fold:.4f} ± {std_fold:.4f} |
| Rótulos avaliados | {metrics["n_labels"]} |

### Comparação de modelos

| Modelo | ROC-AUC macro (OOF) | ROC-AUC micro (OOF) |
|---|---|---|
{comparison_rows}

Referência: o OpenPOM reporta **{OPENPOM_BASELINE:.4f}** com 5-fold CV e ensemble de 10
modelos. A comparação é uma ablation, não uma alegação de superioridade. Os rótulos vêm de
catálogos e contêm ruído.

O arquivo `fingerprint.joblib` acompanha este repositório. O Space usa o ensemble
(80% fingerprint + 20% ChemBERTa), que supera o transformer sozinho neste dataset.

## Uso

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "{MODEL_ID}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

inputs = tokenizer("CCO", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits)[0]
```

## Limitações

- Prevê qualidade olfativa, não intensidade.
- Só moléculas únicas, não misturas.
- Viés para fragrâncias e aromas comerciais, não para a culinária brasileira.
- Rótulos ruidosos e desbalanceados; rótulos raros têm maior variância.
- Não valide cheiros desconhecidos: o modelo sempre escolhe os rótulos mais próximos.

## Citação

```bibtex
@misc{{olfacto2026,
  author = {{Samuel Cagnani}},
  title = {{Olfacto: open molecular odor prediction in Portuguese}},
  year = {{2026}},
  url = {{https://github.com/cagnani/olfacto-ai}}
}}
```

Dados originais: Lee et al. (2023), *A Principal Odor Map Unifies Diverse Tasks in Human
Olfactory Perception*, Science. Código de referência: OpenPOM.
"""


def build_dataset_card(n_molecules: int = 4983, n_labels: int = 138) -> str:
    """Return the dataset card for the curated odor dataset."""
    return f"""---
license: mit
language:
  - en
  - pt
task_categories:
  - text-classification
tags:
  - chemistry
  - odor
  - olfaction
  - smiles
  - multi-label
size_categories:
  - 1K<n<10K
---

# Olfacto — Molecular Odor Dataset (138 labels, PT-BR)

Dataset curado para predição de odor molecular, com {n_molecules} moléculas, {n_labels}
rótulos binários, léxico em português e uma galeria de ingredientes brasileiros.

Demo: [Olfacto Space](https://huggingface.co/spaces/SamuelCagnani/olfacto-live).
Artigo: [PT-BR no Hugging Face](https://huggingface.co/blog/SamuelCagnani/olfacto-ai-br).
Código: [github.com/cagnani/olfacto-ai](https://github.com/cagnani/olfacto-ai).

## Origem

Os rótulos derivam dos catálogos GoodScents e Leffingwell, agregados pelo
[OpenPOM](https://github.com/ARY2260/openpom) (licença MIT). O artigo de referência é
Lee et al. (2023), *A Principal Odor Map Unifies Diverse Tasks in Human Olfactory
Perception*, Science.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `curated_GS_LF_merged_4983.csv` | SMILES, descritores textuais e 138 rótulos binários |
| `odor_labels_ptbr.json` | tradução PT-BR e família olfativa dos 138 rótulos |
| `brazilian_ingredients.csv` | ~40 ingredientes brasileiros e moléculas marcantes |

## Esquema

```
nonStereoSMILES,descriptors,alcoholic,aldehydic,...,woody
CCO,fishy,0,0,...,0
```

A coluna `fold` é gerada pelo código com estratificação iterativa multi-label (seed 42).

## Estatísticas

| Propriedade | Valor |
|---|---|
| Moléculas | {n_molecules} |
| Rótulos | {n_labels} |
| Rótulos por molécula | média 4,88; máx. 21 |
| Rótulo mais comum | fruity (1.902) |
| Rótulo mais raro | chamomile (31) |

## Limitações

- Rótulos de catálogo, ruidosos e enviesados para fragrâncias comerciais.
- Sem anotação de intensidade.
- Sem misturas.

## Citação

```bibtex
@misc{{olfacto2026dataset,
  author = {{Samuel Cagnani}},
  title = {{Olfacto: molecular odor dataset with Portuguese lexicon}},
  year = {{2026}},
  url = {{https://github.com/cagnani/olfacto-ai}}
}}
```
"""


def metrics_from_summary(summary: dict) -> dict:
    """Flatten a ``cv_summary.json`` into the metrics dict used by the card."""
    oof = summary.get("oof")
    if not isinstance(oof, dict):
        raise ValueError("cross-validation summary must contain an 'oof' object")

    metrics = dict(oof)
    for key in ("mean_fold_macro_auc", "std_fold_macro_auc"):
        if key not in summary:
            raise ValueError(f"cross-validation summary is missing '{key}'")
        metrics[key] = summary[key]

    _validate(metrics)
    return metrics
