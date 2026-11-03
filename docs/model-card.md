---
license: mit
language:
  - pt
  - en
library_name: transformers
base_model: DeepChem/ChemBERTa-77M-MLM
datasets:
  - SamuelCagnani/olfacto-odor-138-ptbr
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
| ROC-AUC macro (OOF) | 0.7516 |
| IC 95% (bootstrap) | [0.7465, 0.7574] |
| ROC-AUC micro | 0.8402 |
| PR-AUC macro | 0.1085 |
| AUC por fold (média±dp) | 0.7547 ± 0.0043 |
| Rótulos avaliados | 138 |

### Comparação de modelos

| Modelo | ROC-AUC macro (OOF) | ROC-AUC micro (OOF) |
|---|---|---|
| ChemBERTa (este modelo) | 0.7516 | 0.8402 |
| Morgan + LogReg | 0.8285 | 0.8860 |
| **Ensemble (80% fingerprint)** | 0.8402 | 0.8956 |

Referência: o OpenPOM reporta **0.8872** com 5-fold CV e ensemble de 10
modelos. A comparação é uma ablation, não uma alegação de superioridade. Os rótulos vêm de
catálogos e contêm ruído.

O arquivo `fingerprint.joblib` acompanha este repositório. O Space usa o ensemble
(80% fingerprint + 20% ChemBERTa), que supera o transformer sozinho neste dataset.

## Uso

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "SamuelCagnani/olfacto-chemberta-138"
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
@misc{olfacto2026,
  author = {Samuel Cagnani},
  title = {Olfacto: open molecular odor prediction in Portuguese},
  year = {2026},
  url = {https://github.com/cagnani/olfacto-ai}
}
```

Dados originais: Lee et al. (2023), *A Principal Odor Map Unifies Diverse Tasks in Human
Olfactory Perception*, Science. Código de referência: OpenPOM.
