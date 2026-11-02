# Resultados — validação cruzada

Métricas geradas por `scripts/run_cv.py` e por este renderizador. Nenhum número é manual.

## Configuração

| Parâmetro | Valor |
|---|---|
| Backbone | `DeepChem/ChemBERTa-77M-MLM` |
| Comprimento máximo | 128 |
| Batch | 32 |
| Learning rate | 5e-05 |
| Épocas | 10 |
| Folds | 5 |
| Seed | 42 |

## AUC por fold

| Fold | Melhor época | ROC-AUC macro |
|---|---|---|
| 0 | 10 | 0.7515 |
| 1 | 10 | 0.7582 |
| 2 | 10 | 0.7607 |
| 3 | 10 | 0.7489 |
| 4 | 10 | 0.7544 |

Média dos folds: **0.7547 ± 0.0043**.

## Métricas out-of-fold

| Métrica | Valor |
|---|---|
| ROC-AUC macro (OOF) | 0.7516 |
| IC 95% (bootstrap) | [0.7465, 0.7574] |
| ROC-AUC micro | 0.8402 |
| PR-AUC macro | 0.1085 |
| Recall@5 | 0.3186 |
| Rótulos avaliados | 138 (mínimo de 20 positivos) |

## Comparação de modelos

| Modelo | Avaliação | ROC-AUC macro |
|---|---|---|
| OpenPOM (MPNN) | 5-fold CV, ensemble de 10 | 0.8872 |
| Olfacto (ChemBERTa) | 5-fold OOF, modelo único | 0.7516 |
| Morgan + LogReg | 5-fold OOF, baseline clássico | 0.8285 |
| Ensemble (80% fingerprint) | 5-fold OOF, blend | 0.8402 |

A comparação é uma ablation, não uma disputa de arquitetura. O OpenPOM usa um ensemble de
10 MPNNs; o ChemBERTa é um modelo único e pequeno. O baseline de fingerprints supera o
transformer neste dataset — um resultado honesto que motivou o ensemble usado no demo.

## Melhores rótulos

| Rótulo | Positivos | ROC-AUC |
|---|---|---|
| garlic | 120 | 0.9714 |
| sulfurous | 402 | 0.9694 |
| onion | 177 | 0.9673 |
| alliaceous | 126 | 0.9554 |
| cabbage | 46 | 0.9296 |
| hawthorn | 34 | 0.9252 |
| popcorn | 36 | 0.9129 |
| meaty | 299 | 0.9090 |
| roasted | 305 | 0.9066 |
| potato | 56 | 0.9034 |

## Piores rótulos

| Rótulo | Positivos | ROC-AUC |
|---|---|---|
| strawberry | 49 | 0.4607 |
| grape | 91 | 0.5080 |
| grapefruit | 74 | 0.5400 |
| plum | 61 | 0.5652 |
| earthy | 328 | 0.5704 |
| metallic | 134 | 0.5732 |
| caramellic | 234 | 0.5790 |
| milky | 87 | 0.5855 |
| creamy | 190 | 0.6006 |
| clean | 97 | 0.6097 |

Rótulos com poucos positivos têm maior variância; o filtro de `min_positives` existe para
não maquiar a média com casos degenerados.

## Como reproduzir

```bash
uv sync --extra train
uv run --extra train python scripts/run_cv.py --output-dir outputs/cv --final
uv run --extra train python scripts/render_results.py
```
