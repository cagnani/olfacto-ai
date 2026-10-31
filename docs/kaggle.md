# Rodando o treino no Kaggle

O `ChemBERTa-77M-MLM` é pequeno (3,5M parâmetros com a cabeça de classificação), então o
treino completo roda tanto em GPU T4 gratuita quanto em CPU. A GPU é recomendada para a
validação cruzada de 5 folds com `max_length=128`.

## Opção 1 — notebook

1. Crie um notebook novo em [kaggle.com/code](https://www.kaggle.com/code).
2. Em **Settings**, selecione `Accelerator = GPU T4 x1` e `Internet = On`.
3. Importe `notebooks/02_train_chemberta.ipynb` (File → Import Notebook).
4. Rode todas as células.

## Opção 2 — CLI

```bash
git clone https://github.com/cagnani/olfacto-ai.git
cd olfacto-ai
pip install -e ".[train]"
python scripts/run_cv.py --output-dir /kaggle/working/cv --epochs 10 --final
```

## Opção 3 — CPU local

```bash
make install
uv sync --extra train
uv run --extra train python scripts/run_cv.py \
  --output-dir outputs/cv --epochs 10 --max-length 128 --bootstrap 200 --final
```

Em um i7 de 12 threads, cada fold leva cerca de 10 minutos com `max_length=128`.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `cv_summary.json` | AUC por fold, média/desvio e métricas out-of-fold |
| `per_label.csv` | ROC-AUC e PR-AUC dos 138 rótulos |
| `oof_scores.npy` | probabilidades out-of-fold (4.983 × 138) |
| `oof_true.npy` | rótulos verdadeiros na mesma ordem |
| `final/` | modelo final + `labels.json` para publicação |

## Retomada

O script reutiliza `fold{k}/metrics.json` se ele já existir. Se uma sessão cair, rode de
novo com o mesmo `--output-dir` para continuar de onde parou.
