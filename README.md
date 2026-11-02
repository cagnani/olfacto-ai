# Olfacto AI

Predição aberta de odor molecular, do SMILES ao aroma em português.

[![CI](https://github.com/cagnani/olfacto-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cagnani/olfacto-ai/actions/workflows/ci.yml)

O Olfacto recebe a estrutura de uma molécula (SMILES) e prevê a probabilidade de 138
descritores olfativos — frutado, floral, defumado, amadeirado e assim por diante. O projeto
inclui um léxico em português, uma galeria de ingredientes brasileiros e um explorador
visual do espaço olfativo.

Artigo técnico: [português](docs/article-ptbr.md) · [inglês](docs/article-en.md).

| Artefato | Link |
|---|---|
| Modelo | [SamuelCagnani/olfacto-chemberta-138](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138) |
| Dataset | [SamuelCagnani/olfacto-odor-138-ptbr](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr) |
| Demo | [Space Olfacto](https://huggingface.co/spaces/SamuelCagnani/olfacto) |

## Como funciona

```
SMILES → tokenização por caractere → ChemBERTa (3 camadas, 384 dims)
       → cabeça multi-label (138 saídas) → sigmoide → aromas em PT-BR
```

- **Dados:** 4.983 moléculas do dataset curado do [OpenPOM](https://github.com/ARY2260/openpom),
  derivado dos catálogos GoodScents e Leffingwell. Média de 4,88 rótulos por molécula.
- **Modelo:** fine-tune de `DeepChem/ChemBERTa-77M-MLM`, loss BCE com `pos_weight` por rótulo
  (limitado a 20) para lidar com o desbalanceamento extremo (`fruity` 1.902 vs `chamomile` 31).
- **Ensemble:** fingerprints Morgan (2048 bits, raio 2) + regressão logística por rótulo,
  combinados com o transformer (peso 0.8 para o fingerprint, escolhido out-of-fold).
- **Validação:** 5-fold com estratificação iterativa multi-label, métricas out-of-fold.
- **Interpretação:** embeddings CLS projetados com UMAP para o mapa de cheiros.

## Uso rápido

```bash
make install
make test
make lint
```

Inferência:

```python
from olfacto.predict import OdorPredictor

predictor = OdorPredictor.from_pretrained(
    "SamuelCagnani/olfacto-chemberta-138",
    "data/odor_labels_ptbr.json",
)
for prediction in predictor.predict("CCO", top_k=5):
    print(f"{prediction.label_pt_br}: {prediction.probability:.2f}")
```

## Resultados

Validação cruzada de 5 folds, métricas out-of-fold:

| Modelo | ROC-AUC macro |
|---|---|
| OpenPOM (MPNN, ensemble de 10) | 0.8872 |
| **Olfacto Ensemble (80% fingerprint + 20% ChemBERTa)** | **0.8402** |
| Morgan + LogReg | 0.8285 |
| ChemBERTa (modelo único) | 0.7516 |

O achado central: neste dataset, um baseline clássico de fingerprints Morgan + regressão
logística **supera** o transformer pequeno. O ensemble recupera o melhor dos dois e é o que
roda no Space. O ChemBERTa sozinho tem ROC-AUC micro 0.8402 e por rótulo vai bem em notas
sulfurosas (alho 0.971, cebola 0.967) e mal em frutas finas (morango 0.461, uva 0.508).

A comparação com o OpenPOM é uma ablation, não uma disputa de arquitetura. Tabela completa
em [`docs/results.md`](docs/results.md).

## Reproduzir o treino

```bash
uv sync --extra train
uv run --extra train python scripts/run_cv.py --output-dir outputs/cv --final
```

No Kaggle, use `notebooks/02_train_chemberta.ipynb` (GPU T4) ou siga `docs/kaggle.md`.
Resultados completos em [`docs/results.md`](docs/results.md).

## Estrutura

```
src/olfacto/   data, labels, train, evaluate, predict, map, ingredients, cards
configs/       hiperparâmetros
scripts/       cross-validation, mapa, publicação no HF
space/         app Gradio
tests/         testes em CPU, sem rede
```

## Limitações

- Prevê **qualidade** olfativa, não intensidade.
- Só moléculas únicas, não misturas.
- Rótulos de catálogo são ruidosos e enviesados para fragrâncias comerciais.
- Não é adequado para uso regulatório, médico ou de segurança alimentar.
- O modelo sempre retorna os aromas mais próximos que conhece, mesmo para cheiros novos.

## Citação e licença

Dados originais: Lee et al. (2023), *A Principal Odor Map Unifies Diverse Tasks in Human
Olfactory Perception*, Science. Código de referência: OpenPOM (MIT).

Este projeto é MIT. Veja `docs/results.md` para a comparação com o baseline do paper.
