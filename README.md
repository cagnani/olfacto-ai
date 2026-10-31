# Olfacto AI

Predição aberta de odor molecular, do SMILES ao aroma em português.

O Olfacto é um modelo de aprendizado de máquina que recebe a estrutura de uma molécula
(SMILES) e prevê quais dos 138 descritores olfativos ela provavelmente possui. O projeto
inclui um léxico em português, uma galeria de ingredientes brasileiros e um explorador
visual do espaço olfativo.

> Em construção. Métricas, modelo e demo serão publicados ao final da primeira fase.

## Stack

- `transformers` + `ChemBERTa` para classificação multi-label de SMILES
- `scikit-learn` + `iterative-stratification` para avaliação e splits
- `rdkit` para validação química e renderização
- `gradio` para o Hugging Face Space

## Uso rápido

```bash
make install
make test
make lint
```

## Licença

MIT. Os rótulos derivam dos catálogos GoodScents e Leffingwell, agregados pelo
[OpenPOM](https://github.com/ARY2260/openpom). Consulte a seção de citação no dataset.
