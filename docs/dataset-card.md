---
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

Dataset curado para predição de odor molecular, com 4983 moléculas, 138
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
| Moléculas | 4983 |
| Rótulos | 138 |
| Rótulos por molécula | média 4,88; máx. 21 |
| Rótulo mais comum | fruity (1.902) |
| Rótulo mais raro | chamomile (31) |

## Limitações

- Rótulos de catálogo, ruidosos e enviesados para fragrâncias comerciais.
- Sem anotação de intensidade.
- Sem misturas.

## Citação

```bibtex
@misc{olfacto2026dataset,
  author = {Samuel Cagnani},
  title = {Olfacto: molecular odor dataset with Portuguese lexicon},
  year = {2026},
  url = {https://github.com/cagnani/olfacto-ai}
}
```
