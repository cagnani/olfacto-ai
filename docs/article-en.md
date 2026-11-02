# Teaching an AI to smell — in Portuguese

In 2023, a team at Google Research and the Monell Chemical Senses Center published a model
in *Science* that predicts what a molecule smells like from its structure. The output is a
list of descriptors: fruity, floral, smoky, woody. This is real computational olfaction, and
the open-source replication is called [OpenPOM](https://github.com/ARY2260/openpom).

I built a Portuguese version of it, focused on something close to home: **Brazilian
ingredients**.

This is [Olfacto AI](https://github.com/cagnani/olfacto-ai).

## The problem

Your nose has about 400 types of smell receptors. Each molecule activates a different
combination of them, and your brain reads that combination as "vanilla" or "rotten eggs".
The model skips the nose and learns directly from structure to words.

Structure is written as SMILES, a short string. Ethanol is `CCO`: carbon, carbon, oxygen.
The model receives only that — and returns the probability of 138 odor descriptors, because
a molecule can smell like several things at once. It is multi-label classification, not
multi-class.

## The data

The curated OpenPOM dataset has **4,983 molecules** and **138 binary labels**, sourced from
the GoodScents and Leffingwell catalogues. The average is 4.88 labels per molecule.

The imbalance is brutal: `fruity` appears in 1,902 molecules, `chamomile` in 31. A random
split would make rare labels disappear from validation. So I use **iterative multi-label
stratification**: every fold contains a sample of every label, including rare ones. The loss
is BCE with a per-label `pos_weight`, capped at 20, so the model does not ignore uncommon
smells.

## The model

The paper uses an MPNN (message passing neural network) over the molecular graph. I chose a
different and deliberately simpler path: fine-tuning
[`DeepChem/ChemBERTa-77M-MLM`](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), a SMILES
transformer with 3 layers and 3.5 million parameters including the classification head.

Why?

1. **It fits Hugging Face.** The model is native to the `transformers` ecosystem: it loads
   with `from_pretrained`, runs on CPU and can be published with a model card.
2. **It is fast and cheap.** It trains on a free GPU and runs inference on CPU, so the Space
   does not need paid hardware.
3. **The comparison becomes an honest ablation.** A single small model against the paper's
   ensemble of 10.

Configuration: `max_length=128`, batch 32, lr `5e-5`, 10% warmup, early stopping on
validation macro-AUC. Five folds, out-of-fold metrics.

## What I added

Reproducing is not contributing. The original part is in three layers:

**1. A Portuguese lexicon.** The 138 descriptors translated and grouped into olfactory
families (floral, fruity, green, woody, spicy, sweet, animal, chemical, earthy, food). The
model predicts `smoky`; the interface shows **defumado**.

**2. Brazilian ingredients.** About 40 ingredients with SMILES validated against PubChem:
coffee, cachaça, guaraná, cupuaçu, açaí, passion fruit, pineapple, mango, guava, cocoa,
cinnamon, clove, vanilla, black pepper, garlic, onion, fennel, mint, chamomile, pequi,
jabuticaba, pitanga. Each one points to a marker molecule of its aroma.

**3. An olfactory map.** The model embeddings are projected with UMAP. Molecules that smell
alike end up close on the map — a visual way to explore odor space.

## Results

Five-fold cross-validation, out-of-fold metrics. ChemBERTa alone:

| Metric | Value |
|---|---|
| Macro ROC-AUC (OOF) | 0.7516 |
| 95% CI (bootstrap) | [0.7465, 0.7574] |
| Micro ROC-AUC | 0.8402 |
| Macro PR-AUC | 0.1085 |
| Recall@5 | 0.3186 |
| ROC-AUC per fold | 0.7547 ± 0.0043 |

OpenPOM reports **0.8872** with 5-fold CV and an ensemble of 10 MPNNs. ChemBERTa is a single,
smaller model trained on free hardware. The gap was expected.

### The baseline that beat the transformer

Before wrapping up, I ran the most boring baseline possible: 2,048-bit Morgan fingerprints
(radius 2) plus a balanced logistic regression per label. No transformer, no GPU, no Hugging
Face.

| Model | Macro ROC-AUC (OOF) |
|---|---|
| OpenPOM (MPNN, ensemble of 10) | 0.8872 |
| **Ensemble (80% fingerprint + 20% ChemBERTa)** | **0.8402** |
| Morgan + LogReg | 0.8285 |
| ChemBERTa (single model) | 0.7516 |

The classical baseline **beat** the small transformer by 7.7 AUC points. The ensemble of both
reached 0.8402, with a 95% CI of [0.8355, 0.8447] and micro ROC-AUC of 0.8956.

Why? With 4,983 molecules, a 3-layer transformer has little signal to learn olfactory
representations from scratch. Fingerprints encode local substructures that are already
predictive of smell, and logistic regression handles sparse high-dimensional vectors well.
This is the kind of result you only find when you actually run the baseline — and many people
skip it.

The demo uses the ensemble, with an 80% weight for the fingerprint chosen by out-of-fold
validation. ChemBERTa remains published as an open model and as an ablation.

### What the model learned (and what it did not)

The per-label result is more interesting than the average. ChemBERTa's best descriptors are
the ones with a strong chemical signature:

| Label | Positives | ROC-AUC |
|---|---|---|
| garlic | 120 | 0.971 |
| sulfurous | 402 | 0.969 |
| onion | 177 | 0.967 |
| alliaceous | 126 | 0.955 |
| cabbage | 46 | 0.930 |

The worst are fine fruits:

| Label | Positives | ROC-AUC |
|---|---|---|
| strawberry | 49 | 0.461 |
| grape | 91 | 0.508 |
| grapefruit | 74 | 0.540 |
| plum | 61 | 0.565 |

It makes sense. Sulfur has clear structural patterns, and the catalogue is consistent when
labeling garlic and onion. Distinguishing strawberry from raspberry depends on nuances that
catalogue labels treat noisily — and neither small model handles them well.

Macro PR-AUC (0.1085) looks low, but the baseline is low too: with rare labels, chance sits
near the average prevalence (~0.035). Micro ROC-AUC and the per-label analysis tell a more
useful story.

## What this does not do

- It predicts odor **quality**, not intensity. The dataset has no intensity annotations.
- It handles single molecules, not mixtures. A dish is not a molecule.
- The labels come from fragrance catalogues: they are noisy and biased toward commercial
  aromas, not Brazilian cooking.
- The model always answers with the closest smells it knows. It cannot say "I have never
  smelled this".

## Reproducibility

Everything is open:

- Code: [github.com/cagnani/olfacto-ai](https://github.com/cagnani/olfacto-ai)
- Model: [SamuelCagnani/olfacto-chemberta-138](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138)
- Dataset: [SamuelCagnani/olfacto-odor-138-ptbr](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr)
- Demo: [Olfacto Space](https://huggingface.co/spaces/SamuelCagnani/olfacto)

Training runs with `make cv`; tests with `make test`. The Kaggle notebook reproduces the
cross-validation on a free T4 GPU.

## What I learned

The biggest gain did not come from the model, but from **taking imbalance seriously**. With a
random split, macro-AUC looks fine and the model is useless on rare labels. Stratifying and
weighting the loss visibly changed its behavior.

The second lesson was about scope: choosing ChemBERTa instead of reproducing the MPNN was not
laziness — it is what made it possible to publish a model, a dataset and a demo in a few
days.

The third was the most important: **always run the dumb baseline**. I was about to close the
project at 0.75 AUC with an elegant explanation about the limits of a small transformer. The
Morgan + logistic regression baseline hit 0.83 in 17 minutes of CPU. Had I skipped it, I
would have published a beautiful and wrong story about what was possible with the data.

Accepting the result was part of it too: an ensemble at 0.84 against OpenPOM's 0.89 is an
answer, not a failure. What matters is knowing exactly why the gap exists.

If you work with food, aroma or chemistry, the Space is open. Type a SMILES, or pick a
Brazilian ingredient, and see what the model smells.
