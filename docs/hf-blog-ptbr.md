**TL;DR**

- 4.983 moléculas, 138 descritores olfativos, validação 5-fold out-of-fold.
- Fine-tune de ChemBERTa (3,5M de parâmetros) rodando em CPU.
- Ensemble com fingerprints Morgan: **0,8402** de ROC-AUC macro (micro 0,8956).
- O baseline clássico de fingerprints + regressão logística (0,8285) **venceu** o transformer sozinho (0,7516).

**Links:** [Modelo](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138) · [Dataset](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr) · [Demo interativo](https://huggingface.co/spaces/SamuelCagnani/olfacto-live) · [Código](https://github.com/cagnani/olfacto-ai)

## O problema

Em 2023, um time do Google Research e do Monell Chemical Senses Center publicou na *Science* um modelo que prevê o cheiro de uma molécula a partir da estrutura dela. A saída é uma lista de descritores: frutado, floral, defumado, amadeirado. O código aberto que reproduz o paper é o [OpenPOM](https://github.com/ARY2260/openpom).

Eu construí uma versão em português, com um recorte que interessa a quem trabalha com comida: **ingredientes brasileiros**.

Seu nariz tem cerca de 400 tipos de receptores. Cada molécula ativa uma combinação diferente deles, e o cérebro lê essa combinação como "baunilha" ou "ovo podre". O modelo pula o nariz e aprende diretamente da estrutura para as palavras.

A estrutura é escrita como SMILES, uma string curta. Etanol é `CCO`: carbono, carbono, oxigênio. O modelo recebe só isso e devolve a probabilidade de 138 descritores, porque uma molécula pode ter vários cheiros ao mesmo tempo. É classificação multi-label, não multi-classe.

## Os dados

O dataset curado do OpenPOM tem **4.983 moléculas** e **138 rótulos binários**, vindos dos catálogos GoodScents e Leffingwell. A média é de 4,88 rótulos por molécula.

O desbalanceamento é brutal: `fruity` aparece em 1.902 moléculas, `chamomile` em 31. Um split aleatório faria os rótulos raros sumirem da validação. Por isso uso **estratificação iterativa multi-label**: cada fold contém uma amostra de todos os rótulos. A loss é BCE com `pos_weight` por rótulo, limitado a 20, para o modelo não ignorar os cheiros incomuns.

## O modelo

O paper usa uma MPNN (message passing neural network) sobre o grafo da molécula. Eu escolhi um caminho diferente e deliberadamente mais simples: fine-tune do [`DeepChem/ChemBERTa-77M-MLM`](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), um transformer de SMILES com 3 camadas e 3,5 milhões de parâmetros com a cabeça de classificação.

Por quê?

1. **Cabe no Hugging Face.** O modelo é nativo do ecossistema `transformers`: carrega com `from_pretrained`, roda em CPU e pode ser publicado com model card.
2. **É rápido e barato.** Treina em GPU gratuita e faz inferência em CPU.
3. **A comparação vira uma ablation honesta.** Um modelo pequeno e único contra o ensemble de 10 do paper.

Configuração: `max_length=128`, batch 32, lr `5e-5`, warmup de 10%, early stopping na macro-AUC de validação. Cinco folds, métricas out-of-fold.

## O que eu adicionei

Reproduzir não é contribuição. A parte original está em três camadas:

**1. Léxico em português.** Os 138 descritores traduzidos e agrupados em famílias olfativas (floral, frutado, verde, amadeirado, especiado, doce, animal, químico, terroso, alimentar). O modelo prevê `smoky`; a interface mostra **defumado**.

**2. Ingredientes do Brasil.** Cerca de 40 ingredientes com SMILES validado no PubChem: café, cachaça, guaraná, cupuaçu, açaí, maracujá, abacaxi, manga, goiaba, cacau, canela, cravo, baunilha, pimenta-do-reino, alho, cebola, erva-doce, hortelã, camomila, pequi, jabuticaba, pitanga. Cada um aponta para uma molécula marcante do seu aroma.

![Ingredientes brasileiros e aromas previstos](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/ingredients.png)

**3. Mapa olfativo.** Os embeddings do modelo são projetados com UMAP. Moléculas que cheiram parecido ficam perto no mapa.

![Mapa UMAP do espaço olfativo](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/odor-map.png)

## Resultados

Validação cruzada de 5 folds, métricas out-of-fold. O ChemBERTa sozinho:

| Métrica | Valor |
|---|---|
| ROC-AUC macro (OOF) | 0,7516 |
| IC 95% (bootstrap) | [0,7465; 0,7574] |
| ROC-AUC micro | 0,8402 |
| PR-AUC macro | 0,1085 |
| Recall@5 | 0,3186 |
| ROC-AUC por fold | 0,7547 ± 0,0043 |

O OpenPOM reporta **0,8872** com 5-fold CV e um ensemble de 10 MPNNs. O ChemBERTa é um modelo único, menor, treinado em hardware gratuito. A diferença era esperada.

### O baseline que venceu o transformer

Antes de fechar, rodei o baseline mais chato possível: fingerprints Morgan de 2.048 bits (raio 2) + uma regressão logística balanceada por rótulo. Sem transformer, sem GPU, sem Hugging Face.

| Modelo | ROC-AUC macro (OOF) |
|---|---|
| OpenPOM (MPNN, ensemble de 10) | 0,8872 |
| **Ensemble (80% fingerprint + 20% ChemBERTa)** | **0,8402** |
| Morgan + LogReg | 0,8285 |
| ChemBERTa (modelo único) | 0,7516 |

![Comparação de modelos](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr/resolve/main/images/metrics.png)

O baseline clássico **venceu** o transformer pequeno por 7,7 pontos de AUC. O ensemble dos dois chegou a 0,8402, com IC 95% [0,8355; 0,8447] e ROC-AUC micro 0,8956.

Por quê? Com 4.983 moléculas, um transformer de 3 camadas tem pouco sinal para aprender representações olfativas do zero. Fingerprints codificam subestruturas locais que já são preditivas de odor, e a regressão logística lida bem com vetores esparsos de alta dimensão. É o tipo de resultado que só aparece quando você roda o baseline — e que muita gente pula.

O demo usa o ensemble, com peso de 80% para o fingerprint escolhido por validação out-of-fold. O ChemBERTa continua publicado como modelo aberto e como ablation.

### O que o modelo aprendeu (e o que não)

O resultado por rótulo é mais interessante que a média. Os melhores descritores do ChemBERTa são os que têm assinatura química forte:

| Rótulo | Positivos | ROC-AUC |
|---|---|---|
| garlic | 120 | 0,971 |
| sulfurous | 402 | 0,969 |
| onion | 177 | 0,967 |
| alliaceous | 126 | 0,955 |
| cabbage | 46 | 0,930 |

Os piores são frutas finas:

| Rótulo | Positivos | ROC-AUC |
|---|---|---|
| strawberry | 49 | 0,461 |
| grape | 91 | 0,508 |
| grapefruit | 74 | 0,540 |
| plum | 61 | 0,565 |

Faz sentido. Enxofre tem padrões estruturais claros e o catálogo é consistente ao rotular alho e cebola. Já distinguir morango de framboesa depende de nuances que os rótulos de catálogo tratam de forma ruidosa — e nenhum dos modelos pequenos dá conta.

O PR-AUC macro (0,1085) parece baixo, mas a linha de base também é: com rótulos raros, o acaso fica perto da prevalência média (~0,035). O ROC-AUC micro e a análise por rótulo contam uma história mais útil.

## Limitações

- Prevê **qualidade** olfativa, não intensidade. O dataset não anota intensidade.
- Só moléculas únicas, não misturas. Um prato não é uma molécula.
- Os rótulos vêm de catálogos de fragrância: são ruidosos e enviesados para aromas comerciais, não para a culinária brasileira.
- O modelo sempre responde com os cheiros mais próximos que conhece. Ele não sabe dizer "nunca senti isso".

## Reprodutibilidade

- Código: [github.com/cagnani/olfacto-ai](https://github.com/cagnani/olfacto-ai)
- Modelo: [SamuelCagnani/olfacto-chemberta-138](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138)
- Dataset: [SamuelCagnani/olfacto-odor-138-ptbr](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr)
- Demo: [SamuelCagnani/olfacto-live](https://huggingface.co/spaces/SamuelCagnani/olfacto-live)

O treino roda com `make cv`; os testes, com `make test`. O notebook do Kaggle reproduz a validação cruzada em GPU T4 gratuita.

## O que eu aprendi

O maior ganho não veio do modelo, e sim de **tratar o desbalanceamento com seriedade**. Com split aleatório, a macro-AUC parece boa e o modelo é inútil nos rótulos raros. Estratificar e ponderar a loss mudou o comportamento de forma visível.

O segundo aprendizado foi de escopo: escolher ChemBERTa em vez de reproduzir a MPNN não foi preguiça — foi o que permitiu publicar modelo, dataset e demo em poucos dias.

O terceiro foi o mais importante: **sempre rode o baseline burro**. Eu ia fechar o projeto com 0,75 de AUC e uma explicação elegante sobre limitações de modelo pequeno. O Morgan + regressão logística fez 0,83 em 17 minutos de CPU. Se eu não tivesse rodado, teria publicado uma história bonita e errada sobre o que era possível com os dados.

Aceitar o resultado também fez parte: 0,84 de ensemble contra 0,89 do OpenPOM é uma resposta, não um fracasso. O que importa é saber exatamente por que a diferença existe.
