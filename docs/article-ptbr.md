# Ensinei uma IA a sentir cheiro — e coloquei em português

Em 2023, um time do Google Research e do Monell Chemical Senses Center publicou na
*Science* um modelo que prevê o cheiro de uma molécula a partir da estrutura dela. O
resultado é uma lista de descritores: frutado, floral, defumado, amadeirado. É olfação
computacional de verdade, e o código aberto que reproduz o paper se chama
[OpenPOM](https://github.com/ARY2260/openpom).

Eu decidi construir uma versão em português — e com um recorte que interessa a quem trabalha
com comida: **ingredientes brasileiros**.

Este é o [Olfacto AI](https://github.com/cagnani/olfacto-ai).

## O problema

Seu nariz tem cerca de 400 tipos de receptores. Cada molécula ativa uma combinação
diferente deles, e o cérebro lê essa combinação como "baunilha" ou "ovo podre". O modelo
pula o nariz e aprende diretamente da estrutura para as palavras.

A estrutura é escrita como SMILES, uma string curta. Etanol é `CCO`: carbono, carbono,
oxigênio. O modelo recebe só isso — e devolve a probabilidade de 138 descritores olfativos,
porque uma molécula pode ter vários cheiros ao mesmo tempo. É classificação multi-label,
não multi-classe.

## Os dados

O dataset curado do OpenPOM tem **4.983 moléculas** e **138 rótulos binários**, vindos dos
catálogos GoodScents e Leffingwell. A média é de 4,88 rótulos por molécula.

O desbalanceamento é brutal: `fruity` aparece em 1.902 moléculas, `chamomile` em 31. Se você
splitar aleatoriamente, os rótulos raros somem da validação. Por isso uso
**estratificação iterativa multi-label** (`iterative-stratification`): cada fold contém uma
amostra de todos os rótulos, inclusive os raros. E a loss é BCE com `pos_weight` por rótulo,
limitado a 20 — para o modelo não ignorar justamente os cheiros incomuns.

## O modelo

O paper usa uma MPNN (message passing neural network) sobre o grafo da molécula. Eu escolhi
um caminho diferente e deliberadamente mais simples: fine-tune do
[`DeepChem/ChemBERTa-77M-MLM`](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), um
transformer de SMILES com 3 camadas e 3,5 milhões de parâmetros com a cabeça de
classificação.

Por quê? Três motivos:

1. **Cabe no Hugging Face.** O modelo é nativo do ecossistema `transformers`: sobe com
   `from_pretrained`, roda em CPU e pode ser publicado com model card.
2. **É rápido e barato.** Treina em GPU gratuita e faz inferência em CPU — o Space não
   precisa de hardware pago.
3. **A comparação vira uma ablation honesta.** Um modelo pequeno e único contra o ensemble
   de 10 modelos do paper.

Configuração: `max_length=128`, batch 32, lr `5e-5`, warmup de 10%, early stopping na
macro-AUC de validação. Cinco folds, métricas out-of-fold.

## O que eu adicionei

Reproduzir não é portfólio. A contribuição original está em três camadas:

**1. Léxico em português.** Os 138 descritores traduzidos e agrupados em famílias
olfativas (floral, frutado, verde, amadeirado, especiado, doce, animal, químico, terroso,
alimentar). O modelo prevê `smoky`; a interface mostra **defumado**.

**2. Ingredientes do Brasil.** Cerca de 40 ingredientes com SMILES validado no PubChem:
café, cachaça, guaraná, cupuaçu, açaí, maracujá, abacaxi, manga, goiaba, cacau, canela,
cravo, baunilha, pimenta-do-reino, alho, cebola, erva-doce, hortelã, camomila, pequi,
jabuticaba, pitanga. Cada um aponta para uma molécula marcante do seu aroma.

**3. Mapa olfativo.** Os embeddings do modelo são projetados com UMAP. Moléculas que cheiram
parecido ficam perto no mapa — uma forma visual de explorar o espaço olfativo.

## Resultados

Validação cruzada de 5 folds, um modelo por fold, métricas calculadas out-of-fold:

| Métrica | Valor |
|---|---|
| ROC-AUC macro (OOF) | **0,7516** |
| IC 95% (bootstrap) | [0,7465; 0,7574] |
| ROC-AUC micro | 0,8402 |
| PR-AUC macro | 0,1085 |
| Recall@5 | 0,3186 |
| ROC-AUC por fold | 0,7547 ± 0,0043 |

O OpenPOM reporta **0,8872** com 5-fold CV e um ensemble de 10 MPNNs. O Olfacto é um modelo
único, menor, treinado em hardware gratuito. A diferença é esperada — e o objetivo não era
vencer o paper, era publicar um artefato aberto, em português e reproduzível.

### O que o modelo aprendeu (e o que não)

O resultado por rótulo é mais interessante que a média. Os melhores descritores são os que
têm assinatura química forte:

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

Faz sentido. Enxofre tem padrões estruturais claros e o catálogo é consistente ao rotular
alho e cebola. Já distinguir morango de framboesa depende de nuances que os rótulos de
catálogo tratam de forma ruidosa — e o modelo pequeno não dá conta.

O PR-AUC macro (0,1085) parece baixo, mas a linha de base também é: com rótulos raros, o
acaso fica perto da prevalência média (~0,035). O ROC-AUC micro (0,8402) e a análise por
rótulo contam uma história mais útil.

## O que isso não faz

- Prevê **qualidade** olfativa, não intensidade. O dataset não anota intensidade.
- Só moléculas únicas, não misturas. Um prato não é uma molécula.
- Os rótulos vêm de catálogos de fragrância: são ruidosos e enviesados para aromas
  comerciais, não para a culinária brasileira.
- O modelo sempre responde com os cheiros mais próximos que conhece. Ele não sabe dizer
  "nunca senti isso".

## Reprodutibilidade

Tudo está aberto:

- Código: [github.com/cagnani/olfacto-ai](https://github.com/cagnani/olfacto-ai)
- Modelo: [SamuelCagnani/olfacto-chemberta-138](https://huggingface.co/SamuelCagnani/olfacto-chemberta-138)
- Dataset: [SamuelCagnani/olfacto-odor-138-ptbr](https://huggingface.co/datasets/SamuelCagnani/olfacto-odor-138-ptbr)
- Demo: [Space Olfacto](https://huggingface.co/spaces/SamuelCagnani/olfacto)

O treino roda com `make cv`; os testes, com `make test`. O notebook do Kaggle reproduz a
validação cruzada em GPU T4 gratuita.

## O que eu aprendi

O maior ganho não veio do modelo, e sim de **tratar o desbalanceamento com seriedade**. Com
split aleatório, a macro-AUC parece boa e o modelo é inútil nos rótulos raros. Estratificar
e ponderar a loss mudou o comportamento de forma visível.

O segundo aprendizado foi de escopo: escolher ChemBERTa em vez de reproduzir a MPNN não foi
preguiça — foi o que permitiu publicar modelo, dataset e demo em duas semanas, com uma
comparação honesta contra o baseline. E o terceiro foi aceitar o resultado: 0,75 contra
0,89 é uma resposta, não um fracasso. O que importa é saber exatamente por que a diferença
existe.

Se você trabalha com comida, aroma ou química, o Space está aberto. Digite um SMILES, ou
escolha um ingrediente brasileiro, e veja o que a IA sente.
