[English](README.md) · [Français](README.fr.md) · **Português**

# Predição de odor molecular com redes neurais em grafos

**Predição multi-label de 138 descritores de odor diretamente a partir de SMILES moleculares, replicando e superando benchmarks publicados.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.x-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2024.03-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Visão geral

Mapear a estrutura química de uma molécula para o odor percebido pelo ser humano — a **relação estrutura-odor (SOR)** — é um problema de longa data na informática olfativa, com aplicações diretas na indústria de aromas e fragrâncias e na neurociência sensorial.

Este projeto constrói um pipeline de aprendizado profundo em grafos que recebe uma **string SMILES** e prediz um **vetor binário de 138 dimensões** de descritores de odor (*floral*, *amadeirado*, *frutado*, …). A tarefa é um problema de classificação multi-label extrema: cerca de **97% da matriz de rótulos é composta por zeros**, já que uma molécula típica carrega apenas 3 ou 4 descritores.

O trabalho tem dois objetivos:

1. **Replicar** as arquiteturas de GNN descritas na literatura de machine learning olfativo (em especial Sanchez-Lengeling et al., 2019).
2. **Estendê-las** com um pipeline rigoroso de tratamento de dados — *estratificação iterativa de segunda ordem* e *otimização de thresholds por rótulo* — e avaliar uma arquitetura adicional (GATv2) ausente desses trabalhos.

O MPNN totalmente otimizado alcança **AUROC de 0,9001** e **F1-Score de 0,3895**, superando o benchmark publicado de 0,362.

> Desenvolvido como o projeto *GenAI for Molecular Discovery* na **ENSEA** (École Nationale Supérieure de l'Électronique et de ses Applications), sob supervisão do Prof. Guillaume Renton. Escrito originalmente no Google Colab e reorganizado neste repositório.

---

## Principais resultados

### Replicação da literatura e extensões

| Modelo | AUROC | Precisão | Recall | F1-Score |
|---|---|---|---|---|
| GCN *(benchmark do artigo)* | 0,8900 | 0,3790 | 0,3870 | 0,3620 |
| GCN — replicação ingênua | 0,8464 | 0,1008 | 0,8109 | 0,1793 |
| GCN — **+ estratificação + thresholds** | 0,8561 | 0,2616 | 0,5158 | 0,3471 |
| MPNN *(benchmark do artigo)* | 0,8940 | 0,3790 | 0,3900 | 0,3600 |
| MPNN — replicação ingênua | 0,8642 | 0,1216 | 0,6023 | 0,2023 |
| MPNN — **+ estratificação + thresholds** | **0,9001** | 0,3090 | 0,5270 | **0,3895** |
| GATv2 *(nossa extensão)* | 0,8310 | 0,0713 | 0,7128 | 0,1296 |
| GATv2 — + estratificação + thresholds | 0,8561 | 0,2436 | 0,5042 | 0,3284 |

**Conclusões**

- Uma replicação ingênua das arquiteturas publicadas fica muito aquém das métricas reportadas (F1 de 0,18 a 0,20 contra 0,36). A arquitetura sozinha não reproduz os resultados — **é o pipeline de tratamento de dados que fecha a lacuna.**
- A estratificação iterativa de segunda ordem somada à otimização de thresholds por rótulo **praticamente dobra o F1-score** de todas as arquiteturas testadas, sem alterar um único peso do modelo.
- O MPNN otimizado **supera o benchmark publicado em F1** e se situa na faixa de AUROC do estado da arte. Nossa precisão é ligeiramente inferior à da literatura (0,309 contra 0,379), mas o recall é substancialmente maior (0,527 contra 0,387) — o modelo recupera uma parcela muito maior dos rótulos de odor verdadeiros.
- Um diagnóstico central: a **loss de validação diverge** (a calibração das probabilidades se deteriora) enquanto a **AUROC permanece em um platô estável**. A capacidade de ranqueamento sobrevive à degradação da calibração, o que explica exatamente por que a otimização posterior de thresholds resgata um modelo que parece quebrado no corte padrão de 0,5.
- **A AUROC micro-média esconde grande dispersão entre rótulos.** A AUROC por classe tem média de 0,773 (GCN) e 0,797 (GATv2), com os melhores descritores acima de 0,99 e os piores abaixo de 0,15: odores frequentes são bem aprendidos, os raros não.

### Ablação do GCN baseline — função de readout e profundidade

| Readout | Camadas | Hamming Loss | Micro-F1 | Subset Accuracy |
|---|---|---|---|---|
| Global mean pool | 2 | 0,1907 | 0,1291 | 0,0000 |
| Global mean pool | 3 | 0,1815 | 0,1436 | 0,0421 |
| Global mean pool | 4 | 0,1080 | 0,1954 | 0,0641 |
| Global max pool | 2 | 0,1616 | 0,1600 | 0,0401 |
| Global max pool | 3 | 0,1023 | 0,1915 | 0,0561 |
| Global max pool | 4 | 0,0784 | **0,2259** | 0,0641 |

O max pooling supera consistentemente o mean pooling — algo plausível em olfação, onde um único grupo funcional reativo pode dominar todo o perfil de aroma —, mas ao custo de uma **loss de validação bem menos estável**. O mean pooling converge de forma suave; o max pooling oscila enquanto pontua mais alto.

### Escopo deste repositório

O projeto foi conduzido por uma equipe de três pessoas. Os notebooks publicados aqui cobrem a **ablação do GCN baseline**, o **GCN que aproxima a literatura**, a extensão **GATv2** e o **MPNN**.

Uma ressalva sobre procedência: o notebook do MPNN é uma **reimplementação escrita a partir da especificação arquitetural do relatório**, não o código de treinamento original, escrito por outro integrante da equipe. As métricas do MPNN na tabela acima vêm daquela execução original. Reexecutar o notebook produz valores na mesma faixa, sem reproduzi-los exatamente.

---

## Conjunto de dados

Combinação de duas fontes curadas por especialistas, amplamente utilizadas na indústria de aromas e fragrâncias:

- **The Good Scents Company Information System** — descrições qualitativas de odor provenientes de avaliação sensorial especializada.
- **Leffingwell PMP 2001** — descritores de odor padronizados computacionalmente.

| Propriedade | Valor |
|---|---|
| Moléculas | 4.983 |
| Descritores de odor (rótulos) | 138 |
| Rótulos por molécula (típico) | 3–4 |
| Proporção de zeros na matriz de rótulos | ~97% |
| Elementos químicos distintos | 25 |
| Representação de entrada | SMILES → grafo molecular |

Os notebooks baixam o conjunto de dados automaticamente — nenhuma configuração manual é necessária. Veja [`data/README.md`](data/data_README.md) para os detalhes.

### O desafio central: inflação de zeros

Com 97% de negativos, um modelo que minimiza a loss global aprende a prever "sem odor" para tudo: acurácia alta, utilidade prática nula. Dois mecanismos endereçam isso:

- **Loss BCE ponderada** — um `pos_weight` por classe, derivado da frequência dos rótulos no conjunto de treino, penaliza descritores raros não detectados.
- **Estratificação iterativa de segunda ordem** — a divisão aleatória padrão destrói as razões de coocorrência dos rótulos em dados multi-label: isolar uma molécula para satisfazer a distribuição de um rótulo perturba a de todos os rótulos que aparecem junto com ele. Esse algoritmo preserva as razões de *pares de rótulos* entre treino, validação e teste, distribuindo primeiro as combinações mais raras, de modo que perfis de aroma ultrarraros não fiquem fora da avaliação.

---

## Metodologia

### 1. SMILES → grafo

Cada string SMILES é convertida em grafo topológico com `from_smiles` do PyTorch Geometric (que usa o RDKit internamente): **átomos como nós, ligações como arestas**, hidrogênios implícitos.

- **Features dos nós (9 por átomo):** número atômico, quiralidade, grau, carga formal, número de hidrogênios, elétrons radicalares, hibridização, aromaticidade e pertencimento a anel.
- **Features das arestas:** tipo de ligação, estereoquímica, conjugação. *(Produzidas pelo featurizer, mas não consumidas pelos modelos GCN e GATv2, que operam apenas sobre a conectividade.)*
- **Alvo:** vetor binário de comprimento 138.
- **Batch size:** 32.

### 2. Arquiteturas

**GCN baseline** — camadas `GCNConv` sequenciais com ReLU, em profundidades de 2, 3 e 4 (dimensões ocultas 64 → 128 → 256 → 128), pooling global e uma única camada linear de classificação. Usado para estudar profundidade e readout e para estabelecer o compromisso entre oversmoothing e expressividade.

**GCN aproximando a literatura** — projetado para espelhar o modelo otimizado de Sanchez-Lengeling et al.:
- 4 camadas GCN com dimensões ocultas `[15, 20, 27, 36]`.
- **Readout Jumping Knowledge:** as features de entrada e cada camada intermediária passam por sum pooling individualmente e são concatenadas em um embedding de ~175 dimensões, retendo tanto o detalhe atômico de baixo nível quanto a estrutura de alto nível.
- MLP de 2 camadas `[96, 63]` com Batch Normalization, ReLU e dropout de 0,47.
- Camada linear final → 138 rótulos.

**GATv2** *(nossa extensão além da literatura)* — duas camadas `GATv2Conv`: a primeira com 4 cabeças de atenção (dimensão oculta 64), saídas concatenadas, ELU e dropout de 0,3; a segunda com cabeça única. Pooling médio global e, em seguida, camada linear final. O GATv2 corrige a limitação de atenção estática do GAT original aplicando a transformação linear *depois* de concatenar as features dos nós de origem e destino, permitindo ranqueamento de atenção totalmente dinâmico — relevante para grafos moleculares em que certos grupos funcionais pesam de forma desproporcional.

**MPNN** — a arquitetura descrita por Sanchez-Lengeling et al.: 5 passos de passagem de mensagens, dimensão oculta 43, mensagens por multiplicação matricial condicionada às arestas (`NNConv`), função de atualização GRU, readout por soma com portão de 197 dimensões e caminho residual, seguido de uma rede totalmente conectada de 3 camadas (392 unidades, ReLU, batch norm, dropout de 0,12). É o **único modelo aqui que consome as features das ligações** e o único cujos estados dos nós carregam memória entre os passos de mensagem — duas razões plausíveis para superar as outras duas arquiteturas sobre os mesmos dados. *(Veja a nota de procedência em Escopo.)*

### 3. Treinamento

- Loss: `BCEWithLogitsLoss` com `pos_weight` por classe calculado sobre o conjunto de treino.
- Otimizador: Adam, taxa de aprendizado inicial de 0,001.
- Modelos baseline: early stopping sobre a loss de validação, paciência de 50.
- Arquiteturas avançadas: Cosine Annealing com Warm Restarts (`T_0=50`), estendido para 500 épocas no GCN otimizado final, cuja loss ainda decrescia em 300.

### 4. Avaliação

Acurácia não tem significado sob inflação de zeros, então a avaliação usa **AUROC** (ranqueamento independente de threshold), **AUPRC** (sensível ao desbalanceamento), **precisão**, **recall**, **Micro-F1** e Hamming Loss quando necessário para comparação direta com as tabelas publicadas.

**Otimização de thresholds por rótulo:** em vez de um corte fixo em 0,5, os thresholds de decisão são varridos **independentemente para cada um dos 138 rótulos** no conjunto de validação a fim de maximizar o F1, depois congelados e aplicados ao conjunto de teste. É a etapa de pós-processamento de maior impacto no pipeline: no GCN da literatura, eleva o F1 de 0,2604 (melhor threshold global único, 0,75) para 0,3471.

Os thresholds resultantes são, em si, interpretáveis — o modelo estabelece um corte rigoroso para os descritores que detecta com confiança (*alcoholic*, *malty*, *bergamot* ≈ 0,93) e um corte permissivo para aqueles com que tem dificuldade (*sulfurous* ≈ 0,34).

---

## Estrutura do repositório

```
.
├── notebooks/
│   ├── 01_gcn_implementation.ipynb    # Pré-processamento, ablação do GCN, GCN da literatura
│   ├── 02_gatv2.ipynb                 # Extensão baseada em atenção
│   └── 03_mpnn.ipynb                  # Passagem de mensagens condicionada às arestas
├── data/
│   └── README.md                      # Descrição e origem do conjunto de dados
├── results/
│   ├── figures/                       # Curvas de loss, curvas ROC/PR, AUC por classe
│   └── tables/                        # Métricas em CSV
├── report/
│   └── Project_Report.pdf             # Relatório técnico completo
├── references/
│   └── references_README.md           # Artigos replicados e citados
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Reproduzindo os resultados

```bash
git clone https://github.com/<seu-usuario>/molecular-odor-prediction-gnn.git
cd molecular-odor-prediction-gnn

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O conjunto de dados é baixado pelas primeiras células, então os notebooks rodam de ponta a ponta sem configuração manual. Eles também funcionam no Google Colab pelo badge no topo de cada um. Recomenda-se GPU para as execuções do GATv2 e do GCN da literatura.

> **Nota sobre a divisão dos dados.** Os notebooks vêm com a divisão sequencial simples 80/10/10 ativa, que reproduz as linhas *ingênuas* da tabela de resultados. A rotina de estratificação iterativa de segunda ordem está na seção **"Second-order iterative stratification"**, mas comentada; descomente-a (e os loaders logo abaixo) para reproduzir as linhas otimizadas. As seeds ainda não estão fixadas — espere pequena variação entre execuções.

---

## Limitações e trabalhos futuros

- **Nenhum baseline clássico próprio.** As comparações com Random Forest e XGBoost são citadas da literatura, mas não reproduzidas aqui. Rodar XGBoost sobre fingerprints de Morgan seria barato e tornaria autossuficiente o argumento de que as GNNs valem a pena.
- **Um único conjunto de dados.** Todos os resultados vêm da combinação GoodScents + Leffingwell. A generalização para amostras químicas independentes não foi testada; validar o pipeline em um novo conjunto de dados é o próximo passo imediato.
- **Lacuna de precisão.** O MPNN otimizado troca precisão por recall em relação ao modelo publicado. Métodos de calibração além da varredura de thresholds (temperature scaling, focal loss) poderiam reequilibrar isso.
- **Features de aresta não utilizadas em dois dos três modelos.** O featurizer produz features no nível das ligações que os modelos GCN e GATv2 descartam; só o MPNN as aproveita. Acrescentar convoluções condicionadas às arestas às outras arquiteturas isolaria quanto da vantagem do MPNN vem apenas dessa informação.
- **Overfitting nos modelos de atenção.** A loss de validação do GATv2 sobe após as primeiras épocas; regularização mais forte ou mais camadas de atenção merecem exploração.
- **Descritores raros.** A AUROC por classe cai abaixo de 0,5 para os rótulos mais raros. Um esquema hierárquico de rótulos, ou o agrupamento dos descritores em categorias mais amplas como em Achebouche et al. (2022), pode ajudar.
- **Integração multimodal.** Combinar em ensemble os embeddings de grafo com descritores físico-químicos ou embeddings de CNN de imagens moleculares 2D, na linha de Sharma et al. (2021), é uma direção promissora.

---

## Referências

1. Sanchez-Lengeling, B. et al. (2019). *Machine learning for scent: Learning generalizable perceptual representations of small molecules.* arXiv:1910.10685.
2. Achebouche, R. et al. (2022). *Application of artificial intelligence to decode the relationships between smell, olfactory receptors and small molecules.* Scientific Reports, 12(1), 18817.
3. Sharma, A. et al. (2021). *SMILES to smell: decoding the structure–odor relationship of chemical compounds using the deep neural network approach.* Journal of Chemical Information and Modeling, 61(2), 676–688.
4. Suh, J., Hong, Y., & Park, C. (2025). *A comparative study of machine learning models on molecular fingerprints for odor decoding.* Communications Chemistry, 8(1), 278.

Detalhes completos em [`references/references_README.md`](references/references_README.md).

---

## Autores

Felipe Perone · Vinícius Baião · Paulo Warken

Orientador: Prof. Guillaume Renton — ENSEA

📄 [Relatório técnico completo](report/Project_Report.pdf)

---

## Licença

MIT — veja [LICENSE](LICENSE).
