**English** · [Français](README.fr.md) · [Português](README.pt.md)

# Molecular Odor Prediction with Graph Neural Networks

**Multi-label prediction of 138 odor descriptors directly from molecular SMILES, replicating and surpassing published benchmarks.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.x-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2024.03-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Mapping a molecule's chemical structure to the odor humans perceive — the **Structure–Odor Relationship (SOR)** — is a long-standing problem in olfactory informatics, with direct applications in the flavor and fragrance industry and in sensory neuroscience.

This project builds a graph-based deep learning pipeline that takes a **SMILES string** as input and predicts a **138-dimensional binary vector** of odor descriptors (*floral*, *woody*, *fruity*, …). The task is an extreme multi-label classification problem: roughly **97% of the label matrix is zeros**, since a typical molecule carries only 3–4 descriptors.

The work has two goals:

1. **Replicate** the GNN architectures reported in the olfactory ML literature (notably Sanchez-Lengeling et al., 2019).
2. **Extend** them with a rigorous data-handling pipeline — *second-order iterative stratification* and *per-label threshold optimization* — and evaluate an additional architecture (GATv2) not covered by those papers.

The fully optimized MPNN reaches an **AUROC of 0.9001** and an **F1-Score of 0.3895**, surpassing the published benchmark of 0.362.

> Developed as the *GenAI for Molecular Discovery* project at **ENSEA** (École Nationale Supérieure de l'Électronique et de ses Applications), under the supervision of Prof. Guillaume Renton. Originally written in Google Colab and reorganized into this repository.

---

## Key Results

### Literature replication and extensions

| Model | AUROC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| GCN *(paper benchmark)* | 0.8900 | 0.3790 | 0.3870 | 0.3620 |
| GCN — naive replication | 0.8464 | 0.1008 | 0.8109 | 0.1793 |
| GCN — **+ stratification + thresholds** | 0.8561 | 0.2616 | 0.5158 | 0.3471 |
| MPNN *(paper benchmark)* | 0.8940 | 0.3790 | 0.3900 | 0.3600 |
| MPNN — naive replication | 0.8642 | 0.1216 | 0.6023 | 0.2023 |
| MPNN — **+ stratification + thresholds** | **0.9001** | 0.3090 | 0.5270 | **0.3895** |
| GATv2 *(our extension)* | 0.8310 | 0.0713 | 0.7128 | 0.1296 |
| GATv2 — + stratification + thresholds | 0.8561 | 0.2436 | 0.5042 | 0.3284 |

**Takeaways**

- A naive replication of the published architectures falls far short of the reported metrics (F1 0.18–0.20 vs. 0.36). The architecture alone does not reproduce the results — **the data-handling pipeline is what closes the gap.**
- Second-order iterative stratification plus per-label threshold optimization roughly **doubles the F1-score** of every architecture tested, without changing a single model weight.
- The optimized MPNN **exceeds the published F1 benchmark** and matches the state-of-the-art AUROC range. Our precision is slightly lower than the literature (0.309 vs. 0.379), but recall is substantially higher (0.527 vs. 0.387) — the model recovers a much larger share of true odor labels.
- A key diagnostic: validation **loss diverges** (probability calibration drifts) while **AUROC stays on a stable plateau**. Ranking ability survives even as calibration degrades, which is exactly why post-hoc threshold optimization rescues a model that looks broken at the default 0.5 cutoff.
- **Micro-averaged AUROC hides a wide spread across labels.** Per-class AUROC averages 0.773 (GCN) and 0.797 (GATv2), with the best descriptors above 0.99 and the worst below 0.15 — frequent odors are learned well, rare ones are not.

### Baseline GCN ablation — readout function and depth

| Readout | Layers | Hamming Loss | Micro-F1 | Subset Accuracy |
|---|---|---|---|---|
| Global mean pool | 2 | 0.1907 | 0.1291 | 0.0000 |
| Global mean pool | 3 | 0.1815 | 0.1436 | 0.0421 |
| Global mean pool | 4 | 0.1080 | 0.1954 | 0.0641 |
| Global max pool | 2 | 0.1616 | 0.1600 | 0.0401 |
| Global max pool | 3 | 0.1023 | 0.1915 | 0.0561 |
| Global max pool | 4 | 0.0784 | **0.2259** | 0.0641 |

Max pooling consistently beats mean pooling — plausible in olfaction, where a single reactive functional group can dominate the whole scent profile — but at the cost of markedly **less stable validation loss**. Mean pooling converges smoothly; max pooling oscillates while scoring higher.

### Scope of this repository

The project was carried out by a team of three. The notebooks published here cover the **baseline GCN ablation**, the **literature-approximated GCN**, the **GATv2** extension and the **MPNN**.

One caveat on provenance: the MPNN notebook is a **re-implementation written from the architectural specification in the report**, not the original training code, which was written by another team member. The MPNN metrics in the table above come from that original run. Re-running the notebook lands in the same range without reproducing those numbers exactly.

---

## Dataset

A combination of two expert-curated sources widely used in the flavor and fragrance industry:

- **The Good Scents Company Information System** — qualitative odor descriptions from expert sensory evaluation.
- **Leffingwell PMP 2001** — computationally standardized odor descriptors.

| Property | Value |
|---|---|
| Molecules | 4,983 |
| Odor descriptors (labels) | 138 |
| Labels per molecule (typical) | 3–4 |
| Sparsity of the label matrix | ~97% zeros |
| Distinct chemical elements | 25 |
| Input representation | SMILES → molecular graph |

The notebooks download the dataset automatically — no manual setup required. See [`data/README.md`](data/data_README.md) for details.

### The core challenge: zero-inflation

With 97% negatives, a model minimizing global loss learns to predict "no odor" for everything: high accuracy, zero practical utility. Two mechanisms address this:

- **Weighted BCE loss** — a per-class `pos_weight` derived from training-set label frequency penalizes missed rare descriptors.
- **Second-order iterative stratification** — standard random splitting destroys label co-occurrence ratios in multi-label data: isolating a molecule to satisfy one label's distribution disrupts all the labels it co-occurs with. This algorithm preserves *label-pair* ratios across train/validation/test, distributing the rarest combinations first, so ultra-rare scent profiles are not excluded from evaluation.

---

## Methodology

### 1. SMILES → graph

Each SMILES string is converted into a topological graph with PyTorch Geometric's `from_smiles` (RDKit under the hood): **atoms as nodes, bonds as edges**, hydrogens implicit.

- **Node features (9 per atom):** atomic number, chirality, degree, formal charge, number of hydrogens, radical electrons, hybridization, aromaticity, ring membership.
- **Edge features:** bond type, stereochemistry, conjugation. *(Produced by the featurizer but not consumed by the GCN and GATv2 models, which operate on connectivity alone.)*
- **Target:** binary vector of length 138.
- **Batch size:** 32.

### 2. Architectures

**Baseline GCN** — sequential `GCNConv` layers with ReLU, at depths of 2, 3 and 4 (hidden sizes 64 → 128 → 256 → 128), global pooling, single linear classification head. Used to study depth and readout, and to establish the oversmoothing-vs-expressiveness trade-off.

**Literature-approximated GCN** — designed to mirror the optimized model of Sanchez-Lengeling et al.:
- 4 GCN layers with hidden dims `[15, 20, 27, 36]`.
- **Jumping Knowledge readout:** input features and every intermediate layer are individually sum-pooled and concatenated into a ~175-dim embedding, retaining both low-level atomic detail and high-level structure.
- 2-layer MLP `[96, 63]` with Batch Normalization, ReLU, and dropout 0.47.
- Linear head → 138 labels.

**GATv2** *(our extension beyond the literature)* — two `GATv2Conv` layers: the first with 4 attention heads (hidden dim 64), concatenated outputs, ELU and dropout 0.3; the second single-head. Global mean pooling, then a linear head. GATv2 fixes the static-attention limitation of the original GAT by applying the linear transform *after* concatenating source and target features, allowing fully dynamic attention ranking — relevant for molecular graphs where some functional groups matter disproportionately.

**MPNN** — the architecture reported by Sanchez-Lengeling et al.: 5 message-passing steps, hidden dim 43, edge-conditioned matrix-multiplication messages (`NNConv`), GRU update function, gated sum readout of 197 dimensions with a residual path, then a 3-layer fully connected net (392 units, ReLU, batch norm, dropout 0.12). This is the **only model here that consumes the bond features**, and the only one whose node states carry memory across message-passing steps — both plausible reasons it outperforms the other two architectures on the same data. *(See the provenance note under Scope.)*

### 3. Training

- Loss: `BCEWithLogitsLoss` with per-class `pos_weight` computed from the training split.
- Optimizer: Adam, initial LR 0.001.
- Baselines: early stopping on validation loss, patience 50.
- Advanced architectures: Cosine Annealing with Warm Restarts (`T_0=50`), extended to 500 epochs for the final optimized GCN, whose loss was still decreasing at 300.

### 4. Evaluation

Accuracy is meaningless under zero-inflation, so evaluation uses **AUROC** (threshold-independent ranking), **AUPRC** (sensitive to imbalance), **Precision**, **Recall**, **Micro-F1**, and Hamming Loss where needed for direct comparison with published tables.

**Per-label threshold optimization:** instead of a fixed 0.5 cutoff, decision thresholds are swept **independently for each of the 138 labels** on the validation set to maximize F1, then frozen and applied to the test set. This is the single highest-impact post-processing step in the pipeline: on the literature GCN it lifts F1 from 0.2604 (best single global threshold, 0.75) to 0.3471.

The resulting thresholds are themselves interpretable — the model sets a strict bar for descriptors it detects confidently (*alcoholic*, *malty*, *bergamot* ≈ 0.93) and a permissive one for those it struggles with (*sulfurous* ≈ 0.34).

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_gcn_implementation.ipynb    # Preprocessing, baseline GCN ablation, literature GCN
│   ├── 02_gatv2.ipynb                 # Attention-based extension
│   └── 03_mpnn.ipynb                  # Edge-conditioned message passing
├── data/
│   └── README.md                      # Dataset description and source
├── results/
│   ├── figures/                       # Loss curves, ROC/PR curves, per-class AUC
│   └── tables/                        # Metrics as CSV
├── report/
│   └── Project_Report.pdf             # Full technical report
├── references/
│   └── references_README.md           # Papers replicated and cited
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Reproducing

```bash
git clone https://github.com/<your-username>/molecular-odor-prediction-gnn.git
cd molecular-odor-prediction-gnn

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is downloaded by the first cells, so the notebooks run top to bottom with no manual setup. They also run on Google Colab via the badge at the top of each one. A GPU is recommended for the GATv2 and literature-GCN runs.

> **Note on splitting.** The notebooks ship with the simple sequential 80/10/10 split active, which reproduces the *naive* rows of the results table. The second-order iterative stratification routine is included in the section **"Second-order iterative stratification"** but is commented out; uncomment it (and the loaders below it) to reproduce the optimized rows. Random seeds are not yet fixed — expect small run-to-run variation.

---

## Limitations and Future Work

- **No classical baseline of our own.** Random Forest / XGBoost comparisons are cited from the literature but not reproduced here. Running XGBoost on Morgan fingerprints would be cheap and would make the "GNNs are worth it" argument self-contained.
- **Single dataset.** All results come from the GoodScents + Leffingwell combination. Generalization to independent chemical samples is untested; validating the pipeline on a new dataset is the immediate next step.
- **Precision gap.** The optimized MPNN trades precision for recall relative to the published model. Calibration methods beyond threshold sweeping (e.g. temperature scaling, focal loss) could rebalance this.
- **Edge features unused in two of three models.** The featurizer produces bond-level features that the GCN and GATv2 models discard; only the MPNN consumes them. Adding edge-conditioned convolutions to the other architectures would isolate how much of the MPNN's advantage comes from the bond information alone.
- **Overfitting in attention models.** GATv2 validation loss rises after the initial epochs; stronger regularization or more attention layers deserve exploration.
- **Rare descriptors.** Per-class AUROC falls below 0.5 for the rarest labels. A hierarchical label scheme, or grouping descriptors into broader categories as in Achebouche et al. (2022), might help.
- **Multimodal integration.** Ensembling graph embeddings with physicochemical descriptors or CNN embeddings of 2D molecular images, following Sharma et al. (2021), is a promising direction.

---

## References

1. Sanchez-Lengeling, B. et al. (2019). *Machine learning for scent: Learning generalizable perceptual representations of small molecules.* arXiv:1910.10685.
2. Achebouche, R. et al. (2022). *Application of artificial intelligence to decode the relationships between smell, olfactory receptors and small molecules.* Scientific Reports, 12(1), 18817.
3. Sharma, A. et al. (2021). *SMILES to smell: decoding the structure–odor relationship of chemical compounds using the deep neural network approach.* Journal of Chemical Information and Modeling, 61(2), 676–688.
4. Suh, J., Hong, Y., & Park, C. (2025). *A comparative study of machine learning models on molecular fingerprints for odor decoding.* Communications Chemistry, 8(1), 278.

Full details in [`references/references_README.md`](references/references_README.md).

---

## Authors

Felipe Perone · Vinícius Baião · Paulo Warken

Supervisor: Prof. Guillaume Renton — ENSEA

📄 [Full technical report](report/Project_Report.pdf)

---

## License

MIT — see [LICENSE](LICENSE).
