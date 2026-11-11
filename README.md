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

> Developed as the *GenAI for Molecular Discovery* project at **ENSEA** (École Nationale Supérieure de l'Électronique et de ses Applications), under the supervision of Prof. Guillaume Renton.

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

### Baseline GCN ablation — readout function and depth

| Readout | Layers | Hamming Loss | Micro-F1 | Subset Accuracy |
|---|---|---|---|---|
| Global mean pool | 2 | 0.1907 | 0.1291 | 0.0000 |
| Global mean pool | 3 | 0.1815 | 0.1436 | 0.0410 |
| Global mean pool | 4 | 0.1080 | 0.1954 | 0.0641 |
| Global max pool | 2 | 0.1616 | 0.1600 | 0.0401 |
| Global max pool | 3 | 0.1023 | 0.1915 | 0.0561 |
| Global max pool | 4 | 0.0784 | **0.2259** | 0.0641 |

Max pooling consistently beats mean pooling — plausible in olfaction, where a single reactive functional group can dominate the whole scent profile — but at the cost of markedly **less stable validation loss**. Mean pooling converges smoothly; max pooling oscillates while scoring higher.

---

## Dataset

A combination of two expert-curated sources widely used in the flavor and fragrance industry:

- **The Good Scents Company Information System** — qualitative odor descriptions from expert sensory evaluation.
- **Leffingwell PMP 2001** — computationally standardized odor descriptors.

| Property | Value |
|---|---|
| Molecules | ~5,000 |
| Odor descriptors (labels) | 138 |
| Labels per molecule (typical) | 3–4 |
| Sparsity of the label matrix | ~97% zeros |
| Input representation | SMILES → molecular graph |

See [`data/README.md`](data/README.md) for how to obtain the data.

### The core challenge: zero-inflation

With 97% negatives, a model minimizing global loss learns to predict "no odor" for everything: high accuracy, zero practical utility. Two mechanisms address this:

- **Weighted BCE loss** — a per-class `pos_weight` derived from training-set label frequency penalizes missed rare descriptors.
- **Second-order iterative stratification** — standard random splitting destroys label co-occurrence ratios in multi-label data: isolating a molecule to satisfy one label's distribution disrupts all the labels it co-occurs with. This algorithm preserves *label-pair* ratios across train/validation/test, distributing the rarest combinations first, so ultra-rare scent profiles are not excluded from evaluation.

---

## Methodology

### 1. SMILES → graph

Each SMILES string is converted with RDKit into a topological graph: **atoms as nodes, bonds as edges**.

- **Node features:** atom type, hybridization state, partial charge, and other physicochemical descriptors.
- **Edge features:** bond type.
- **Target:** binary vector of length 138.
- **Batch size:** 32.

### 2. Architectures

**Baseline GCN** — sequential `GCNConv` layers with ReLU, at depths of 2, 3 and 4 (hidden sizes 64 → 128 → 256 → 128), global pooling, single linear classification head. Used to study depth and readout, and to establish the oversmoothing-vs-expressiveness trade-off.

**Literature-approximated GCN** — designed to mirror the optimized model of Sanchez-Lengeling et al.:
- 4 GCN layers with hidden dims `[15, 20, 27, 36]` and **SELU** activations (self-normalizing message passing).
- **Jumping Knowledge readout:** input features and every intermediate layer are individually sum-pooled and concatenated into a ~175-dim embedding, retaining both low-level atomic detail and high-level structure.
- 2-layer MLP `[96, 63]` with Batch Normalization, ReLU, and dropout 0.47.
- Linear head → 138 labels.

**MPNN** — the architecture reported by Sanchez-Lengeling et al.: 5 message-passing layers, hidden dim 43, edge-conditioned matrix-multiplication messages, GRU update function, global sum pooling with a 197-dim softmax readout and residual connections, then a 3-layer fully connected net (392 units, ReLU, batch norm, dropout 0.12).

**GATv2** *(our extension beyond the literature)* — two `GATv2Conv` layers; the first with multi-head attention, concatenated outputs, ELU and dropout; the second single-head. Global mean pooling, then a linear head. GATv2 fixes the static-attention limitation of the original GAT by applying the linear transform *after* concatenating source and target features, allowing fully dynamic attention ranking — relevant for molecular graphs where some functional groups matter disproportionately.

### 3. Training

- Loss: `BCEWithLogitsLoss` with per-class `pos_weight`.
- Optimizer: Adam, initial LR 0.001.
- Baselines: early stopping, patience 50.
- Advanced architectures: Cosine Annealing with Warm Restarts, 300 epochs (extended to 500 for the final optimized GCN, whose loss was still decreasing).

### 4. Evaluation

Accuracy is meaningless under zero-inflation, so evaluation uses **AUROC** (threshold-independent ranking), **AUPRC** (sensitive to imbalance), **Precision**, **Recall**, **Micro-F1**, and Hamming Loss where needed for direct comparison with published tables.

**Per-label threshold optimization:** instead of a fixed 0.5 cutoff, decision thresholds are swept **independently for each of the 138 labels** on the validation set to maximize F1, then frozen and applied to the test set. This is the single highest-impact post-processing step in the pipeline.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_eda.ipynb                   # Dataset exploration, label distribution, imbalance
│   ├── 02_baseline_gcn.ipynb          # Depth and readout ablation
│   ├── 03_literature_replication.ipynb# Literature GCN + MPNN, naive vs. optimized
│   └── 04_gatv2.ipynb                 # Attention-based extension
├── src/
│   ├── featurization.py               # SMILES → PyG graph objects
│   ├── splitting.py                   # Second-order iterative stratification
│   ├── models.py                      # GCN, MPNN, GATv2 definitions
│   ├── train.py                       # Training loop, weighted BCE, schedulers
│   └── evaluate.py                    # Metrics and per-label threshold optimization
├── data/
│   └── README.md                      # Data source and download instructions
├── results/
│   ├── figures/                       # Loss curves, metric evolution
│   └── tables/                        # Metrics as CSV
├── report/
│   └── report.pdf                     # Full technical report
├── references/
│   └── README.md                      # Papers replicated and cited
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

Then follow the instructions in [`data/README.md`](data/README.md) to place the dataset under `data/`, and run the notebooks in order (`01` → `04`). Each notebook also runs on Google Colab via the badge at the top.

Random seeds are fixed throughout; GPU is recommended for the MPNN and GATv2 runs.

---

## Limitations and Future Work

- **Single dataset.** All results come from the GoodScents + Leffingwell combination. Generalization to independent chemical samples is untested; validating the pipeline on a new dataset is the immediate next step.
- **Precision gap.** The optimized MPNN trades precision for recall relative to the published model. Calibration methods beyond threshold sweeping (e.g. temperature scaling, focal loss) could rebalance this.
- **Overfitting in attention models.** GATv2 validation loss rises after the initial epochs even with stratification; stronger regularization or more attention layers deserve exploration.
- **Multimodal integration.** Ensembling graph embeddings with physicochemical descriptors or CNN embeddings of 2D molecular images, following Sharma et al. (2021), is a promising direction.
- **Label granularity.** Working with 138 fine-grained descriptors is harder than the grouped-category setting used in some of the literature; a hierarchical label scheme might improve rare-descriptor performance.

---

## References

1. Sanchez-Lengeling, B. et al. (2019). *Machine learning for scent: Learning generalizable perceptual representations of small molecules.* arXiv:1910.10685.
2. Achebouche, R. et al. (2022). *Application of artificial intelligence to decode the relationships between smell, olfactory receptors and small molecules.* Scientific Reports, 12(1), 18817.
3. Sharma, A. et al. (2021). *SMILES to smell: decoding the structure–odor relationship of chemical compounds using the deep neural network approach.* Journal of Chemical Information and Modeling, 61(2), 676–688.
4. Suh, J., Hong, Y., & Park, C. (2025). *A comparative study of machine learning models on molecular fingerprints for odor decoding.* Communications Chemistry, 8(1), 278.

Full details in [`references/README.md`](references/README.md).

---

## Authors

Felipe Perone · Vinícius Baião · Paulo Warken

Supervisor: Prof. Guillaume Renton — ENSEA

📄 [Full technical report](report/report.pdf)

---

## License

MIT — see [LICENSE](LICENSE).
