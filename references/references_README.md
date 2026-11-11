# References

> Save this file as `references/README.md` in the repository.
> The PDFs themselves are not included here — they are copyrighted by their publishers. Links and DOIs are provided instead.

## Papers replicated

### [1] Machine learning for scent: Learning generalizable perceptual representations of small molecules
Sanchez-Lengeling, B., Wei, J. N., Lee, B. K., Gerkin, R. C., Aspuru-Guzik, A., & Wiltschko, A. B. (2019).
*arXiv preprint* arXiv:1910.10685.
🔗 https://arxiv.org/abs/1910.10685

**Role in this project:** primary reference. Source of the GoodScents + Leffingwell dataset construction (~5,030 molecules, 138 descriptors), of the optimized GCN and MPNN architectures and hyperparameters we replicate, and of the AUROC ≈ 0.894 / F1 ≈ 0.362 benchmarks our Table 2 compares against. Baselines in the paper include Random Forests and k-NN on fingerprint representations; the GNN embeddings transfer competitively to the DREAM Olfaction Challenge.

### [2] Application of artificial intelligence to decode the relationships between smell, olfactory receptors and small molecules
Achebouche, R., Tromelin, A., Audouze, K., & Taboureau, O. (2022).
*Scientific Reports*, 12(1), 18817.
🔗 https://doi.org/10.1038/s41598-022-23176-y

**Role in this project:** methodological reference for GCN-based odor prediction extended to olfactory receptor interactions. Reports AUROC ≈ 0.92 and PRC-AUC ≈ 0.67 on 23 grouped odor categories, and AUROC ≈ 0.99 for compound–receptor prediction. Not directly comparable to our numbers because of the coarser label granularity — a useful contrast on how label grouping inflates metrics.

### [3] SMILES to smell: decoding the structure–odor relationship of chemical compounds using the deep neural network approach
Sharma, A., Saha, B. K., Kumar, R., & Varadwaj, P. K. (2021).
*Journal of Chemical Information and Modeling*, 61(2), 676–688.
🔗 https://doi.org/10.1021/acs.jcim.0c01288

**Role in this project:** multimodal reference. Ensembles a DNN on physicochemical properties and molecular fingerprints with an Xception CNN on 2D molecular images, over 5,185 compounds and 104 percepts. Reports very high accuracy but lower AUROC than graph-based approaches — motivation for the multimodal direction in our Future Work.

### [4] A comparative study of machine learning models on molecular fingerprints for odor decoding
Suh, J., Hong, Y., & Park, C. (2025).
*Communications Chemistry*, 8(1), 278.
🔗 https://doi.org/10.1038/s42004-025-01678-w

**Role in this project:** classical-ML counterpoint. Benchmarks Random Forest, XGBoost and LightGBM on functional-group, physicochemical and Morgan fingerprint representations over 8,681 molecules and 200+ descriptors, with XGBoost on Morgan fingerprints reaching AUROC ≈ 0.828. Supports the argument that well-designed feature representations remain competitive, with advantages in interpretability and efficiency.

## Related resources

- **DREAM Olfaction Prediction Challenge** — external benchmark referenced by [1] and [2] for evaluating transferability of molecular embeddings.
- **RDKit** — cheminformatics toolkit used for SMILES parsing and feature extraction. https://www.rdkit.org/
- **PyTorch Geometric** — graph deep learning framework. https://pytorch-geometric.readthedocs.io/
- **Iterative stratification for multi-label data** — Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011), *On the stratification of multi-label data*, ECML PKDD. The second-order variant is the splitting strategy used throughout this project.
