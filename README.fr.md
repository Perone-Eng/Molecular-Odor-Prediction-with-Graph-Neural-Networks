[English](README.md) · **Français** · [Português](README.pt.md)

# Prédiction des odeurs moléculaires par réseaux de neurones sur graphes

**Prédiction multi-label de 138 descripteurs olfactifs directement à partir des SMILES moléculaires, en répliquant et dépassant les références publiées.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.x-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2024.03-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Présentation

Relier la structure chimique d'une molécule à l'odeur perçue par l'être humain — la **relation structure-odeur (SOR)** — est un problème de longue date en informatique olfactive, avec des applications directes dans l'industrie des arômes et parfums ainsi qu'en neurosciences sensorielles.

Ce projet construit un pipeline d'apprentissage profond sur graphes qui prend en entrée une **chaîne SMILES** et prédit un **vecteur binaire de dimension 138** de descripteurs olfactifs (*floral*, *boisé*, *fruité*, …). La tâche relève de la classification multi-label extrême : environ **97 % de la matrice de labels est composée de zéros**, une molécule type ne portant que 3 à 4 descripteurs.

Le travail poursuit deux objectifs :

1. **Répliquer** les architectures GNN décrites dans la littérature en olfaction computationnelle (notamment Sanchez-Lengeling et al., 2019).
2. **Les étendre** par un pipeline rigoureux de traitement des données — *stratification itérative de second ordre* et *optimisation des seuils par label* — et évaluer une architecture supplémentaire (GATv2) absente de ces travaux.

Le MPNN entièrement optimisé atteint une **AUROC de 0,9001** et un **F1-Score de 0,3895**, dépassant la référence publiée de 0,362.

> Réalisé dans le cadre du projet *GenAI for Molecular Discovery* à l'**ENSEA** (École Nationale Supérieure de l'Électronique et de ses Applications), sous la supervision du Prof. Guillaume Renton. Développé initialement sur Google Colab, puis réorganisé dans ce dépôt.

---

## Principaux résultats

### Réplication de la littérature et extensions

| Modèle | AUROC | Précision | Rappel | F1-Score |
|---|---|---|---|---|
| GCN *(référence article)* | 0,8900 | 0,3790 | 0,3870 | 0,3620 |
| GCN — réplication naïve | 0,8464 | 0,1008 | 0,8109 | 0,1793 |
| GCN — **+ stratification + seuils** | 0,8561 | 0,2616 | 0,5158 | 0,3471 |
| MPNN *(référence article)* | 0,8940 | 0,3790 | 0,3900 | 0,3600 |
| MPNN — réplication naïve | 0,8642 | 0,1216 | 0,6023 | 0,2023 |
| MPNN — **+ stratification + seuils** | **0,9001** | 0,3090 | 0,5270 | **0,3895** |
| GATv2 *(notre extension)* | 0,8310 | 0,0713 | 0,7128 | 0,1296 |
| GATv2 — + stratification + seuils | 0,8561 | 0,2436 | 0,5042 | 0,3284 |

**Enseignements**

- Une réplication naïve des architectures publiées reste très en deçà des métriques rapportées (F1 de 0,18 à 0,20 contre 0,36). L'architecture seule ne reproduit pas les résultats — **c'est le pipeline de traitement des données qui comble l'écart.**
- La stratification itérative de second ordre combinée à l'optimisation des seuils par label **double approximativement le F1-score** de chaque architecture testée, sans modifier un seul poids du modèle.
- Le MPNN optimisé **dépasse la référence publiée en F1** et se situe dans la plage AUROC de l'état de l'art. Notre précision est légèrement inférieure à la littérature (0,309 contre 0,379), mais le rappel est nettement supérieur (0,527 contre 0,387) : le modèle retrouve une part bien plus grande des labels olfactifs réels.
- Un diagnostic central : la **perte de validation diverge** (la calibration des probabilités dérive) tandis que l'**AUROC se maintient sur un plateau stable**. La capacité de classement survit à la dégradation de la calibration, ce qui explique précisément pourquoi l'optimisation a posteriori des seuils rattrape un modèle qui paraît défaillant au seuil par défaut de 0,5.
- **L'AUROC micro-moyennée masque une forte dispersion entre labels.** L'AUROC par classe vaut en moyenne 0,773 (GCN) et 0,797 (GATv2), avec les meilleurs descripteurs au-dessus de 0,99 et les pires en dessous de 0,15 : les odeurs fréquentes sont bien apprises, les rares ne le sont pas.

### Ablation du GCN de référence — fonction de readout et profondeur

| Readout | Couches | Hamming Loss | Micro-F1 | Exactitude par sous-ensemble |
|---|---|---|---|---|
| Global mean pool | 2 | 0,1907 | 0,1291 | 0,0000 |
| Global mean pool | 3 | 0,1815 | 0,1436 | 0,0421 |
| Global mean pool | 4 | 0,1080 | 0,1954 | 0,0641 |
| Global max pool | 2 | 0,1616 | 0,1600 | 0,0401 |
| Global max pool | 3 | 0,1023 | 0,1915 | 0,0561 |
| Global max pool | 4 | 0,0784 | **0,2259** | 0,0641 |

Le max pooling surpasse systématiquement le mean pooling — ce qui est plausible en olfaction, où un seul groupe fonctionnel réactif peut dominer tout le profil odorant — mais au prix d'une **perte de validation nettement moins stable**. Le mean pooling converge de façon régulière ; le max pooling oscille tout en obtenant de meilleurs scores.

### Périmètre de ce dépôt

Le projet a été mené par une équipe de trois personnes. Les notebooks publiés ici couvrent l'**ablation du GCN de référence**, le **GCN approximant la littérature**, l'extension **GATv2** et le **MPNN**.

Une réserve sur la provenance : le notebook MPNN est une **réimplémentation écrite à partir de la spécification architecturale du rapport**, et non le code d'entraînement d'origine, rédigé par un autre membre de l'équipe. Les métriques MPNN du tableau ci-dessus proviennent de cette exécution d'origine. Réexécuter le notebook donne des valeurs du même ordre, sans les reproduire exactement.

---

## Jeu de données

Combinaison de deux sources expertes largement utilisées dans l'industrie des arômes et parfums :

- **The Good Scents Company Information System** — descriptions olfactives qualitatives issues d'évaluations sensorielles expertes.
- **Leffingwell PMP 2001** — descripteurs olfactifs standardisés par voie computationnelle.

| Propriété | Valeur |
|---|---|
| Molécules | 4 983 |
| Descripteurs olfactifs (labels) | 138 |
| Labels par molécule (typique) | 3–4 |
| Proportion de zéros dans la matrice de labels | ~97 % |
| Éléments chimiques distincts | 25 |
| Représentation d'entrée | SMILES → graphe moléculaire |

Les notebooks téléchargent le jeu de données automatiquement — aucune configuration manuelle n'est requise. Voir [`data/README.md`](data/data_README.md) pour le détail.

### Le défi central : l'inflation de zéros

Avec 97 % de négatifs, un modèle qui minimise la perte globale apprend à prédire « aucune odeur » partout : exactitude élevée, utilité pratique nulle. Deux mécanismes y répondent :

- **Perte BCE pondérée** — un `pos_weight` par classe, dérivé de la fréquence des labels dans l'ensemble d'entraînement, pénalise les descripteurs rares manqués.
- **Stratification itérative de second ordre** — un découpage aléatoire classique détruit les ratios de co-occurrence des labels dans les données multi-label : isoler une molécule pour satisfaire la distribution d'un label perturbe celle de tous les labels qui l'accompagnent. Cet algorithme préserve les ratios de *paires de labels* entre entraînement, validation et test, en distribuant d'abord les combinaisons les plus rares, de sorte que les profils olfactifs ultra-rares ne soient pas exclus de l'évaluation.

---

## Méthodologie

### 1. SMILES → graphe

Chaque chaîne SMILES est convertie en graphe topologique via `from_smiles` de PyTorch Geometric (qui s'appuie sur RDKit) : **atomes comme nœuds, liaisons comme arêtes**, hydrogènes implicites.

- **Caractéristiques des nœuds (9 par atome) :** numéro atomique, chiralité, degré, charge formelle, nombre d'hydrogènes, électrons radicalaires, hybridation, aromaticité, appartenance à un cycle.
- **Caractéristiques des arêtes :** type de liaison, stéréochimie, conjugaison. *(Produites par le featurizer mais non exploitées par les modèles GCN et GATv2, qui n'utilisent que la connectivité.)*
- **Cible :** vecteur binaire de longueur 138.
- **Taille de batch :** 32.

### 2. Architectures

**GCN de référence** — couches `GCNConv` séquentielles avec ReLU, aux profondeurs 2, 3 et 4 (dimensions cachées 64 → 128 → 256 → 128), pooling global, une unique couche linéaire de classification. Utilisé pour étudier la profondeur et le readout, et pour établir le compromis entre sur-lissage et expressivité.

**GCN approximant la littérature** — conçu pour reproduire le modèle optimisé de Sanchez-Lengeling et al. :
- 4 couches GCN de dimensions cachées `[15, 20, 27, 36]`.
- **Readout Jumping Knowledge :** les caractéristiques d'entrée et chaque couche intermédiaire sont sum-poolées individuellement puis concaténées en un plongement d'environ 175 dimensions, conservant à la fois le détail atomique de bas niveau et la structure de haut niveau.
- MLP à 2 couches `[96, 63]` avec Batch Normalization, ReLU et dropout de 0,47.
- Couche linéaire finale → 138 labels.

**GATv2** *(notre extension au-delà de la littérature)* — deux couches `GATv2Conv` : la première avec 4 têtes d'attention (dimension cachée 64), sorties concaténées, ELU et dropout de 0,3 ; la seconde à tête unique. Pooling moyen global, puis couche linéaire finale. GATv2 corrige la limitation d'attention statique du GAT original en appliquant la transformation linéaire *après* concaténation des caractéristiques des nœuds source et cible, permettant un classement d'attention pleinement dynamique — pertinent pour des graphes moléculaires où certains groupes fonctionnels pèsent de façon disproportionnée.

**MPNN** — l'architecture décrite par Sanchez-Lengeling et al. : 5 étapes de passage de messages, dimension cachée 43, messages par multiplication matricielle conditionnée aux arêtes (`NNConv`), fonction de mise à jour GRU, readout par somme pondérée de 197 dimensions avec chemin résiduel, puis un réseau entièrement connecté à 3 couches (392 unités, ReLU, batch norm, dropout de 0,12). C'est le **seul modèle ici qui exploite les caractéristiques des liaisons**, et le seul dont les états de nœuds conservent une mémoire entre les étapes de passage de messages — deux raisons plausibles de sa supériorité sur les deux autres architectures, à données égales. *(Voir la note de provenance sous Périmètre.)*

### 3. Entraînement

- Perte : `BCEWithLogitsLoss` avec `pos_weight` par classe calculé sur l'ensemble d'entraînement.
- Optimiseur : Adam, taux d'apprentissage initial de 0,001.
- Modèles de référence : arrêt anticipé sur la perte de validation, patience de 50.
- Architectures avancées : Cosine Annealing avec Warm Restarts (`T_0=50`), étendu à 500 époques pour le GCN optimisé final, dont la perte décroissait encore à 300.

### 4. Évaluation

L'exactitude n'a pas de sens sous inflation de zéros ; l'évaluation repose donc sur l'**AUROC** (classement indépendant du seuil), l'**AUPRC** (sensible au déséquilibre), la **précision**, le **rappel**, le **Micro-F1**, et la Hamming Loss lorsqu'une comparaison directe avec les tableaux publiés l'exige.

**Optimisation des seuils par label :** au lieu d'un seuil fixe à 0,5, les seuils de décision sont balayés **indépendamment pour chacun des 138 labels** sur l'ensemble de validation afin de maximiser le F1, puis figés et appliqués à l'ensemble de test. C'est l'étape de post-traitement la plus déterminante du pipeline : sur le GCN de la littérature, elle fait passer le F1 de 0,2604 (meilleur seuil global unique, 0,75) à 0,3471.

Les seuils obtenus sont eux-mêmes interprétables : le modèle fixe une barre stricte pour les descripteurs qu'il détecte avec confiance (*alcoholic*, *malty*, *bergamot* ≈ 0,93) et une barre permissive pour ceux qui lui résistent (*sulfurous* ≈ 0,34).

---

## Structure du dépôt

```
.
├── notebooks/
│   ├── 01_gcn_implementation.ipynb    # Prétraitement, ablation GCN, GCN de la littérature
│   ├── 02_gatv2.ipynb                 # Extension fondée sur l'attention
│   └── 03_mpnn.ipynb                  # Passage de messages conditionné aux arêtes
├── data/
│   └── README.md                      # Description et source du jeu de données
├── results/
│   ├── figures/                       # Courbes de perte, courbes ROC/PR, AUC par classe
│   └── tables/                        # Métriques au format CSV
├── report/
│   └── Project_Report.pdf             # Rapport technique complet
├── references/
│   └── references_README.md           # Articles répliqués et cités
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Reproduire les résultats

```bash
git clone https://github.com/<votre-identifiant>/molecular-odor-prediction-gnn.git
cd molecular-odor-prediction-gnn

python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Le jeu de données est téléchargé par les premières cellules : les notebooks s'exécutent de bout en bout sans configuration manuelle. Ils fonctionnent également sur Google Colab via le badge en tête de chacun. Un GPU est recommandé pour les exécutions GATv2 et GCN-littérature.

> **Note sur le découpage.** Les notebooks sont livrés avec le découpage séquentiel simple 80/10/10 activé, qui reproduit les lignes *naïves* du tableau de résultats. La routine de stratification itérative de second ordre figure dans la section **« Second-order iterative stratification »** mais est commentée ; décommentez-la (ainsi que les loaders qui la suivent) pour reproduire les lignes optimisées. Les graines aléatoires ne sont pas encore fixées : de légères variations d'une exécution à l'autre sont attendues.

---

## Limites et perspectives

- **Aucune référence classique produite par nous.** Les comparaisons Random Forest / XGBoost sont citées depuis la littérature mais non reproduites ici. Exécuter XGBoost sur des empreintes de Morgan serait peu coûteux et rendrait l'argument « les GNN en valent la peine » autoportant.
- **Un seul jeu de données.** Tous les résultats proviennent de la combinaison GoodScents + Leffingwell. La généralisation à des échantillons chimiques indépendants n'est pas testée ; valider le pipeline sur un nouveau jeu de données est l'étape immédiate.
- **Écart de précision.** Le MPNN optimisé échange de la précision contre du rappel par rapport au modèle publié. Des méthodes de calibration au-delà du balayage de seuils (temperature scaling, focal loss) pourraient rééquilibrer ce compromis.
- **Caractéristiques des arêtes inutilisées dans deux modèles sur trois.** Le featurizer produit des caractéristiques au niveau des liaisons que les modèles GCN et GATv2 ignorent ; seul le MPNN les exploite. Ajouter des convolutions conditionnées aux arêtes aux autres architectures permettrait d'isoler la part de l'avantage du MPNN qui provient de cette seule information.
- **Surapprentissage des modèles d'attention.** La perte de validation du GATv2 remonte après les premières époques ; une régularisation plus forte ou davantage de couches d'attention méritent d'être explorées.
- **Descripteurs rares.** L'AUROC par classe descend sous 0,5 pour les labels les plus rares. Un schéma de labels hiérarchique, ou un regroupement des descripteurs en catégories plus larges comme chez Achebouche et al. (2022), pourrait aider.
- **Intégration multimodale.** Combiner en ensemble les plongements de graphes avec des descripteurs physico-chimiques ou des plongements CNN d'images moléculaires 2D, dans la lignée de Sharma et al. (2021), constitue une piste prometteuse.

---

## Références

1. Sanchez-Lengeling, B. et al. (2019). *Machine learning for scent: Learning generalizable perceptual representations of small molecules.* arXiv:1910.10685.
2. Achebouche, R. et al. (2022). *Application of artificial intelligence to decode the relationships between smell, olfactory receptors and small molecules.* Scientific Reports, 12(1), 18817.
3. Sharma, A. et al. (2021). *SMILES to smell: decoding the structure–odor relationship of chemical compounds using the deep neural network approach.* Journal of Chemical Information and Modeling, 61(2), 676–688.
4. Suh, J., Hong, Y., & Park, C. (2025). *A comparative study of machine learning models on molecular fingerprints for odor decoding.* Communications Chemistry, 8(1), 278.

Détail complet dans [`references/references_README.md`](references/references_README.md).

---

## Auteurs

Felipe Perone · Vinícius Baião · Paulo Warken

Encadrant : Prof. Guillaume Renton — ENSEA

📄 [Rapport technique complet](report/Project_Report.pdf)

---

## Licence

MIT — voir [LICENSE](LICENSE).
