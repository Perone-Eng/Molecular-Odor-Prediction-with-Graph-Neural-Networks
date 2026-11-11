# Dataset

> Save this file as `data/README.md` in the repository.

## Source

The dataset combines two expert-curated olfactory databases:

| Source | Description |
|---|---|
| **The Good Scents Company Information System** | Qualitative odor descriptions used in the flavor and fragrance industry, curated from expert sensory evaluation and professional literature. |
| **Leffingwell PMP 2001** | Computationally standardized odor descriptors categorized by industry experts. |

The combined, standardized version follows the construction described by Sanchez-Lengeling et al. (2019), with rare labels filtered out. It is widely circulated as the *Multi-Labelled SMILES Odors Dataset*.

## How the notebooks obtain it

No manual download is needed. The first cells read the CSV directly:

```python
import pandas as pd

df = pd.read_csv(
    'https://raw.githubusercontent.com/fperone/'
    'Projet_IA-Generative-pour-la-decouverte-moleculaire/'
    'refs/heads/main/Multi-Labelled_Smiles_Odors_dataset.csv'
)
df.fillna(0, inplace=True)
```

To work offline, download that file once and place it at `data/Multi-Labelled_Smiles_Odors_dataset.csv`, then point the `read_csv` call at the local path.

## Characteristics

| Property | Value |
|---|---|
| Molecules (rows) | 4,983 |
| Columns | 140 |
| Odor descriptors (labels) | 138 |
| Labels per molecule (typical) | 3–4 |
| Share of zeros in the label matrix | ~97% |
| Distinct chemical elements | 25 (C, O, N, S, Cl, Br, I, Si, P, and metals such as Na, K, Ca, Fe, Zn, …) |
| Task | Multi-label binary classification |

## Format

| Column | Description |
|---|---|
| `nonStereoSMILES` | SMILES representation of the molecule, stereochemistry stripped |
| `descriptors` | Semicolon-separated odor descriptors, e.g. `rose;floral;fatty;sweet` |
| `alcoholic` … `woody` | 138 binary columns, 1 if the descriptor applies |

The two metadata columns come first, so the label matrix is `df.iloc[:, 2:]`. Labels are read as the target vector and the SMILES column feeds the graph featurizer:

```python
from torch_geometric.utils import from_smiles

data = from_smiles(df['nonStereoSMILES'][i], with_hydrogen=0)
data.x = data.x.float()
data.y = torch.tensor(df.iloc[i, 2:], dtype=torch.float)
```

A handful of entries contain salts and counter-ions (e.g. `...[K+].[K+]`), which `from_smiles` represents as disconnected components in the same graph.

## Known characteristics to be aware of

- **Zero-inflation.** ~97% of the label matrix is zeros. Any model trained with an unweighted loss collapses to predicting "no odor" everywhere. See the weighted-BCE and stratification sections in the main README.
- **Highly uneven label frequency.** Common descriptors (*fruity*, *sweet*, *green*) appear orders of magnitude more often than rare ones, and the rarest may be absent from a given test split — one more reason the project uses second-order iterative stratification rather than a random split.
- **No missing values after `fillna(0)`.** Empty label cells mean "descriptor not assigned", which is treated as a negative.

## Licensing note

The underlying GoodScents and Leffingwell data are distributed by their original owners under their own terms:

- **The Good Scents Company:** http://www.thegoodscentscompany.com/
- **Leffingwell & Associates (PMP 2001):** https://www.leffingwell.com/

Check those terms before redistributing any derived dataset.
