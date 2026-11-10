# Optical band-gap prediction for organic semiconductors

ExtraTrees model on 2048-bit RDKit fingerprints. The training set has 455 experimental optical gaps: 282 molecules from this project and 173 non-fullerene-acceptor optical gaps from Hao, Zenodo, doi:10.5281/zenodo.10791580.

The number to report is the cross-validated error, not the fit on all 455 molecules.

| Evaluation | R² | MAE |
|---|---|---|
| 10-fold cross-validation | 0.70 ± 0.10 | 0.12 ± 0.02 eV |
| Murcko scaffold holdout | 0.66 ± 0.10 | 0.13 ± 0.02 eV |

Settings were chosen inside each training fold. The saved model was then refit on all 455 molecules for prediction. Its settings are 565 trees, unlimited depth, `sqrt` features, and a minimum leaf size of 2.

## Repository contents

- `Dataset/molecules.csv` — canonical SMILES, optical gap (eV), source, and DOI.
- `Dataset/nfa_optical_gaps_added.csv` — the 173 added Zenodo molecules.
- `Model/extratrees_rdkit_455.joblib` — the saved model.
- `Results/predictions.csv` — out-of-fold predictions for the 10-fold and scaffold tests.
- `Results/model_report.json` — the summary numbers.
- `Results/shap_top20.csv` — the 20 RDKit bits with the largest mean absolute SHAP value.
- `Results/similarity_neighbors.csv` — ten nearest neighbors of the lowest-gap and highest-gap molecules.
- `Figures/` — held-out scatter plots, SHAP plots, and the similarity grids.
- `predict.py` — predict one SMILES string.

## Predict

```bash
pip install -r requirements.txt
python predict.py "CSc1ccc(-c2cccs2)cc1"
```

With no argument, the script predicts the example molecule stored in `predict.py`. If the nearest training molecule has a Tanimoto similarity below 0.80, the script prints a warning.

## Web server

https://huggingface.co/spaces/AsadWazir/bandgap

## Developer contact

- Basir Akbar (basirakbar98@gmail.com)
- Asad Khan (asadkhan@jbnu.ac.kr)
