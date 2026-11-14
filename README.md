# Optimized selection of band gaps in organic semiconductors through integrated machine learning and validation on experimentally reported band gaps for enhanced organic solar cell performance

Data, code and trained model for the stacking regressor (Random Forest + Gradient Boosting + XGBoost, linear regression meta-learner) trained on RDKit fingerprints.

## Contents

| Path | Description |
|---|---|
| `Dataset/dataset_285.csv` | 285 molecules: SMILES, experimental band gap (eV), source DOI, train/test assignment, duplicate flag |
| `Dataset/external_validation_29.csv` | Independent test dataset (29 compounds): SMILES, experimental optical band gap, source DOI, prediction |
| `Code/features.py` | RDKit, Morgan and MACCS fingerprint generation |
| `Code/hyperparameters.json` | Final hyperparameters of the three base estimators |
| `Code/train_stacking.py` | Trains the stacking regressor from the SMILES and reports train/test metrics |
| `Code/predict.py` | Loads the deposited model and reproduces the reported metrics |
| `Model/rdkit_stacking_regressor.joblib` | Trained stacking regressor |
| `requirements.txt` | Package versions |

## Settings

| Setting | Value |
|---|---|
| Train/test split | 80:20, `train_test_split(test_size=0.2, random_state=42)` (228 / 57 molecules) |
| Fingerprints | RDKit (2048 bits), Morgan (radius 2, 1024 bits), MACCS (167 bits) |
| Hyperparameter optimization | Optuna, TPE sampler, 800 trials per base estimator and fingerprint type |
| Random Forest `random_state` | 0 |
| Gradient Boosting `random_state` | 49 |
| XGBoost `random_state` | 0 |
| Stacking | 5-fold out-of-fold predictions, `LinearRegression` meta-learner |

Search ranges are listed in Tables S15-S17 of the Supporting Information.

## Usage

```
pip install -r requirements.txt
cd Code
python predict.py
python train_stacking.py
```

`predict.py` output with the deposited model:

```
train R2 = 0.907  MAE = 0.074 eV  RMSE = 0.097 eV
test R2 = 0.817  MAE = 0.115 eV  RMSE = 0.151 eV
independent test dataset R2 = 0.69  MAE = 0.072 eV  RMSE = 0.105 eV
```

## Web server

https://huggingface.co/spaces/BasirAkbar/bandgap-predictor
