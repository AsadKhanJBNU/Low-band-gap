import inspect, joblib
import numpy as np, pandas as pd
import xgboost as xgb
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from features import rdkit_fp

model = joblib.load("../Model/rdkit_stacking_regressor.joblib")
for est in model.estimators_:
    if isinstance(est, xgb.XGBModel):
        for k in inspect.signature(xgb.XGBModel.__init__).parameters:
            if k not in ("self", "kwargs") and not hasattr(est, k):
                setattr(est, k, None)

d = pd.read_csv("../Dataset/dataset_285.csv")
X = np.array([rdkit_fp(Chem.MolFromSmiles(s)) for s in d.smiles]).astype(float)
y = d.band_gap_eV.values
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
for name, idx in (("train", tr), ("test", te)):
    p = model.predict(X[idx])
    print(name, "R2 = %.3f  MAE = %.3f eV  RMSE = %.3f eV" % (
        r2_score(y[idx], p), mean_absolute_error(y[idx], p), mean_squared_error(y[idx], p) ** 0.5))

e = pd.read_csv("../Dataset/external_validation_29.csv")
Xe = np.array([rdkit_fp(Chem.MolFromSmiles(s)) for s in e.smiles_model]).astype(float)
pe = model.predict(Xe)
print("independent test dataset R2 = %.2f  MAE = %.3f eV  RMSE = %.3f eV" % (
    r2_score(e.band_gap_eV, pe), mean_absolute_error(e.band_gap_eV, pe), mean_squared_error(e.band_gap_eV, pe) ** 0.5))
