import json
import numpy as np, pandas as pd
import xgboost as xgb
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from features import rdkit_fp

H = json.load(open("hyperparameters.json"))
d = pd.read_csv("../Dataset/dataset_285.csv")
X = np.array([rdkit_fp(Chem.MolFromSmiles(s)) for s in d.smiles]).astype(float)
y = d.band_gap_eV.values
tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

model = StackingRegressor(
    estimators=[("rf", RandomForestRegressor(**H["random_forest"])),
                ("gb", GradientBoostingRegressor(**H["gradient_boosting"])),
                ("xgb", xgb.XGBRegressor(**H["xgboost"]))],
    final_estimator=LinearRegression())
model.fit(X[tr], y[tr])

for name, idx in (("train", tr), ("test", te)):
    p = model.predict(X[idx])
    print(name, "R2 = %.3f  MAE = %.3f eV  RMSE = %.3f eV" % (
        r2_score(y[idx], p), mean_absolute_error(y[idx], p), mean_squared_error(y[idx], p) ** 0.5))
