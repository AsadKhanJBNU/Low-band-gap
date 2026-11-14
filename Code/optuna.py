
# ============================================================
# BAND GAP PREDICTION USING MOLECULAR FINGERPRINTS AND ML MODELS
# Dataset: dataset_285.csv
# SMILES column: smiles
# Target column: band_gap_eV
# Train/Test split: 80% / 20%
# Hyperparameter optimization: Optuna
# ============================================================

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import optuna

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "dataset_285.csv"

SMILES_COLUMN = "smiles"
TARGET_COLUMN = "band_gap_eV"

TEST_SIZE = 0.20
RANDOM_STATE = 42

N_TRIALS = 10
N_SPLITS = 5

# Directory for saving all results
SAVE_DIR = "BandGap_Results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD AND CLEAN DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Check whether the required columns exist
required_columns = [SMILES_COLUMN, TARGET_COLUMN]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' not found in dataset.")

# Remove rows with missing SMILES or target values
df = df.dropna(subset=required_columns).copy()

# Convert target values to numeric
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="coerce"
)

df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

print("Dataset shape after removing missing values:", df.shape)

# ------------------------------------------------------------
# 3. MOLECULAR FINGERPRINT GENERATION FUNCTIONS
# ------------------------------------------------------------

def compute_morgan_fingerprint(smiles):
    """
    Generate Morgan fingerprint.
    Radius = 2
    Number of bits = 1024
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    return fp


def compute_maccs_fingerprint(smiles):
    """
    Generate MACCS structural keys.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return MACCSkeys.GenMACCSKeys(mol)


def compute_rdkit_fingerprint(smiles):
    """
    Generate RDKit topological fingerprint.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.RDKFingerprint(
        mol,
        fpSize=2048
    )


def compute_atompair_fingerprint(smiles):
    """
    Generate hashed AtomPair fingerprint.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return AllChem.GetHashedAtomPairFingerprintAsBitVect(
        mol,
        nBits=2048
    )


# ------------------------------------------------------------
# 4. GENERATE ALL MOLECULAR FINGERPRINTS
# ------------------------------------------------------------

fingerprint_functions = {
    "Morgan": compute_morgan_fingerprint,
    "MACCS": compute_maccs_fingerprint,
    "RDKit": compute_rdkit_fingerprint,
    "AtomPair": compute_atompair_fingerprint
}

fingerprints = {
    name: [] for name in fingerprint_functions
}

valid_targets = []
valid_smiles = []

print("\nGenerating molecular fingerprints...")

for _, row in df.iterrows():

    smiles = str(row[SMILES_COLUMN])
    target = row[TARGET_COLUMN]

    # Validate the SMILES before generating fingerprints
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    # Generate each fingerprint
    generated_fps = {}

    for fp_name, fp_function in fingerprint_functions.items():

        fp = fp_function(smiles)

        if fp is None:
            break

        generated_fps[fp_name] = fp

    # Keep only molecules with valid fingerprints
    if len(generated_fps) == len(fingerprint_functions):

        for fp_name in fingerprint_functions:
            fingerprints[fp_name].append(
                generated_fps[fp_name]
            )

        valid_targets.append(target)
        valid_smiles.append(smiles)


# Convert target values into NumPy array
y = np.asarray(valid_targets, dtype=float)

print("Number of valid molecules:", len(y))

if len(y) < 5:
    raise ValueError("At least five valid molecules are required for 5-fold CV.")

# ------------------------------------------------------------
# 5. CONVERT RDKit FINGERPRINTS TO NUMPY ARRAYS
# ------------------------------------------------------------

from rdkit import DataStructs

X_data = {}

for fp_name, fp_list in fingerprints.items():

    # Convert RDKit bit vectors into numerical arrays
    fp_array = np.zeros(
        (len(fp_list), fp_list[0].GetNumBits()),
        dtype=np.uint8
    )

    for i, fp in enumerate(fp_list):
        DataStructs.ConvertToNumpyArray(
            fp,
            fp_array[i]
        )

    X_data[fp_name] = fp_array

    print(
        f"{fp_name} fingerprint shape:",
        fp_array.shape
    )

# ------------------------------------------------------------
# 6. CREATE ONE CONSISTENT 80:20 TRAIN/TEST SPLIT
# ------------------------------------------------------------

# Generate indices once so every fingerprint uses
# exactly the same training and testing molecules.

indices = np.arange(len(y))

train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

y_train = y[train_indices]
y_test = y[test_indices]

print("\nTraining samples:", len(train_indices))
print("Testing samples:", len(test_indices))

# Save the target splits
np.save(
    os.path.join(SAVE_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(SAVE_DIR, "y_test.npy"),
    y_test
)

# Save the corresponding SMILES splits
pd.DataFrame({
    "smiles": np.array(valid_smiles)[train_indices],
    "band_gap_eV": y_train
}).to_csv(
    os.path.join(SAVE_DIR, "train_data.csv"),
    index=False
)

pd.DataFrame({
    "smiles": np.array(valid_smiles)[test_indices],
    "band_gap_eV": y_test
}).to_csv(
    os.path.join(SAVE_DIR, "test_data.csv"),
    index=False
)

# ------------------------------------------------------------
# 7. DEFINE OPTUNA HYPERPARAMETER SEARCH SPACES
# ------------------------------------------------------------

def suggest_parameters(trial, model_name):

    """
    Define model-specific hyperparameter search spaces.
    Optuna selects the best combination using CV R2.
    """

    if model_name == "Random Forest":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 500
            ),

            "max_depth": trial.suggest_int(
                "max_depth", 2, 20
            ),

            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 20
            ),

            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),

            "bootstrap": trial.suggest_categorical(
                "bootstrap", [True, False]
            )
        }

    elif model_name == "SVR":

        return {
            "C": trial.suggest_float(
                "C", 0.01, 100, log=True
            ),

            "epsilon": trial.suggest_float(
                "epsilon", 0.001, 1, log=True
            ),

            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "rbf", "poly", "sigmoid"]
            ),

            "gamma": trial.suggest_categorical(
                "gamma", ["scale", "auto"]
            ),

            "degree": trial.suggest_int(
                "degree", 2, 5
            ),

            "coef0": trial.suggest_float(
                "coef0", 0.0, 1.0
            ),

            "shrinking": trial.suggest_categorical(
                "shrinking", [True, False]
            )
        }

    elif model_name == "Gradient Boosting":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 500
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.3, log=True
            ),

            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 2, 100
            ),

            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 20
            ),

            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0
            ),

            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            )
        }

    elif model_name == "AdaBoost":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 500
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 1.0, log=True
            ),

            "loss": trial.suggest_categorical(
                "loss", ["linear", "square", "exponential"]
            )
        }

    elif model_name == "XGBoost":

        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 500
            ),

            "max_depth": trial.suggest_int(
                "max_depth", 2, 12
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.3, log=True
            ),

            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.0, 1.0
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.0, 5.0
            ),

            "gamma": trial.suggest_float(
                "gamma", 0.0, 1.0
            )
        }

    elif model_name == "KNeighbors":

        return {
            "n_neighbors": trial.suggest_int(
                "n_neighbors", 1, 50
            ),

            "weights": trial.suggest_categorical(
                "weights", ["uniform", "distance"]
            ),

            "leaf_size": trial.suggest_int(
                "leaf_size", 10, 50
            ),

            "p": trial.suggest_int(
                "p", 1, 2
            )
        }


# ------------------------------------------------------------
# 8. INITIALIZE MACHINE LEARNING MODELS
# ------------------------------------------------------------

def initialize_model(model_name, params):

    """
    Create a fresh model using the selected hyperparameters.
    """

    if model_name == "Random Forest":

        return RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "SVR":

        # Scaling is fitted separately inside each CV fold
        return make_pipeline(
            StandardScaler(),
            SVR(**params)
        )

    elif model_name == "Gradient Boosting":

        return GradientBoostingRegressor(
            **params,
            random_state=RANDOM_STATE
        )

    elif model_name == "AdaBoost":

        return AdaBoostRegressor(
            **params,
            random_state=RANDOM_STATE
        )

    elif model_name == "XGBoost":

        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

    elif model_name == "KNeighbors":

        return KNeighborsRegressor(
            **params,
            n_jobs=-1
        )


# ------------------------------------------------------------
# 9. DEFINE CROSS-VALIDATION OBJECTIVE FUNCTION
# ------------------------------------------------------------

def objective(trial, model_name, X_train):

    """
    Optimize hyperparameters using 5-fold cross-validation.
    The independent test set is not used during optimization.
    """

    # Obtain trial hyperparameters
    params = suggest_parameters(trial, model_name)

    # Initialize the model
    model = initialize_model(model_name, params)

    # Perform shuffled 5-fold cross-validation
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1
    )

    # Return the average validation R2
    return np.mean(scores)


# ------------------------------------------------------------
# 10. OPTUNA HYPERPARAMETER OPTIMIZATION
# ------------------------------------------------------------

model_names = [
    "Random Forest",
    "SVR",
    "Gradient Boosting",
    "AdaBoost",
    "XGBoost",
    "KNeighbors"
]

results = {}
best_params = {}
all_metrics = []

print("\nStarting hyperparameter optimization...")

# Optimize every model using every fingerprint
for fingerprint_name, X in X_data.items():

    print("\n======================================")
    print("Fingerprint:", fingerprint_name)
    print("======================================")

    # Apply the same training/test indices to each fingerprint
    X_train = X[train_indices]
    X_test = X[test_indices]

    # Save fingerprint data splits
    np.save(
        os.path.join(
            SAVE_DIR, f"X_{fingerprint_name}_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            SAVE_DIR, f"X_{fingerprint_name}_test.npy"
        ),
        X_test
    )

    results[fingerprint_name] = {}
    best_params[fingerprint_name] = {}

    for model_name in model_names:

        print(f"\nOptimizing {model_name}...")

        # Create an independent Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE
            )
        )

        # Optimize using cross-validation
        study.optimize(
            lambda trial: objective(
                trial,
                model_name,
                X_train
            ),
            n_trials=N_TRIALS
        )

        # Retrieve the best hyperparameters
        optimal_params = study.best_params

        # Initialize the final model with optimal parameters
        best_model = initialize_model(
            model_name,
            optimal_params
        )

        # Train the model on the complete training set
        best_model.fit(X_train, y_train)

        # Evaluate on the independent test set
        y_pred = best_model.predict(X_test)

        # Calculate performance metrics
        test_r2 = r2_score(y_test, y_pred)

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        # Save model and optimization results
        results[fingerprint_name][model_name] = {
            "Model": best_model,
            "BestParams": optimal_params,
            "CV_R2": study.best_value,
            "Test_R2": test_r2,
            "MAE": mae,
            "RMSE": rmse
        }

        best_params[fingerprint_name][model_name] = optimal_params

        # Store metrics in a table-friendly format
        all_metrics.append({
            "Fingerprint": fingerprint_name,
            "Model": model_name,
            "Best_CV_R2": study.best_value,
            "Test_R2": test_r2,
            "MAE": mae,
            "RMSE": rmse
        })

        print("Best CV R2:", round(study.best_value, 4))
        print("Test R2:", round(test_r2, 4))
        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("Best parameters:", optimal_params)

# ------------------------------------------------------------
# 11. SAVE BEST HYPERPARAMETERS
# ------------------------------------------------------------

# Save hyperparameters for all models and fingerprints
with open(
    os.path.join(SAVE_DIR, "best_hyperparameters.pkl"),
    "wb"
) as f:

    pickle.dump(best_params, f)

# Save hyperparameters in readable JSON format
with open(
    os.path.join(SAVE_DIR, "best_hyperparameters.json"),
    "w"
) as f:

    json.dump(best_params, f, indent=4)


# ------------------------------------------------------------
# 12. SAVE TRAINED MODELS
# ------------------------------------------------------------

# Save each optimized model separately
for fingerprint_name in results:

    for model_name in results[fingerprint_name]:

        model = results[fingerprint_name][model_name]["Model"]

        # Make safe filenames
        safe_model_name = model_name.replace(" ", "_")

        model_filename = (
            f"{fingerprint_name}_{safe_model_name}_model.pkl"
        )

        with open(
            os.path.join(SAVE_DIR, model_filename),
            "wb"
        ) as f:

            pickle.dump(model, f)


# ------------------------------------------------------------
# 13. SAVE PERFORMANCE METRICS
# ------------------------------------------------------------

metrics_df = pd.DataFrame(all_metrics)

# Sort models by test R2
metrics_df = metrics_df.sort_values(
    by="Test_R2",
    ascending=False
)

# Save all performance metrics
metrics_df.to_csv(
    os.path.join(SAVE_DIR, "model_performance.csv"),
    index=False
)

print("\nFinal Model Performance:")
print(metrics_df.to_string(index=False))


# ------------------------------------------------------------
# 14. SAVE BEST MODEL INFORMATION
# ------------------------------------------------------------

best_result = metrics_df.iloc[0]

print("\n======================================")
print("BEST MODEL")
print("======================================")

print("Fingerprint:", best_result["Fingerprint"])
print("Model:", best_result["Model"])
print("Test R2:", best_result["Test_R2"])
print("MAE:", best_result["MAE"])
print("RMSE:", best_result["RMSE"])

# Save the best model information
best_model_info = {
    "Fingerprint": best_result["Fingerprint"],
    "Model": best_result["Model"],
    "Test_R2": float(best_result["Test_R2"]),
    "MAE": float(best_result["MAE"]),
    "RMSE": float(best_result["RMSE"])
}

with open(
    os.path.join(SAVE_DIR, "best_model_info.json"),
    "w"
) as f:

    json.dump(best_model_info, f, indent=4)


print("\n======================================")
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("======================================")

print("All results saved in:", SAVE_DIR)
