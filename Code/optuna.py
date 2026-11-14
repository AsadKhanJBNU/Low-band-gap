import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import optuna



# Load dataset
df = pd.read_csv('bandgap.csv')

# Splitting the DataFrame into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Remove rows with invalid SMILES
df = df.dropna(subset=['Smiles'])

# Function to compute MACCS keys
def compute_maccs_keys(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return MACCSkeys.GenMACCSKeys(mol)
    else:
        return None

# Function to compute RDKit fingerprints
def compute_rdkit_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.RDKFingerprint(mol)
    else:
        return None

# Function to compute Morgan fingerprints
def compute_morgan_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Morgan fingerprint with radius 2 and 1024 bits
    else:
        return None

# Function to compute AtomPair fingerprints
def compute_atompair_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
    else:
        return None



morgan_fps = []
maccs_fps = []
rdkit_fps = []
atompair_fps = []
valid_indices = []  # Track indices of valid SMILES
for i, smi in enumerate(df['Smiles']):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        morgan_fps.append(compute_morgan_fingerprint(smi))
        maccs_fps.append(compute_maccs_keys(smi))
        rdkit_fps.append(compute_rdkit_fingerprint(smi))
        atompair_fps.append(compute_atompair_fingerprint(smi))
        valid_indices.append(i)



# Filter dataframe based on valid indices
df = df.iloc[valid_indices]

# Split the dataset into training and testing sets for each type of fingerprint
X_morgan_train, X_morgan_test, y_train, y_test = train_test_split(morgan_fps, df['BG'], test_size=0.2, random_state=42)
X_maccs_train, X_maccs_test, _, _ = train_test_split(maccs_fps, df['BG'], test_size=0.2, random_state=42)
X_rdkit_train, X_rdkit_test, _, _ = train_test_split(rdkit_fps, df['BG'], test_size=0.2, random_state=42)
X_atompair_train, X_atompair_test, _, _ = train_test_split(atompair_fps, df['BG'], test_size=0.2, random_state=42)



import numpy as np

# Define the directory to save the files
save_dir = '/home/basir/basir/BandGap/Labels/'

# Save the data splits as numpy arrays in the specified directory
np.save(save_dir + 'X_morgan_train.npy', X_morgan_train)
np.save(save_dir + 'X_morgan_test.npy', X_morgan_test)
np.save(save_dir + 'X_maccs_train.npy', X_maccs_train)
np.save(save_dir + 'X_maccs_test.npy', X_maccs_test)
np.save(save_dir + 'X_rdkit_train.npy', X_rdkit_train)
np.save(save_dir + 'X_rdkit_test.npy', X_rdkit_test)
np.save(save_dir + 'X_atompair_train.npy', X_atompair_train)
np.save(save_dir + 'X_atompair_test.npy', X_atompair_test)
np.save(save_dir + 'y_train.npy', y_train)
np.save(save_dir + 'y_test.npy', y_test)




import numpy as np

# Define the directory where the files are saved
load_dir = '/home/basir/basir/BandGap/Labels/'

# Load the data splits
X_morgan_train = np.load(load_dir + 'X_morgan_train.npy')
X_morgan_test = np.load(load_dir + 'X_morgan_test.npy')
X_maccs_train = np.load(load_dir + 'X_maccs_train.npy')
X_maccs_test = np.load(load_dir + 'X_maccs_test.npy')
X_rdkit_train = np.load(load_dir + 'X_rdkit_train.npy')
X_rdkit_test = np.load(load_dir + 'X_rdkit_test.npy')
X_atompair_train = np.load(load_dir + 'X_atompair_train.npy')
X_atompair_test = np.load(load_dir + 'X_atompair_test.npy')
y_train = np.load(load_dir + 'y_train.npy')
y_test = np.load(load_dir + 'y_test.npy')




search_space = {
    'Random Forest': {
        'n_estimators': optuna.distributions.IntDistribution(50, 500),
        'max_depth': optuna.distributions.IntDistribution(1, 20),  # Adjusted range
        'min_samples_split': optuna.distributions.IntDistribution(2, 20),  # Adjusted range
        'min_samples_leaf': optuna.distributions.IntDistribution(1, 20),  # Adjusted range
        'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2', None]),
        'max_leaf_nodes': optuna.distributions.IntDistribution(2, 100),  # Adjusted upper bound
        'min_impurity_decrease': optuna.distributions.FloatDistribution(0.0, 0.5),
        'random_state': optuna.distributions.IntDistribution(0, 100),
        'ccp_alpha': optuna.distributions.FloatDistribution(0.0, 0.5)
    },
    'SVR': {
        'degree': optuna.distributions.IntDistribution(0, 5),
        'gamma': optuna.distributions.CategoricalDistribution(['scale', 'auto']) or optuna.distributions.FloatDistribution(0.0, None),
        'coef0': optuna.distributions.FloatDistribution(-1.0, 1.0),
        'tol': optuna.distributions.FloatDistribution(1e-5, 1e-1),
        'C': optuna.distributions.FloatDistribution(0.01, 100, log=True),
        'epsilon': optuna.distributions.FloatDistribution(0.001, 1, log=True),
        'shrinking': optuna.distributions.CategoricalDistribution([True, False]),
        'cache_size': optuna.distributions.FloatDistribution(100, 1000),
        'max_iter': optuna.distributions.IntDistribution(100, 1000),
        'kernel': optuna.distributions.CategoricalDistribution(['linear', 'poly', 'rbf', 'sigmoid'])
    },
    'Gradient Boosting': {
        'n_estimators': optuna.distributions.IntDistribution(50, 500),
        'max_depth': optuna.distributions.IntDistribution(1, 20),  # Adjusted range
        'learning_rate': optuna.distributions.FloatDistribution(0.001, 1, log=True),
        'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
        'min_samples_split': optuna.distributions.IntDistribution(2, 20),
        'min_samples_leaf': optuna.distributions.IntDistribution(1, 20),
        'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2', None]),
        'max_leaf_nodes': optuna.distributions.IntDistribution(2, 100),  # Adjusted upper bound
        'min_impurity_decrease': optuna.distributions.FloatDistribution(0.0, 0.5),
        'random_state': optuna.distributions.IntDistribution(0, 100)
    },


    'AdaBoost': {
        'n_estimators': optuna.distributions.IntDistribution(50, 500),
        'learning_rate': optuna.distributions.FloatDistribution(0.001, 1, log=True),
        'random_state': optuna.distributions.IntDistribution(0, 100)
    },
    'XGBoost': {
        'n_estimators': optuna.distributions.IntDistribution(50, 500),
        'max_depth': optuna.distributions.IntDistribution(1, 20),  # Adjusted range
        'learning_rate': optuna.distributions.FloatDistribution(0.001, 1, log=True),
        'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
        'colsample_bytree': optuna.distributions.FloatDistribution(0.5, 1.0),
        'colsample_bylevel': optuna.distributions.FloatDistribution(0.5, 1.0),
        'colsample_bynode': optuna.distributions.FloatDistribution(0.5, 1.0),
        'reg_alpha': optuna.distributions.FloatDistribution(0.0, 1.0),
        'reg_lambda': optuna.distributions.FloatDistribution(0.0, 1.0),
        'random_state': optuna.distributions.IntDistribution(0, 100),
        'tree_method': optuna.distributions.CategoricalDistribution(['hist', 'gpu_hist']),
        'gpu_id': optuna.distributions.IntDistribution(0, 3)
    },
    'KNeighbors': {
        'n_neighbors': optuna.distributions.IntDistribution(3, 50),
        'weights': optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
        'leaf_size': optuna.distributions.IntDistribution(10, 50),
        'p': optuna.distributions.IntDistribution(1, 5)
    }
}




# Objective function for Optuna
def objective(trial):
    # Get the hyperparameters from the search space
    params = {param_name: trial._suggest(param_name, distribution) for param_name, distribution in search_space[name].items()}
    
    # Initialize the model with suggested hyperparameters
    model = models[name].set_params(**params)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute R^2 score
    r2 = r2_score(y_test, y_pred)
    
    return r2



# Initialize models
models = {
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'XGBoost': XGBRegressor(),
    'KNeighbors': KNeighborsRegressor()
}



# Hyperparameter tuning with Optuna
best_params = {}
results = {}

for name, model in models.items():
    for X_train, X_test, fingerprint_type in [(X_morgan_train, X_morgan_test, 'Morgan'),
                                              (X_maccs_train, X_maccs_test, 'MACCS'),
                                              (X_rdkit_train, X_rdkit_test, 'RDKit')]:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        
        # Get the best parameters
        best_params[name] = study.best_params
        
        # Create the best model with the best parameters
        best_model = model.set_params(**study.best_params)
        
        # Fit the best model
        best_model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = best_model.predict(X_test)
        
        # Compute R^2 score
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        if name not in results:
            results[name] = {}
        results[name][fingerprint_type] = {'R^2': r2, 'Model': best_model,'BestParms':study.best_params}




best_params




# Iterate over the results dictionary
best_params = {}
for model_name, model_results in results.items():
    for fingerprint_type, result in model_results.items():
        
        if fingerprint_type == 'RDKit':
            best_params[model_name] = result['BestParms']





import pickle

with open('best_params_rdkit.pkl', 'wb') as f:
    pickle.dump(best_params, f)