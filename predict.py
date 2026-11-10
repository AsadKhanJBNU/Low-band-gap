"""Predict an optical gap from a SMILES string with the revised ExtraTrees model."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
EXAMPLE = "CC1(C)c2ccccc2-c2ccc(-c3ccc(C=C(C#N)C#N)c4nsnc34)cc21"


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse: {smiles}")
    bit_vector = Chem.RDKFingerprint(mol, fpSize=2048)
    array = np.zeros((2048,), dtype=np.float32)
    ConvertToNumpyArray(bit_vector, array)
    return bit_vector, array


def main():
    smiles = sys.argv[1] if len(sys.argv) > 1 else EXAMPLE
    model = joblib.load(ROOT / "Model" / "extratrees_rdkit_455.joblib")
    molecules = pd.read_csv(ROOT / "Dataset" / "molecules.csv")
    query_bits, query = fingerprint(smiles)
    predicted = float(model.predict(query.reshape(1, -1))[0])
    bank = [fingerprint(value)[0] for value in molecules["canonical_smiles"]]
    similarity = max(BulkTanimotoSimilarity(query_bits, bank))
    print(f"Predicted optical gap: {predicted:.2f} eV")
    print(f"Similarity to the nearest training molecule: {similarity:.2f}")
    if similarity < 0.80:
        print("Warning: this molecule is dissimilar to the training set.")
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    recorded = molecules.loc[molecules["canonical_smiles"].eq(canonical), "BG"]
    if len(recorded):
        print(f"This structure is in the training set. Recorded gap: {float(recorded.iloc[0]):.2f} eV.")


if __name__ == "__main__":
    main()
