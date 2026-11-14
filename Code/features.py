import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

def to_array(fp):
    a = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a

def rdkit_fp(m): return to_array(Chem.RDKFingerprint(m))
def morgan_fp(m): return to_array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024))
def maccs_fp(m): return to_array(MACCSkeys.GenMACCSKeys(m))

FINGERPRINTS = {"RDKit": rdkit_fp, "Morgan": morgan_fp, "MACCS": maccs_fp}
