import numpy as np
from rdkit import Chem
import selfies as sf
import time

import deepsmiles
import requests


def array_of_ints_without_nan(arr):
    return arr[~np.isnan(arr)].astype(int)


def try_exccept_nan(f, x):
    try:
        return f(x)
    except:
        return np.nan


def augment_smiles(smiles, int_aug=50, deduplicate=True):
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        if int_aug > 0:
            augmented = [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(int_aug)
            ]
            if deduplicate:
                augmented = list(set(augmented))
            return augmented
        else:
            raise ValueError("int_aug must be greater than zero.")


def get_mode(arr):
    return np.argmax(np.bincount(array_of_ints_without_nan(arr)))


def smiles_to_selfies(smiles):
    """
    Takes a SMILES and return the selfies encoding.
    """

    return [sf.encoder(smiles)]


def smiles_to_deepsmiles(smiles):
    """
    Takes a SMILES and return the DeepSMILES encoding.
    """
    converter = deepsmiles.Converter(rings=True, branches=True)
    return converter.encode(smiles)


def smiles_to_canoncial(smiles):
    """
    Takes a SMILES and return the canoncial SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def smiles_to_inchi(smiles):
    """
    Takes a SMILES and return the InChI.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol)


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


def smiles_to_iupac_name(smiles: str):
    """Use the chemical name resolver https://cactus.nci.nih.gov/chemical/structure.
    If this does not work, use pubchem.
    """
    import pubchempy as pcp

    try:
        time.sleep(0.001)
        rep = "iupac_name"
        url = CACTUS.format(smiles, rep)
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        name = response.text
        if "html" in name:
            return None
        return name
    except Exception:
        try:
            compound = pcp.get_compounds(smiles, "smiles")
            return compound[0].iupac_name
        except Exception:
            return None


def _try_except_none(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


def line_reps_from_smiles(smiles: str):
    """
    Takes a SMILES and returns a dictionary with the different representations.
    Use None if some representation cannot be computed.
    """
    representations = {
        "smiles": smiles,
        "selfies": _try_except_none(smiles_to_selfies, smiles),
        "deepsmiles": _try_except_none(smiles_to_deepsmiles, smiles),
        "canonical": _try_except_none(smiles_to_canoncial, smiles),
        "inchi": _try_except_none(smiles_to_inchi, smiles),
        "iupac_name": _try_except_none(smiles_to_iupac_name, smiles),
    }
    return representations
