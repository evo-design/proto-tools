"""
proto_tools/entities/ligands/utils.py

Utility functions for working with ligands.
"""

import time
from urllib.parse import quote

import requests
from rdkit import Chem


# ===============================
# Validation
# ===============================
def is_smiles_valid(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return True
    except Exception:
        return False

def is_mol_valid(mol: Chem.Mol) -> bool:
    """
    Check if a RDKit Mol object is valid.
    """
    if mol is None:
        return False
    if not mol.GetNumAtoms() > 0:
        return False
    return True

# ===============================
# PubChem Retrieval
# ===============================

MAX_RETRIES = 10
TIMEOUT = 10

import time

import requests

TIMEOUT = 10
MAX_RETRIES = 10

def fetch_pubchem_txt(url: str) -> str | None:
    """
    Fetch a PubChem TXT response from the given URL, with retries and timeout.

    Args:
        url (str): The PubChem REST URL

    Returns:
        str | None: The response text if successful, else None
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text.strip()
            elif resp.status_code == 429:  # rate limited
                time.sleep(2 ** attempt)
                continue
        except requests.RequestException:
            time.sleep(TIMEOUT)
    return None


def get_smiles_from_name(name: str) -> str:
    """
    Retrieve the canonical SMILES for a molecule given its name using PubChem.

    Args:
        name (str): Name of the molecule (e.g., "Aspirin")

    Returns:
        str: Canonical SMILES string

    Raises:
        ValueError: If the molecule is not found.
    """
    encoded_name = quote(name, safe='')
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/TXT"
    txt = fetch_pubchem_txt(url)
    if not txt:
        raise ValueError(f"Could not find SMILES for {name}")
    return txt


def get_name_from_smiles(smiles: str) -> str:
    """
    Retrieve the primary compound name from PubChem given a SMILES string. If
    the molecule is not found, returns "Unknown".

    Args:
        smiles (str): Canonical SMILES string

    Returns:
        str: Name of the compound (string)
    """
    if not smiles:
        return "Unknown"

    cid_txt = fetch_pubchem_txt(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT"
    )
    if not cid_txt:
        return "Unknown"

    cid = cid_txt.splitlines()[0]

    name_txt = fetch_pubchem_txt(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/TXT"
    )
    return name_txt or "Unknown"
