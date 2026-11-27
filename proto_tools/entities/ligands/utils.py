"""proto_tools/entities/ligands/utils.py.

Utility functions for working with ligands.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from urllib.parse import quote

import requests

if TYPE_CHECKING:
    from rdkit import Chem


# ============================================================================
# Validation
# ============================================================================
def is_smiles_valid(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return True
    except Exception:
        return False


def is_mol_valid(mol: Chem.Mol) -> bool:
    """Check if a RDKit Mol object is valid."""
    if mol is None:
        return False  # type: ignore[unreachable]
    return bool(mol.GetNumAtoms() > 0)


# ============================================================================
# PubChem Retrieval — opt-in, network-bound
# ============================================================================

MAX_RETRIES = 3
TIMEOUT = 5


def fetch_pubchem_txt(url: str) -> str | None:
    """Fetch a PubChem TXT response, with retries and timeout.

    Retries with exponential backoff on 429, 5xx, and network exceptions.
    Other 4xx are terminal.

    Args:
        url (str): The PubChem REST URL.

    Returns:
        str | None: The response text if successful, else None.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200 and resp.text.strip():
                result: str = resp.text.strip()
                return result
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            return None
        except requests.RequestException:
            time.sleep(2**attempt)
    return None


def lookup_smiles_via_pubchem(name: str) -> str:
    """Look up the canonical SMILES for a molecule by name via PubChem.

    Args:
        name (str): Name of the molecule (e.g., ``"Aspirin"``).

    Returns:
        str: Canonical SMILES string.

    Raises:
        ValueError: If no SMILES is found for the given name.
    """
    encoded_name = quote(name, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/TXT"
    txt = fetch_pubchem_txt(url)
    if not txt:
        raise ValueError(f"Could not find SMILES for {name}")
    return txt


def lookup_name_via_pubchem(smiles: str) -> str:
    """Look up the IUPAC name for a SMILES via PubChem.

    For CCD-known ligands, prefer the offline :func:`get_ccd_description`.

    Args:
        smiles (str): Canonical SMILES string.

    Returns:
        str: IUPAC name, or ``"Unknown"`` if not found.
    """
    if not smiles:
        return "Unknown"

    cid_txt = fetch_pubchem_txt(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/TXT")
    if not cid_txt:
        return "Unknown"

    cid = cid_txt.splitlines()[0]

    name_txt = fetch_pubchem_txt(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/TXT")
    return name_txt or "Unknown"
