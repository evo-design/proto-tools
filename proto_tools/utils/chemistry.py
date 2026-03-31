"""proto_tools/utils/chemistry.py

Chemistry utilities for small molecules and ligands."""
from __future__ import annotations

import warnings


def validate_smiles(smiles: str, verbose: bool = True) -> bool:
    """
    Validate SMILES string using RDKit if available.

    Args:
        smiles (str): The SMILES string to validate.
        verbose (bool): Print warnings.

    Returns:
        bool: True if valid SMILES, False if invalid or RDKit unavailable.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if verbose:
                warnings.warn(
                    f"RDKit could not parse SMILES: '{smiles}'. "
                    "This may not be a valid molecule."
                )
            return False
        return True
    except ImportError:
        if verbose:
            warnings.warn("RDKit not installed. Cannot validate SMILES.")
        return False
