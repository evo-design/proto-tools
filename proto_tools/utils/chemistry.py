"""proto_tools/utils/chemistry.py.

Chemistry utilities for small molecules and ligands.
"""

import warnings


def validate_smiles(smiles: str, verbose: bool = True) -> bool:
    """Validate SMILES string using RDKit if available.

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
            if verbose:  # type: ignore[unreachable]
                warnings.warn(
                    f"RDKit could not parse SMILES: '{smiles}'. This may not be a valid molecule.", stacklevel=2
                )
            return False
        return True
    except ImportError:
        if verbose:
            warnings.warn("RDKit not installed. Cannot validate SMILES.", stacklevel=2)
        return False
