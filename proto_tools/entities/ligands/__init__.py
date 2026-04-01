"""Small-molecule ligand representations and CCD utilities."""

from proto_tools.entities.ligands.ccd_utils import (
    CCD_DATABASE_PATH,
    get_ccd_description,
    is_valid_ccd_code,
    map_ccd_code_to_smiles,
    map_smiles_to_ccd_code,
)
from proto_tools.entities.ligands.ligands import Fragment, Ligands
from proto_tools.entities.ligands.utils import get_name_from_smiles, get_smiles_from_name, is_smiles_valid

__all__ = [
    # Base Classes
    "Fragment",
    "Ligands",
    # Utilities
    "is_smiles_valid",
    "get_name_from_smiles",
    "get_smiles_from_name",
    # CCD Utilities
    "CCD_DATABASE_PATH",
    "get_ccd_description",
    "is_valid_ccd_code",
    "map_ccd_code_to_smiles",
    "map_smiles_to_ccd_code",
]
