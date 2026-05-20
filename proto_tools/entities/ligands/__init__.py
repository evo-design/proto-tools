"""Small-molecule ligand representations and CCD utilities."""

from proto_tools.entities.ligands.ccd_utils import (
    CCD_DATABASE_PATH,
    count_heavy_atoms_for_ccd,
    get_ccd_description,
    is_valid_ccd_code,
    map_ccd_code_to_smiles,
    map_smiles_to_ccd_code,
)
from proto_tools.entities.ligands.ligands import Fragment, Ligands
from proto_tools.entities.ligands.utils import (
    is_smiles_valid,
    lookup_name_via_pubchem,
    lookup_smiles_via_pubchem,
)

__all__ = [
    # Base Classes
    "Fragment",
    "Ligands",
    # Utilities
    "is_smiles_valid",
    # PubChem lookups (opt-in, network-bound — not called by library construction)
    "lookup_name_via_pubchem",
    "lookup_smiles_via_pubchem",
    # CCD Utilities (offline)
    "CCD_DATABASE_PATH",
    "count_heavy_atoms_for_ccd",
    "get_ccd_description",
    "is_valid_ccd_code",
    "map_ccd_code_to_smiles",
    "map_smiles_to_ccd_code",
]
