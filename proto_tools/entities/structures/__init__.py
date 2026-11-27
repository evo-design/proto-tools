"""Macromolecular structure representations and format conversion."""

from proto_tools.entities.structures.examples import GFP_CIF_PATH, get_gfp_structure
from proto_tools.entities.structures.selection import (
    ChainSelection,
    ResidueSelection,
    SingleChainSelection,
    StructureInputBase,
)
from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.entities.structures.structure_ensemble import StructureEnsemble
from proto_tools.entities.structures.utils import (
    adjacent_distances,
    convert_cif_str_to_pdb_str,
    convert_pdb_str_to_cif_str,
    detect_structure_format,
    distances_to_centroid,
    get_atomarray_in_residue_range,
    get_backbone_atoms,
    get_centroid,
    is_valid_structure,
    load_structure_file,
    pairwise_distances,
    pdb_file_to_atomarray,
)

__all__ = [
    # IO
    "load_structure_file",
    "detect_structure_format",
    "is_valid_structure",
    # Geometry
    "pdb_file_to_atomarray",
    "get_atomarray_in_residue_range",
    "pairwise_distances",
    "adjacent_distances",
    "get_centroid",
    "distances_to_centroid",
    "get_backbone_atoms",
    # Convert
    "convert_pdb_str_to_cif_str",
    "convert_cif_str_to_pdb_str",
    # Structure
    "Structure",
    "BFactorType",
    "StructureEnsemble",
    # Selection
    "ChainSelection",
    "SingleChainSelection",
    "ResidueSelection",
    "StructureInputBase",
    # Examples
    "GFP_CIF_PATH",
    "get_gfp_structure",
]
