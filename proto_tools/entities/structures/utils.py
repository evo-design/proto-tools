"""proto_tools/entities/structures/utils.py.

Utility functions for working with protein structures.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import gemmi
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

# ===============================
# I/O
# ===============================
SUPPORTED_EXTENSIONS = ("pdb", "cif", "mmcif")


def load_structure_file(filepath: Path | str) -> str:
    """Loads the contents of a structure file (PDB or CIF) and returns it as a string.

    Args:
        filepath (Path | str): Path to the structure file (PDB or CIF).

    Returns:
        str: String of content from the structure file.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file extension is not .pdb, .cif, or .mmcif
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Normalize extension check
    suffix_lower = filepath.suffix.lower()
    if not suffix_lower.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(
            f"Invalid structure file extension: {filepath.suffix}. "
            f"Must one of the following extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Read the structure file
    return filepath.read_text(encoding="utf-8")


def detect_structure_format(structure_content: str) -> str:
    """Detect if structure content is CIF or PDB format.

    Args:
        structure_content (str): Structure file content as string

    Returns:
        str: "cif" or "pdb"
    """
    # Strip leading whitespace and get first meaningful lines
    lines = [line.strip() for line in structure_content.split("\n") if line.strip()]

    if not lines:
        raise ValueError("Empty structure content. Must be PDB or CIF format.")

    first_line = lines[0]

    # CIF files start with specific markers
    if first_line.startswith("data_"):
        return "cif"

    # Check for CIF loop_ or category markers in first few lines
    for line in lines[:10]:
        if line.startswith(("loop_", "_")):
            return "cif"

    # PDB files typically start with specific record types
    pdb_keywords = ["HEADER", "TITLE", "ATOM", "HETATM", "MODEL", "CRYST1"]
    for line in lines[:20]:
        if any(line.startswith(keyword) for keyword in pdb_keywords):
            return "pdb"

    # If still unsure, check for CIF-style underscore fields anywhere
    if any("_atom_site" in line or "_entity" in line for line in lines[:50]):
        return "cif"

    # Default to PDB if we see ATOM/HETATM anywhere in first 100 lines
    for line in lines[:100]:
        if line.startswith(("ATOM", "HETATM")):
            return "pdb"

    raise ValueError("Could not determine structure format (CIF or PDB).")


def is_valid_structure(structure_filepath_or_content: str | Path) -> bool:
    """Ensures that a structure content string/file has valid PDB or CIF format.

    by checking the following:
    - there is at least one model in the structure
    - there is at least one atom in the structure

    Args:
        structure_filepath_or_content (str | Path): Path to the structure file or string of
            content (PDB or CIF).

    Returns:
        bool: True if the structure content string/file has valid PDB or CIF format, False otherwise
    """
    try:
        # Determine if input is a file path or content string
        input_str = str(structure_filepath_or_content)

        if input_str.lower().endswith(SUPPORTED_EXTENSIONS):
            # It's a file path - read directly
            structure = gemmi.read_structure(input_str)
        else:
            struct_format = detect_structure_format(input_str)
            if struct_format == "cif":
                doc = gemmi.cif.read_string(input_str)
                structure = gemmi.make_structure_from_block(doc[0])
            elif struct_format == "pdb":
                structure = gemmi.read_pdb_string(input_str)

    except Exception:
        # If parsing fails for any reason, return False
        return False

    # Must have at least one atom
    has_atoms = False
    for model in structure:
        for chain in model:
            for residue in chain:
                if len(residue) > 0:  # residue has atoms
                    has_atoms = True
                    break
            if has_atoms:
                break
        if has_atoms:
            break

    return has_atoms


# ===============================
# Geometry
# ===============================


def pdb_file_to_atomarray(pdb_path: str | StringIO) -> AtomArray:
    """Convert a PDB file to a Biotite AtomArray."""
    return PDBFile.read(pdb_path).get_structure(model=1)


def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int) -> AtomArray:
    """Extract atoms within a specific residue range."""
    return atoms[np.logical_and(atoms.res_id >= start, atoms.res_id < end)]


def _is_Nx3(array: np.ndarray) -> bool:
    """Check if array is Nx3 shaped."""
    return len(array.shape) == 2 and array.shape[1] == 3


def pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances between all coordinates."""
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."  # noqa: S101
    m = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(m, axis=-1)
    return distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]  # type: ignore[no-any-return]


def adjacent_distances(coordinates: np.ndarray) -> np.ndarray:
    """Calculate distances between adjacent coordinates."""
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."  # noqa: S101
    m = coordinates - np.roll(coordinates, shift=1, axis=0)
    return np.linalg.norm(m, axis=-1)  # type: ignore[no-any-return]


def get_centroid(coordinates: np.ndarray) -> np.ndarray:
    """Calculate the centroid of coordinates."""
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."  # noqa: S101
    return coordinates.mean(axis=0).reshape(1, 3)  # type: ignore[no-any-return]


def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
    """Computes the distances from each of the coordinates to the.

    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."  # noqa: S101
    centroid = get_centroid(coordinates)
    m = coordinates - centroid
    return np.linalg.norm(m, axis=-1)  # type: ignore[no-any-return]


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    """Extract backbone atoms (CA, N, C) from an AtomArray."""
    return atoms[(atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")]


# ===============================
# Conversion
# ===============================


def convert_pdb_str_to_cif_str(pdb_content: str) -> str:
    """Converts a structure from PDB format to mmCIF format using gemmi.

    Args:
        pdb_content (str): Structure content in PDB format

    Returns:
        str: Structure in mmCIF format (empty string if input is empty)
    """
    if not pdb_content.strip():
        return ""

    try:
        structure = gemmi.read_pdb_string(pdb_content)
        doc = structure.make_mmcif_document()
        return doc.as_string()
    except Exception as e:
        raise ValueError(f"Failed to convert PDB to CIF: {e}") from e


def convert_cif_str_to_pdb_str(cif_content: str) -> str:
    """Converts a structure from mmCIF format to PDB format using gemmi.

    WARNING: PDB format has limitations that may cause data loss:
    - Chain IDs limited to 1 character (multi-character chains truncated)
    - Coordinate precision limited to 3 decimal places
    - Line length limited to 80 characters
    - Atom serial numbers limited to 99,999
    - Residue numbers limited to 9,999

    Args:
        cif_content (str): Structure content in mmCIF format

    Returns:
        str: Structure in PDB format (empty string if input is empty)
    """
    if not cif_content.strip():
        return ""

    try:
        doc = gemmi.cif.read_string(cif_content)

        # Find first valid structure block
        structure = None
        for block in doc:
            try:
                structure = gemmi.make_structure_from_block(block)
                if structure is not None and len(structure) > 0:  # type: ignore[redundant-expr]
                    break
            except Exception:  # noqa: S112 -- skip unparseable blocks
                continue

        if structure is None:
            raise ValueError("No valid structure found in CIF content")

        # Convert to PDB string using gemmi's make_pdb_string
        return structure.make_pdb_string()

    except Exception as e:
        raise ValueError(f"Failed to convert CIF to PDB: {e}") from e
