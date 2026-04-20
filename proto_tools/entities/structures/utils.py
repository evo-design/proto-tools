"""proto_tools/entities/structures/utils.py.

Utility functions for working with protein structures.
"""

import warnings
from io import StringIO
from pathlib import Path
from typing import Literal

import gemmi
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

# PDB format field limits — anything beyond these is truncated, remapped, or dropped
# when converting from CIF to PDB.
_PDB_MAX_CHAIN_ID_LEN = 1
_PDB_MAX_RESIDUE_NAME_LEN = 3
_PDB_MAX_ATOM_NAME_LEN = 4
_PDB_MAX_ATOMS = 99_999
_PDB_MAX_RESIDUES_PER_CHAIN = 9_999

# CIF extension categories that PDB truly cannot represent. `_pdbx_struct_assembly` and
# `_pdbx_struct_oper_list` are intentionally excluded — they appear in virtually every
# PDB-derived mmCIF (even single-chain monomers) and would turn the warning into noise.
# Non-crystallographic symmetry operators are rare and genuinely unrepresentable.
_CIF_ONLY_METADATA_MARKERS = ("_struct_ncs_oper",)

# Upper bound for treating a string as a possible filesystem path. Structure content
# strings are routinely 5-50 KB; any string longer than this is definitely content,
# not a path, and we skip the filesystem probe to avoid pointless work.
_MAX_PATH_STRING_LEN = 4096

# ===============================
# I/O
# ===============================
SUPPORTED_EXTENSIONS = ("pdb", "cif", "mmcif")


def looks_like_structure_path(value: object) -> bool:
    """Return True if ``value`` is a filesystem path to an existing structure file.

    Used by ``Structure._handle_construction`` so that ``Structure(structure="foo.pdb")``
    transparently loads from disk (restoring the old plain-class ergonomics) while
    content strings pass straight through.

    Args:
        value (object): Value to probe — a ``Path`` or a string that may be either a
            filesystem path or raw PDB/CIF content.

    Returns:
        bool: True if ``value`` points to an existing file with a supported extension.
    """
    if isinstance(value, Path):
        try:
            return value.is_file()
        except OSError:
            return False
    if isinstance(value, str):
        # Content strings are multi-line and/or long; only probe the filesystem for
        # short single-line strings whose suffix matches a structure extension.
        if "\n" in value or len(value) > _MAX_PATH_STRING_LEN:
            return False
        if not value.lower().endswith(SUPPORTED_EXTENSIONS):
            return False
        try:
            return Path(value).is_file()
        except OSError:
            return False
    return False


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


def _serialize_gemmi(
    struct: gemmi.Structure,
    target_format: Literal["pdb", "cif"],
    *,
    source_format: Literal["pdb", "cif"] | None = None,
    cif_content_for_warnings: str | None = None,
) -> str:
    """Single emission path for ``gemmi.Structure → str``.

    This function is the standardized conversion point from ``gemmi.Structure``
    to content-string for structure objects; all serializers and re-emitters
    route through here. CIF→PDB conversions defer to
    :func:`_warn_cif_to_pdb_lossy` for warning behavior.

    Args:
        struct (gemmi.Structure): The (possibly mutated) gemmi.Structure to emit.
        target_format (Literal["pdb", "cif"]): Output format.
        source_format (Literal["pdb", "cif"] | None): Original format. ``"cif"``
            with ``target_format="pdb"`` triggers the lossy-warning scan. Pass
            ``None`` when there's no meaningful single source (e.g. ``concat``
            building a struct from scratch).
        cif_content_for_warnings (str | None): Raw CIF text scanned for CIF-only
            metadata markers (NCS operators, etc.). When ``None``, only
            structure-level warnings fire (long names, atom counts).

    Returns:
        str: Serialized structure content in the requested format.
    """
    if target_format == "cif":
        return struct.make_mmcif_document().as_string()
    if source_format == "cif":
        _warn_cif_to_pdb_lossy(struct, cif_content_for_warnings or "")
    return struct.make_pdb_string()


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
        return _serialize_gemmi(structure, "cif", source_format="pdb")
    except Exception as e:
        raise ValueError(f"Failed to convert PDB to CIF: {e}") from e


def _warn_cif_to_pdb_lossy(structure: gemmi.Structure, cif_content: str) -> None:
    """Emit warnings for CIF content that can't round-trip cleanly through PDB format.

    PDB has fixed-width fields and hard size caps that modern mmCIF files routinely
    exceed. When conversion would silently truncate, remap, or drop information, the
    caller should know. Warns on:

    * chain IDs longer than ``_PDB_MAX_CHAIN_ID_LEN`` (1 character)
    * atom names longer than ``_PDB_MAX_ATOM_NAME_LEN`` (4 characters)
    * residue names longer than ``_PDB_MAX_RESIDUE_NAME_LEN`` (3 characters)
    * total atom count exceeding ``_PDB_MAX_ATOMS`` (99,999)
    * per-chain residue count exceeding ``_PDB_MAX_RESIDUES_PER_CHAIN`` (9,999)
    * CIF-only metadata categories (``_CIF_ONLY_METADATA_MARKERS``, e.g. NCS
      operators) present in the source CIF text

    Args:
        structure (gemmi.Structure): Parsed structure used to scan for per-atom,
            per-residue, and per-chain PDB field overflows.
        cif_content (str): Raw mmCIF source text, scanned for CIF-only metadata
            categories that have no PDB equivalent.
    """
    long_chain_ids: set[str] = set()
    long_residue_names: set[str] = set()
    long_atom_names: set[str] = set()
    total_atoms = 0
    chains_over_residue_cap: set[str] = set()

    for model in structure:
        for chain in model:
            if len(chain.name) > _PDB_MAX_CHAIN_ID_LEN:
                long_chain_ids.add(chain.name)
            residue_count = 0
            for residue in chain:
                residue_count += 1
                if len(residue.name) > _PDB_MAX_RESIDUE_NAME_LEN:
                    long_residue_names.add(residue.name)
                for atom in residue:
                    total_atoms += 1
                    if len(atom.name) > _PDB_MAX_ATOM_NAME_LEN:
                        long_atom_names.add(atom.name)
            if residue_count > _PDB_MAX_RESIDUES_PER_CHAIN:
                chains_over_residue_cap.add(chain.name)

    if long_chain_ids:
        warnings.warn(
            f"CIF→PDB conversion: {len(long_chain_ids)} chain ID(s) exceed PDB's "
            f"{_PDB_MAX_CHAIN_ID_LEN}-character limit and will be truncated or remapped "
            f"(e.g., {sorted(long_chain_ids)[:3]}).",
            stacklevel=4,
        )
    if long_atom_names:
        warnings.warn(
            f"CIF→PDB conversion: {len(long_atom_names)} atom name(s) exceed PDB's "
            f"{_PDB_MAX_ATOM_NAME_LEN}-character field width and may be mangled "
            f"(e.g., {sorted(long_atom_names)[:3]}).",
            stacklevel=4,
        )
    if total_atoms > _PDB_MAX_ATOMS:
        warnings.warn(
            f"CIF→PDB conversion: structure has {total_atoms} atoms, exceeding PDB's "
            f"{_PDB_MAX_ATOMS}-atom cap. Atoms beyond the cap will be dropped.",
            stacklevel=4,
        )
    if chains_over_residue_cap:
        warnings.warn(
            f"CIF→PDB conversion: {len(chains_over_residue_cap)} chain(s) exceed PDB's "
            f"{_PDB_MAX_RESIDUES_PER_CHAIN}-residue-per-chain cap and will be truncated "
            f"(chains: {sorted(chains_over_residue_cap)[:3]}).",
            stacklevel=4,
        )
    if long_residue_names:
        warnings.warn(
            f"CIF→PDB conversion: {len(long_residue_names)} residue name(s) exceed PDB's "
            f"{_PDB_MAX_RESIDUE_NAME_LEN}-character limit and will be truncated "
            f"(e.g., {sorted(long_residue_names)[:3]}).",
            stacklevel=4,
        )
    if any(marker in cif_content for marker in _CIF_ONLY_METADATA_MARKERS):
        warnings.warn(
            "CIF→PDB conversion: source CIF contains extension metadata (assemblies, "
            "NCS operations, or rich entity info) that PDB format cannot represent; "
            "this metadata will be lost.",
            stacklevel=4,
        )


def convert_cif_str_to_pdb_str(cif_content: str) -> str:
    """Converts a structure from mmCIF format to PDB format using gemmi.

    Emits ``UserWarning`` via ``warnings.warn`` when the source CIF contains data that
    PDB's fixed-width fields or size caps cannot represent (long chain IDs, long atom
    or residue names, >99,999 atoms, >9,999 residues per chain, or CIF-only metadata
    categories). PDB→CIF is lossless and has no equivalent warnings.

    Args:
        cif_content (str): Structure content in mmCIF format.

    Returns:
        str: Structure in PDB format (empty string if input is empty).

    Raises:
        ValueError: If the CIF content is not parseable or contains no valid structure.
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

        return _serialize_gemmi(
            structure,
            "pdb",
            source_format="cif",
            cif_content_for_warnings=cif_content,
        )

    except Exception as e:
        raise ValueError(f"Failed to convert CIF to PDB: {e}") from e
