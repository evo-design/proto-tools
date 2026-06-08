"""Utilities for working with the wwPDB Chemical Component Dictionary (CCD).

The CCD database contains information about all chemical components (ligands,
modified residues, etc.) found in PDB structures. This module provides functions
to map between SMILES strings and CCD codes, and to validate modification codes.

The database file uses OpenEye-canonical SMILES, which differ from RDKit-canonical
SMILES for the same molecule. All public functions in this module canonicalize via
RDKit internally so that lookups work regardless of the input SMILES representation.

Data source: https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

CCD_DATABASE_PATH = Path(__file__).parent / "ccd_maps" / "Components-smiles-stereo-oe.smi"
CCD_CACHE_PICKLE_PATH = Path(__file__).parent / "ccd_maps" / "ccd_caches.pkl"
_CCD_CACHE_PICKLE_VERSION = 1


def _ensure_ccd_database() -> None:
    """Ensure the CCD database exists."""
    if not CCD_DATABASE_PATH.exists():
        logger.error(f"CCD database not found at {CCD_DATABASE_PATH}")
        raise FileNotFoundError(f"CCD database not found at {CCD_DATABASE_PATH}")


# ============================================================================
# CCD code caches
#
# Four independent caches are built from a single pass over the database file:
#
# - ``_ALL_CCD_CODES``: the raw set of every CCD code in the file. Used by
#   ``is_valid_ccd_code`` and ``get_all_ccd_codes`` so validity checks remain
#   broad — codes whose SMILES RDKit cannot parse are still considered valid
#   (they exist in the wwPDB CCD file).
#
# - ``_RDKIT_CANONICAL_TO_CCD`` / ``_CCD_TO_RDKIT_CANONICAL``: bidirectional
#   maps keyed by RDKit-canonical SMILES. Used by ``map_smiles_to_ccd_code`` /
#   ``map_ccd_code_to_smiles`` for the primary lookup. Codes whose SMILES RDKit
#   cannot parse are excluded (no canonical form means no lookup is possible).
#
# - ``_INCHIKEY_TO_CCD``: secondary lookup keyed by InChIKey, used as a fallback
#   when the canonical-SMILES match misses. InChIKey is the IUPAC standard
#   chemical identifier — designed for cross-tool / cross-database identity
#   matching (used by ChEMBL, RCSB, PubChem, PDBe for the same purpose). It
#   normalizes more aggressively than canonical SMILES (handles tautomers,
#   ionization, aromaticity-perception edge cases). Only *unambiguous*
#   InChIKeys are added — ~500/49k CCD entries collide on InChIKey (mostly
#   single-atom ions and tautomer/stereo variants), and silently picking one
#   would be worse than reporting no match. Ambiguous keys are dropped.
# ============================================================================
_ALL_CCD_CODES: set[str] | None = None
_RDKIT_CANONICAL_TO_CCD: dict[str, str] | None = None
_CCD_TO_RDKIT_CANONICAL: dict[str, str] | None = None
_INCHIKEY_TO_CCD: dict[str, str] | None = None


def _build_caches() -> tuple[set[str], dict[str, str], dict[str, str], dict[str, str]]:
    """Build the raw-code set + canonical maps + InChIKey map in one pass.

    Returns:
        tuple[set[str], dict[str, str], dict[str, str], dict[str, str]]:
            (all_codes, smiles_to_ccd, ccd_to_smiles, inchikey_to_ccd).
    """
    from collections import defaultdict

    from rdkit import Chem, RDLogger

    _ensure_ccd_database()

    all_codes: set[str] = set()
    smiles_to_ccd: dict[str, str] = {}
    ccd_to_smiles: dict[str, str] = {}
    # Build with collision tracking; only unambiguous InChIKey → CCD entries
    # are kept in the final cache (see module docstring).
    inchikey_groups: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    # Silence RDKit's C++ logger (bypasses Python logging) during the cache build only — the wwPDB
    # CCD file has hundreds of organometallic/hypervalent entries we already drop via the mol-is-None
    # guard below. Restored after so genuine sanitize warnings on user SMILES still surface.
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    try:
        with open(CCD_DATABASE_PATH) as f:
            for line in f:
                fields = line.rstrip().split("\t")
                if len(fields) < 2:
                    continue
                ccd_code = fields[1]
                all_codes.add(ccd_code.upper())
                mol = Chem.MolFromSmiles(fields[0])
                if mol is None:
                    skipped += 1  # type: ignore[unreachable]
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if canonical not in smiles_to_ccd:
                    smiles_to_ccd[canonical] = ccd_code
                ccd_to_smiles[ccd_code.upper()] = canonical
                inchikey = Chem.MolToInchiKey(mol)  # type: ignore[no-untyped-call]
                # MolToInchiKey returns "" for entries InChI can't process
                # (single atoms, certain ions). Skip those.
                if inchikey:
                    inchikey_groups[inchikey].append(ccd_code)
    finally:
        RDLogger.EnableLog("rdApp.*")  # type: ignore[attr-defined]

    inchikey_to_ccd = {ikey: ccds[0] for ikey, ccds in inchikey_groups.items() if len(ccds) == 1}
    n_ambiguous = len(inchikey_groups) - len(inchikey_to_ccd)

    if skipped:
        logger.debug("CCD canonical cache: skipped %d RDKit-unparseable entries", skipped)
    if n_ambiguous:
        logger.debug(
            "CCD InChIKey cache: dropped %d ambiguous keys (multiple CCDs sharing the same InChIKey)",
            n_ambiguous,
        )

    return all_codes, smiles_to_ccd, ccd_to_smiles, inchikey_to_ccd


def _smi_sha256() -> str:
    """SHA256 hex digest of the source .smi file used to invalidate the pickled cache."""
    return hashlib.sha256(CCD_DATABASE_PATH.read_bytes()).hexdigest()


def _load_pickled_caches() -> tuple[set[str], dict[str, str], dict[str, str], dict[str, str]] | None:
    """Return the four caches from the prebuilt pickle, or None on miss/mismatch."""
    if not CCD_CACHE_PICKLE_PATH.exists():
        return None
    try:
        with CCD_CACHE_PICKLE_PATH.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 -- pickle is a repo-checked-in build artifact
    except Exception as exc:
        logger.warning("CCD cache pickle at %s failed to load (%s); rebuilding from .smi", CCD_CACHE_PICKLE_PATH, exc)
        return None
    if payload.get("version") != _CCD_CACHE_PICKLE_VERSION:
        logger.warning("CCD cache pickle schema mismatch; rebuilding from .smi")
        return None
    if payload.get("smi_sha256") != _smi_sha256():
        logger.warning("CCD cache pickle is stale (source .smi changed); rebuilding from .smi")
        return None
    return (
        set(payload["all_codes"]),
        payload["smiles_to_ccd"],
        payload["ccd_to_smiles"],
        payload["inchikey_to_ccd"],
    )


def _write_pickled_caches(
    all_codes: set[str],
    smiles_to_ccd: dict[str, str],
    ccd_to_smiles: dict[str, str],
    inchikey_to_ccd: dict[str, str],
) -> None:
    """Atomically write the four caches to ``CCD_CACHE_PICKLE_PATH``. Silently no-op on read-only filesystems."""
    import os

    import rdkit

    payload = {
        "version": _CCD_CACHE_PICKLE_VERSION,
        "smi_sha256": _smi_sha256(),
        "rdkit_version": rdkit.__version__,
        # Sort everything so the pickle bytes are stable across rebuilds on different machines.
        "all_codes": sorted(all_codes),
        "smiles_to_ccd": dict(sorted(smiles_to_ccd.items())),
        "ccd_to_smiles": dict(sorted(ccd_to_smiles.items())),
        "inchikey_to_ccd": dict(sorted(inchikey_to_ccd.items())),
    }
    tmp_path = CCD_CACHE_PICKLE_PATH.with_suffix(CCD_CACHE_PICKLE_PATH.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, CCD_CACHE_PICKLE_PATH)
    except OSError as exc:
        logger.debug(
            "Could not persist CCD cache pickle at %s (%s); continuing without it.", CCD_CACHE_PICKLE_PATH, exc
        )
        tmp_path.unlink(missing_ok=True)


def _ensure_caches() -> None:
    """Lazy-initialize all caches on first use; rebuild + persist if the pickle is missing or stale."""
    global _ALL_CCD_CODES, _RDKIT_CANONICAL_TO_CCD, _CCD_TO_RDKIT_CANONICAL, _INCHIKEY_TO_CCD
    if _ALL_CCD_CODES is not None:
        return
    cached = _load_pickled_caches()
    if cached is not None:
        _ALL_CCD_CODES, _RDKIT_CANONICAL_TO_CCD, _CCD_TO_RDKIT_CANONICAL, _INCHIKEY_TO_CCD = cached
        return
    logger.warning("Building CCD lookup tables from .smi (one-time, ~30s)...")
    _ALL_CCD_CODES, _RDKIT_CANONICAL_TO_CCD, _CCD_TO_RDKIT_CANONICAL, _INCHIKEY_TO_CCD = _build_caches()
    _write_pickled_caches(_ALL_CCD_CODES, _RDKIT_CANONICAL_TO_CCD, _CCD_TO_RDKIT_CANONICAL, _INCHIKEY_TO_CCD)


def _get_canonical_caches() -> tuple[dict[str, str], dict[str, str]]:
    """Return the bidirectional canonical caches (lazy-initialized).

    Returns:
        tuple[dict[str, str], dict[str, str]]: (smiles_to_ccd, ccd_to_smiles) caches.
    """
    _ensure_caches()
    assert _RDKIT_CANONICAL_TO_CCD is not None and _CCD_TO_RDKIT_CANONICAL is not None  # noqa: S101
    return _RDKIT_CANONICAL_TO_CCD, _CCD_TO_RDKIT_CANONICAL


def _get_inchikey_cache() -> dict[str, str]:
    """Return the InChIKey → CCD code cache (lazy-initialized, unambiguous only).

    Returns:
        dict[str, str]: InChIKey → CCD code mapping for entries with a unique InChIKey.
    """
    _ensure_caches()
    assert _INCHIKEY_TO_CCD is not None  # noqa: S101
    return _INCHIKEY_TO_CCD


def _get_all_ccd_codes_cache() -> set[str]:
    """Return the raw-file set of every CCD code (lazy-initialized).

    Returns:
        set[str]: All CCD codes in the database, regardless of RDKit parseability.
    """
    _ensure_caches()
    assert _ALL_CCD_CODES is not None  # noqa: S101
    return _ALL_CCD_CODES


# ============================================================================
# Public mapping functions
# ============================================================================
def map_smiles_to_ccd_code(smiles: str) -> str | None:
    """Map a SMILES string to its corresponding CCD code (offline-only).

    Two-tier lookup, both fully local:

    1. **Canonical-SMILES match** (primary): canonicalize the input via RDKit
       and look it up in the bundled CCD cache. Handles the common case
       cleanly; OpenEye-canonical entries in the wwPDB CCD file are
       re-canonicalized via RDKit at cache build time so the keys agree.

    2. **InChIKey match** (fallback): when (1) misses, compute the IUPAC-standard
       InChIKey of the input molecule and look it up. InChIKey normalizes
       more aggressively than canonical SMILES (handles tautomers, ionization,
       aromaticity-perception edge cases) and is the cross-tool standard used
       by ChEMBL, RCSB, PubChem, and PDBe for chemical identity. Only
       unambiguous matches are honored; ambiguous InChIKeys (mostly single-atom
       ions and tautomer/stereo variants) return None and let the caller
       decide what to do.

    Args:
        smiles (str): SMILES string representation of the molecule.

    Returns:
        str | None: CCD code if found in either tier, None otherwise.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # type: ignore[unreachable]

    # Tier 1: canonical SMILES match (primary, kept for backward-compat with
    # existing callers and slightly faster than InChIKey for the common case).
    canonical = Chem.MolToSmiles(mol, canonical=True)
    smiles_to_ccd, _ = _get_canonical_caches()
    if (result := smiles_to_ccd.get(canonical)) is not None:
        return result

    # Tier 2: InChIKey match (catches canonical-SMILES misses).
    inchikey = Chem.MolToInchiKey(mol)  # type: ignore[no-untyped-call]
    if inchikey:
        return _get_inchikey_cache().get(inchikey)

    return None


def map_ccd_code_to_smiles(ccd_code: str) -> str | None:
    """Map a CCD code to its RDKit-canonical SMILES string.

    Args:
        ccd_code (str): CCD code (e.g., ``"SEP"``, ``"TPO"``, ``"ATP"``).
            Case-insensitive.

    Returns:
        str | None: RDKit-canonical SMILES string if found, None otherwise.
    """
    _, ccd_to_smiles = _get_canonical_caches()
    return ccd_to_smiles.get(ccd_code.upper())


def count_heavy_atoms_for_ccd(ccd_code: str) -> int:
    """Count heavy (non-hydrogen) atoms for a CCD-coded component.

    Args:
        ccd_code (str): CCD code. Case-insensitive.

    Returns:
        int: Heavy-atom count of the component's RDKit-canonical structure.

    Raises:
        ValueError: If the CCD code is not found in the database.
    """
    from rdkit import Chem

    smiles = map_ccd_code_to_smiles(ccd_code)
    if smiles is None:
        raise ValueError(f"Unknown CCD code: {ccd_code!r}")
    mol = Chem.MolFromSmiles(smiles)
    return int(mol.GetNumHeavyAtoms())


def get_ccd_description(ccd_code: str) -> str | None:
    """Get the description/name for a CCD code.

    Args:
        ccd_code (str): Three-letter CCD code

    Returns:
        str | None: Description string if found, None otherwise

    Examples:
        >>> get_ccd_description("SEP")
        'PHOSPHOSERINE'

        >>> get_ccd_description("2MG")
        '2-METHYL-GUANOSINE-5'-MONOPHOSPHATE'
    """
    _ensure_ccd_database()

    ccd_code_upper = ccd_code.upper()

    with open(CCD_DATABASE_PATH) as f:
        for line in f:
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                continue
            if fields[1].upper() == ccd_code_upper:
                return fields[2]

    return None


def is_valid_ccd_code(ccd_code: str) -> bool:
    """Check if a CCD code exists in the database (O(1) with caching).

    Validity is checked against the raw set of all CCD codes in the wwPDB file,
    independent of whether RDKit can parse the entry's SMILES. This avoids
    rejecting codes whose SMILES happens to be unparseable.

    Args:
        ccd_code (str): CCD code to validate.

    Returns:
        bool: True if the code exists in the CCD database, False otherwise.
    """
    code_upper = ccd_code.upper()
    if code_upper in COMMON_MODIFICATIONS:
        return True
    return code_upper in _get_all_ccd_codes_cache()


def get_all_ccd_codes() -> set[str]:
    """Get all CCD codes from the database.

    Returns:
        set[str]: Set of all CCD codes in the database (raw, regardless of
            RDKit parseability).
    """
    return set(_get_all_ccd_codes_cache())


COMMON_MODIFICATIONS = {
    # Protein modifications (Post-translational modifications)
    "SEP",
    "TPO",
    "PTR",
    "HY3",
    "HYP",
    "P1L",
    "PCA",
    "MLY",
    "MLZ",
    "M3L",
    "ALY",
    "MEA",
    "CSD",
    "CSO",
    # RNA modifications
    "2MG",
    "5MC",
    "PSU",
    "7MG",
    "H2U",
    "M2G",
    "OMC",
    "OMG",
    "1MA",
    # DNA modifications
    "6OG",
    "6MA",
    "5CM",
    "8OG",
    "M5C",
}


# Path to CCD parent mapping file
CCD_PARENT_MAPPING_PATH = Path(__file__).parent / "ccd_maps" / "ccd_to_og.csv"

# Cache for the CCD parent mapping dataframe
_CCD_PARENT_DF_CACHE: pd.DataFrame | None = None


def _get_ccd_parent_dataframe() -> pd.DataFrame:
    """Load and cache the CCD parent mapping dataframe.

    Returns:
        pd.DataFrame: DataFrame with columns: ccd_code, parent_3letter, parent_1letter
    """
    import pandas as pd

    global _CCD_PARENT_DF_CACHE  # noqa: PLW0603 -- module-level cache

    if _CCD_PARENT_DF_CACHE is None:
        if not CCD_PARENT_MAPPING_PATH.exists():
            raise FileNotFoundError(f"CCD parent mapping file not found at {CCD_PARENT_MAPPING_PATH}")
        _CCD_PARENT_DF_CACHE = pd.read_csv(CCD_PARENT_MAPPING_PATH, comment="#")

    return _CCD_PARENT_DF_CACHE


def get_canonical_component(ccd_code: str) -> str | None:
    """Get the canonical (unmodified) form for a modified CCD component.

    This function maps CCD codes for modified components to their canonical
    single-letter code. Works for all biological polymers: proteins, DNA, and RNA.

    Use cases:
    - Converting modified amino acids to their parent for MSA generation
    - Converting modified nucleotides to their parent for sequence analysis
    - Structure prediction tools that need canonical sequences for modifications

    Supports:
    - Protein PTMs (post-translational modifications): SEP→S, TPO→T, PTR→Y, etc.
    - RNA modifications: 2MG→G, 5MC→C, PSU→U, etc.
    - DNA modifications: 6MA→A, 8OG→G, etc.

    Args:
        ccd_code (str): Three-letter CCD code for a modified component

    Returns:
        str | None: Single-letter code for the canonical (parent) amino acid or nucleotide,
            or None if the CCD code is not a known modified component

    Examples:
        >>> # Protein PTMs
        >>> get_canonical_component("SEP")
        'S'  # Phosphoserine -> Serine
        >>> get_canonical_component("TPO")
        'T'  # Phosphothreonine -> Threonine
        >>> get_canonical_component("MSE")
        'M'  # Selenomethionine -> Methionine

        >>> # RNA modifications
        >>> get_canonical_component("2MG")
        'G'  # 2'-O-methylguanosine -> Guanosine
        >>> get_canonical_component("PSU")
        'U'  # Pseudouridine -> Uridine

        >>> # DNA modifications
        >>> get_canonical_component("6MA")
        'A'  # N6-methyladenine -> Adenine

        >>> # Standard amino acid (not modified)
        >>> get_canonical_component("ALA")
        None

    Note:
        Mapping loaded from ccd_to_og.csv (1,301+ modified components from wwPDB CCD).
        The dataframe is cached in memory after first access for performance.
    """
    df = _get_ccd_parent_dataframe()
    result = df[df["ccd_code"] == ccd_code.upper()]

    if len(result) == 0:
        return None

    return result.iloc[0]["parent_1letter"]  # type: ignore[no-any-return]


def get_modifications_for_component(entity_type: str, canonical_letter: str) -> list[str]:
    """Get all CCD modification codes allowed for a given canonical component.

    This function returns all known modifications for a specific amino acid or
    nucleotide base in a given entity type (protein, DNA, or RNA).

    Args:
        entity_type (str): Type of biological polymer - "protein", "dna", or "rna"
        canonical_letter (str): Single-letter code for the canonical amino acid or base
            - For proteins: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
            - For DNA: A, T, C, G
            - For RNA: A, U, C, G

    Returns:
        list[str]: List of CCD codes for modifications of this component. Empty list if none exist.

    Examples:
        >>> # Protein modifications
        >>> get_modifications_for_component("protein", "S")
        ['SEP', ...]  # Phosphoserine and other serine modifications

        >>> get_modifications_for_component("protein", "T")
        ['TPO', ...]  # Phosphothreonine and other threonine modifications

        >>> # RNA modifications
        >>> get_modifications_for_component("rna", "G")
        ['2MG', '7MG', 'M2G', 'OMG', ...]  # Various guanosine modifications

        >>> # DNA modifications
        >>> get_modifications_for_component("dna", "A")
        ['6MA', ...]  # N6-methyladenine and other adenine modifications

        >>> # No known modifications
        >>> get_modifications_for_component("protein", "X")
        []

    Note:
        The entity_type is used to distinguish between DNA and RNA modifications,
        as they have different parent codes in the CCD (e.g., "DG" for DNA vs "G" for RNA).
        The dataframe is cached in memory after first access for performance.
    """
    entity_type = entity_type.lower()
    canonical_letter = canonical_letter.upper()

    # Validate entity_type
    if entity_type not in {"protein", "dna", "rna"}:
        raise ValueError(f"Invalid entity_type: {entity_type}. Must be 'protein', 'dna', or 'rna'")

    # Get cached dataframe
    df = _get_ccd_parent_dataframe()

    # Filter by canonical letter
    df = df[df["parent_1letter"] == canonical_letter]

    # Filter by entity type based on parent_3letter pattern
    if entity_type == "protein":
        # Protein: 3-letter codes (ALA, SER, etc.) that don't start with 'D'
        df = df[(df["parent_3letter"].str.len() == 3) & (~df["parent_3letter"].str.startswith("D"))]
    elif entity_type == "dna":
        # DNA: starts with 'D' (DA, DG, DC, DT)
        df = df[df["parent_3letter"].str.startswith("D")]
    elif entity_type == "rna":
        # RNA: single letter (A, C, G, U)
        df = df[df["parent_3letter"].str.len() == 1]

    return df["ccd_code"].tolist()  # type: ignore[no-any-return]
