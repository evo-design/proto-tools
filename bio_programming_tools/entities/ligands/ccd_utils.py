"""
ccd_utils.py

Utilities for working with the wwPDB Chemical Component Dictionary (CCD).

The CCD database contains information about all chemical components (ligands,
modified residues, etc.) found in PDB structures. This module provides functions
to map between SMILES strings and CCD codes, and to validate modification codes.

Data source: https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

CCD_DATABASE_PATH = Path(__file__).parent / "ccd_maps" / "Components-smiles-stereo-oe.smi"


def _ensure_ccd_database() -> None:
    """Ensure the CCD database exists, downloading if necessary."""
    if not CCD_DATABASE_PATH.exists():
        logger.error(f"CCD database not found at {CCD_DATABASE_PATH}")
        raise FileNotFoundError(f"CCD database not found at {CCD_DATABASE_PATH}")


def map_smiles_to_ccd_code(smiles: str, use_name_fallback: bool = True) -> Optional[str]:
    """Map a SMILES string to its corresponding CCD code.

    This function searches the CCD database for an exact SMILES match and returns
    the corresponding three-letter CCD code. If no exact match is found and
    use_name_fallback is True, it attempts to find a match by looking up the
    molecule name via PubChem and matching it against CCD descriptions.

    Args:
        smiles: SMILES string representation of the molecule
        use_name_fallback: If True, attempt name-based lookup when exact match fails

    Returns:
        CCD code if found, None otherwise

    Examples:
        >>> map_smiles_to_ccd_code("CNC[C@@H](c1ccc(c(c1)O)O)O")
        'ALE'  # epinephrine

        >>> map_smiles_to_ccd_code("CC(C)C")
        'IBU'  # isobutane

    Note:
        The database file is formatted as a tab-delimited file with 3 columns:
        SMILES, CCD code, and description. If the database doesn't exist locally,
        it will be automatically downloaded.

        The name-based fallback uses PubChem to get the molecule name and searches
        CCD descriptions. It only returns a match if exactly one description contains
        the name as a substring.
    """
    _ensure_ccd_database()

    # Try exact SMILES match first
    with open(CCD_DATABASE_PATH, 'r') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            if len(fields) < 2:
                continue
            if fields[0] == smiles:
                return fields[1]

    # If exact match failed and fallback is enabled, try name-based lookup
    if use_name_fallback:
        return _map_smiles_to_ccd_via_name(smiles)

    return None


def _map_smiles_to_ccd_via_name(smiles: str) -> Optional[str]:
    """Attempt to map SMILES to CCD code via molecule name lookup.

    This is a fallback method when exact SMILES matching fails. It:
    1. Gets the molecule name from PubChem
    2. Searches CCD descriptions for the name (case-insensitive substring match)
    3. Returns the CCD code if exactly one match is found
    4. Logs all matches if multiple are found and returns None

    Args:
        smiles: SMILES string representation of the molecule

    Returns:
        CCD code if exactly one match found, None otherwise
    """
    try:
        # Import here to avoid circular dependency and only when needed
        from bio_programming_tools.entities.ligands.utils import get_name_from_smiles

        # Get molecule name from PubChem
        name = get_name_from_smiles(smiles)

        if not name or name == "Unknown":
            return None

        # Search for matching descriptions in CCD database
        matches = []
        with open(CCD_DATABASE_PATH, "r") as f:
            for line in f:
                fields = line.rstrip().split("\t")
                if len(fields) < 3:
                    continue

                ccd_code = fields[1]
                description = fields[2]

                # Case-insensitive substring match
                if name.lower() in description.lower():
                    matches.append(
                        {"ccd_code": ccd_code, "description": description, "smiles": fields[0]}
                    )

        # If exactly one match, return it
        if len(matches) == 1:
            logger.debug(
                f"Found CCD code via name lookup: {smiles} -> '{name}' -> {matches[0]['ccd_code']}"
            )
            return matches[0]["ccd_code"]

        # If multiple matches, log them all and return None
        elif len(matches) > 1:
            logger.debug(
                f"Multiple CCD matches found for SMILES '{smiles}' with name '{name}':"
            )
            for match in matches:
                logger.debug(
                    f"  - CCD: {match['ccd_code']}, Description: {match['description']}"
                )
            return None

        # No matches found
        return None

    except Exception as e:
        # If name lookup fails for any reason, silently return None
        logger.debug(f"Name-based CCD lookup failed for '{smiles}': {e}")
        return None


def map_ccd_code_to_smiles(ccd_code: str) -> Optional[str]:
    """Map a CCD code to its corresponding SMILES string.

    This function searches the CCD database for a CCD code and returns the
    corresponding SMILES representation.

    Args:
        ccd_code: Three-letter CCD code (e.g., "SEP", "TPO", "ALE")

    Returns:
        SMILES string if found, None otherwise

    Examples:
        >>> map_ccd_code_to_smiles("SEP")
        'C([C@@H](C(=O)O)N)OP(=O)(O)O'  # phosphoserine

        >>> map_ccd_code_to_smiles("TPO")
        'CC([C@@H](C(=O)O)N)OP(=O)(O)O'  # phosphothreonine

    Note:
        CCD codes are case-insensitive. If the database doesn't exist locally,
        it will be automatically downloaded.
    """
    _ensure_ccd_database()

    ccd_code_upper = ccd_code.upper()

    with open(CCD_DATABASE_PATH, 'r') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            if len(fields) < 2:
                continue
            if fields[1].upper() == ccd_code_upper:
                return fields[0]

    return None


def get_ccd_description(ccd_code: str) -> Optional[str]:
    """Get the description/name for a CCD code.

    Args:
        ccd_code: Three-letter CCD code

    Returns:
        Description string if found, None otherwise

    Examples:
        >>> get_ccd_description("SEP")
        'PHOSPHOSERINE'

        >>> get_ccd_description("2MG")
        '2-METHYL-GUANOSINE-5'-MONOPHOSPHATE'
    """
    _ensure_ccd_database()

    ccd_code_upper = ccd_code.upper()

    with open(CCD_DATABASE_PATH, 'r') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            if len(fields) < 3:
                continue
            if fields[1].upper() == ccd_code_upper:
                return fields[2]

    return None


def is_valid_ccd_code(ccd_code: str) -> bool:
    """Check if a CCD code exists in the database (O(1) with caching).

    This function uses an in-memory cache to avoid repeated file scans.
    The cache is populated on first access to a non-common modification code.

    Args:
        ccd_code: Three-letter CCD code to validate

    Returns:
        True if the code exists in the CCD database, False otherwise

    Examples:
        >>> is_valid_ccd_code("SEP")
        True

        >>> is_valid_ccd_code("XYZ")
        False
    """
    code_upper = ccd_code.upper()

    # Fast path: check common modifications first (O(1))
    if code_upper in COMMON_MODIFICATIONS:
        return True

    # Slow path (first call only): load full database into memory cache
    return code_upper in _get_ccd_code_cache()


def get_all_ccd_codes() -> Set[str]:
    """Get all CCD codes from the database.

    Warning: This loads all codes into memory. Use sparingly.

    Returns:
        Set of all CCD codes in the database
    """
    _ensure_ccd_database()

    codes = set()
    with open(CCD_DATABASE_PATH, 'r') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            if len(fields) >= 2:
                codes.add(fields[1])

    return codes


# Cache for CCD codes - populated on first access
_CCD_CODE_CACHE: Optional[Set[str]] = None


def _get_ccd_code_cache() -> Set[str]:
    """Lazy-load all CCD codes into memory on first use.

    This function caches the entire set of CCD codes from the database
    to avoid repeated file scans. The cache is populated on the first
    call and reused for all subsequent calls.

    Returns:
        Set of all CCD codes in the database
    """
    global _CCD_CODE_CACHE
    if _CCD_CODE_CACHE is None:
        _CCD_CODE_CACHE = get_all_ccd_codes()
    return _CCD_CODE_CACHE


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
_CCD_PARENT_DF_CACHE: Optional[pd.DataFrame] = None


def _get_ccd_parent_dataframe() -> pd.DataFrame:
    """Load and cache the CCD parent mapping dataframe.

    Returns:
        DataFrame with columns: ccd_code, parent_3letter, parent_1letter
    """
    global _CCD_PARENT_DF_CACHE

    if _CCD_PARENT_DF_CACHE is None:
        if not CCD_PARENT_MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"CCD parent mapping file not found at {CCD_PARENT_MAPPING_PATH}"
            )
        _CCD_PARENT_DF_CACHE = pd.read_csv(CCD_PARENT_MAPPING_PATH, comment='#')

    return _CCD_PARENT_DF_CACHE


def get_canonical_component(ccd_code: str) -> Optional[str]:
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
        ccd_code: Three-letter CCD code for a modified component

    Returns:
        Single-letter code for the canonical (parent) amino acid or nucleotide,
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
    result = df[df['ccd_code'] == ccd_code.upper()]

    if len(result) == 0:
        return None

    return result.iloc[0]['parent_1letter']


def get_modifications_for_component(entity_type: str, canonical_letter: str) -> list[str]:
    """Get all CCD modification codes allowed for a given canonical component.

    This function returns all known modifications for a specific amino acid or
    nucleotide base in a given entity type (protein, DNA, or RNA).

    Args:
        entity_type: Type of biological polymer - "protein", "dna", or "rna"
        canonical_letter: Single-letter code for the canonical amino acid or base
            - For proteins: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
            - For DNA: A, T, C, G
            - For RNA: A, U, C, G

    Returns:
        List of CCD codes for modifications of this component. Empty list if none exist.

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
        raise ValueError(
            f"Invalid entity_type: {entity_type}. Must be 'protein', 'dna', or 'rna'"
        )

    # Get cached dataframe
    df = _get_ccd_parent_dataframe()

    # Filter by canonical letter
    df = df[df['parent_1letter'] == canonical_letter]

    # Filter by entity type based on parent_3letter pattern
    if entity_type == "protein":
        # Protein: 3-letter codes (ALA, SER, etc.) that don't start with 'D'
        df = df[(df['parent_3letter'].str.len() == 3) & (~df['parent_3letter'].str.startswith('D'))]
    elif entity_type == "dna":
        # DNA: starts with 'D' (DA, DG, DC, DT)
        df = df[df['parent_3letter'].str.startswith('D')]
    elif entity_type == "rna":
        # RNA: single letter (A, C, G, U)
        df = df[df['parent_3letter'].str.len() == 1]

    return df['ccd_code'].tolist()
