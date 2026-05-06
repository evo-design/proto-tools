"""proto_tools/utils/sequence.py.

Sequence validation, detection, and ID resolution utilities.
"""

import logging
from typing import Literal, get_args

# ============================================================================
# Sequence Constants
# ============================================================================

# Valid characters for different sequence types
DNA_NUCLEOTIDES = "ACGT"
RNA_NUCLEOTIDES = "ACGU"

# Source of truth for the 20 canonical amino acids.
# ``PROTEIN_AMINO_ACIDS`` (str) and standalone-helpers' ``AMINO_ACIDS_LIST``
# (list[str]) derive from this Literal so all three stay in sync.
AminoAcid = Literal["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
PROTEIN_AMINO_ACIDS = "".join(get_args(AminoAcid))


_PROTEIN_AA_INDEX = {aa: i for i, aa in enumerate(PROTEIN_AMINO_ACIDS)}


def one_hot_protein_logits(sequence: str, *, sharpness: float = 20.0) -> list[list[float]]:
    """One-hot encode a protein sequence as logits (L x 20) in ``PROTEIN_AMINO_ACIDS`` order.

    Default ``sharpness=20.0`` saturates a downstream softmax to ≈ one-hot; use a milder
    value (e.g. ``2.0``) for a biased-but-not-saturated seed.

    Args:
        sequence (str): Protein sequence; each character must be in ``PROTEIN_AMINO_ACIDS``.
        sharpness (float): Value placed on the one-hot column; all other columns are 0.

    Returns:
        list[list[float]]: Logits matrix with shape ``(len(sequence), 20)``.
    """
    n = len(PROTEIN_AMINO_ACIDS)
    rows: list[list[float]] = []
    for aa in sequence:
        row = [0.0] * n
        row[_PROTEIN_AA_INDEX[aa]] = sharpness
        rows.append(row)
    return rows


def calculate_gc_content(sequence: str) -> float:
    """Calculate the GC content percentage of a DNA/RNA sequence.

    Args:
        sequence (str): DNA or RNA sequence string.

    Returns:
        float: GC content as a percentage (0-100).
    """
    if not sequence:
        return 0.0

    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count("G") + sequence_upper.count("C")
    return 100.0 * gc_count / len(sequence)


def resolve_sequence_ids(sequences: list[str], ids: list[str] | None) -> list[str]:
    """Resolve sequence identifiers, using provided IDs or generating defaults.

    Args:
        sequences (list[str]): List of sequences to generate IDs for.
        ids (list[str] | None): Optional list of user-provided sequence identifiers.

    Returns:
        list[str]: List of sequence identifiers (provided IDs or seq_0, seq_1, ...).

    Raises:
        ValueError: If ids length doesn't match sequences length.
    """
    if ids is not None:
        if len(ids) != len(sequences):
            raise ValueError(
                f"resolve_sequence_ids: ids length ({len(ids)}) != sequences length ({len(sequences)}); "
                f"pass ids=None to auto-generate seq_0, seq_1, ..."
            )
        return ids
    return [f"seq_{i}" for i in range(len(sequences))]


def validate_positions_list(
    positions: list[int],
    *,
    label: str = "positions",
    logger_obj: logging.Logger | None = None,
) -> list[int]:
    """Validate a 1-indexed positions list and return it deduped.

    Args:
        positions (list[int]): Candidate 1-indexed positions.
        label (str): Prefix for error / warning messages, typically the field name.
        logger_obj (logging.Logger | None): When provided, dedup actions log a
            warning here. ``None`` suppresses the warning.

    Returns:
        list[int]: Deduped, order-preserving copy of ``positions``.

    Raises:
        ValueError: If ``positions`` is empty or contains a value ``< 1``.
    """
    if not positions:
        raise ValueError(f"{label}: position list cannot be empty")
    # bool is a subclass of int in Python, so [True, False] would otherwise
    # slip through and get reported as "got invalid [False]" — reject up front.
    bool_values = [p for p in positions if isinstance(p, bool)]
    if bool_values:
        raise ValueError(
            f"{label}: positions must be int, not bool; got {bool_values}",
        )
    invalid = [p for p in positions if p < 1]
    if invalid:
        raise ValueError(
            f"{label}: positions must be >= 1 (1-indexed); got invalid {invalid}",
        )
    unique = list(dict.fromkeys(positions))
    if logger_obj is not None and len(unique) < len(positions):
        logger_obj.warning(
            "%s: dropped %d duplicate position(s) (%d -> %d unique).",
            label,
            len(positions) - len(unique),
            len(positions),
            len(unique),
        )
    return unique


# ============================================================================
# Sequence Validation
# ============================================================================


def _return_invalid_chars(sequence: str, valid_chars: set[str]) -> set[str]:
    """Return the invalid characters in a sequence given a set of valid characters.

    Args:
        sequence (str): The sequence string to validate.
        valid_chars (set[str]): The set of valid characters.

    Returns:
        set[str]: The set of invalid characters.
    """
    return set(sequence) - valid_chars


def return_invalid_dna_chars(
    sequence: str,
    additional_valid_chars: str | None = None,
) -> set[str]:
    """Helper function that returns the invalid characters in a DNA sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default DNA characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = DNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_rna_chars(
    sequence: str,
    additional_valid_chars: str | None = None,
) -> set[str]:
    """Helper function that returns the invalid characters in a RNA sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default RNA characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = RNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_nucleotide_chars(
    sequence: str,
    additional_valid_chars: str | None = None,
) -> set[str]:
    """Helper function that returns the invalid characters in a nucleotide sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default nucleotide characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = DNA_NUCLEOTIDES + RNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_protein_chars(
    sequence: str,
    additional_valid_chars: str | None = None,
) -> set[str]:
    """Return the invalid characters in a protein sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (str | None): Additional valid characters to add to the default protein amino acids.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = PROTEIN_AMINO_ACIDS + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def detect_sequence_type(sequence: str) -> str:
    """Attempts to determine the type of a sequence based on the characters it contains.

    Starts with more specific sequence types (less characters allowed) and works
    its way down to the least specific. Returns "unknown" if the sequence type
    cannot be determined.

    Note that there are ambiguous cases (e.g., "CCCCCC" could be DNA, RNA, protein, or
    ligand SMILES). Priority is: DNA, RNA, protein, ligand.

    Args:
        sequence (str): The sequence string to detect the type of.

    Returns:
       str: The type of the sequence ("dna", "rna", "protein", "ligand", or "unknown").
    """
    # DNA
    invalid_chars = return_invalid_dna_chars(sequence, additional_valid_chars="N")
    if not invalid_chars:
        return "dna"

    # RNA
    invalid_chars = return_invalid_rna_chars(sequence, additional_valid_chars="TN")
    if not invalid_chars:
        return "rna"

    # Protein
    invalid_chars = return_invalid_protein_chars(sequence, additional_valid_chars="X*")
    if not invalid_chars:
        return "protein"

    # Ligand/SMILES
    from proto_tools.utils.chemistry import validate_smiles

    if validate_smiles(sequence, verbose=False):
        return "ligand"

    # Otherwise, return unknown
    return "unknown"
