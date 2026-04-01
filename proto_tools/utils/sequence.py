"""proto_tools/utils/sequence.py.

Sequence validation, detection, and ID resolution utilities.
"""

from __future__ import annotations

# ============================================================================
# Sequence Constants
# ============================================================================

# Valid characters for different sequence types
DNA_NUCLEOTIDES = "ACGT"
RNA_NUCLEOTIDES = "ACGU"
PROTEIN_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


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
            raise ValueError(f"sequence_ids length ({len(ids)}) must match sequences length ({len(sequences)})")
        return ids
    return [f"seq_{i}" for i in range(len(sequences))]


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
