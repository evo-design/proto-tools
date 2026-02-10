"""
Sequence validation helpers for bio_programming.bio_tools.tools.

Extracted subset from language/core/sequence.py.
"""
from __future__ import annotations

import warnings
from typing import Optional, Set

# Valid characters for different sequence types
DNA_NUCLEOTIDES = "ACGT"
RNA_NUCLEOTIDES = "ACGU"
PROTEIN_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _return_invalid_chars(sequence: str, valid_chars: Set[str]) -> Set[str]:
    """
    Return the invalid characters in a sequence given a set of valid characters.

    Args:
        sequence: The sequence string to validate.
        valid_chars: The set of valid characters.

    Returns:
        The set of invalid characters.
    """
    invalid_chars = set(sequence) - valid_chars
    return invalid_chars


def return_invalid_dna_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Helper function that returns the invalid characters in a DNA sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default DNA characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = DNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_rna_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Helper function that returns the invalid characters in a RNA sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default RNA characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = RNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_nucleotide_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Helper function that returns the invalid characters in a nucleotide sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default nucleotide characters.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = DNA_NUCLEOTIDES + RNA_NUCLEOTIDES + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def return_invalid_protein_chars(
    sequence: str,
    additional_valid_chars: Optional[str] = None,
) -> Set[str]:
    """
    Return the invalid characters in a protein sequence.

    Args:
        sequence (str): The sequence string to validate.
        additional_valid_chars (Optional[str]): Additional valid characters to add to the default protein amino acids.

    Returns:
        Set[str]: The set of invalid characters.
    """
    if additional_valid_chars is None:
        additional_valid_chars = ""

    valid_chars = PROTEIN_AMINO_ACIDS + additional_valid_chars
    return _return_invalid_chars(sequence, set(valid_chars))


def validate_smiles(smiles: str, verbose: bool = True) -> bool:
    """
    Validate SMILES string using RDKit if available.

    Args:
        smiles: The SMILES string to validate.
        verbose: Print warnings.

    Returns:
        True if valid SMILES, False if invalid or RDKit unavailable.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if verbose:
                warnings.warn(
                    f"RDKit could not parse SMILES: '{smiles}'. "
                    "This may not be a valid molecule."
                )
            return False
        return True
    except ImportError:
        if verbose:
            warnings.warn("RDKit not installed. Cannot validate SMILES.")
        return False


def detect_sequence_type(sequence: str) -> str:
    """
    Attempts to determine the type of a sequence based on the characters it contains.
    Starts with more specific sequence types (less characters allowed) and works
    its way down to the least specific. Returns "unknown" if the sequence type
    cannot be determined.

    Note that there are ambiguous cases (e.g., "CCCCCC" could be DNA, RNA, protein, or
    ligand SMILES). Priority is: DNA, RNA, protein, ligand.

    Args:
        sequence (str): The sequence string to detect the type of.

    Returns:
       string: The type of the sequence ("dna", "rna", "protein", "ligand", or "unknown").
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
    if validate_smiles(sequence, verbose=False):
        return "ligand"

    # Otherwise, return unknown
    return "unknown"
