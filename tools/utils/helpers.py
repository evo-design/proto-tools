"""
Minimal helper utilities for bio_programming.bio_tools.tools.
"""
from __future__ import annotations

from typing import List, Optional


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate the GC content percentage of a DNA/RNA sequence.

    Args:
        sequence: DNA or RNA sequence string.

    Returns:
        GC content as a percentage (0-100).
    """
    if not sequence:
        return 0.0

    sequence_upper = sequence.upper()
    gc_count = sequence_upper.count("G") + sequence_upper.count("C")
    return 100.0 * gc_count / len(sequence)


def resolve_sequence_ids(sequences: List[str], ids: Optional[List[str]]) -> List[str]:
    """Resolve sequence identifiers, using provided IDs or generating defaults.

    Args:
        sequences: List of sequences to generate IDs for.
        ids: Optional list of user-provided sequence identifiers.

    Returns:
        List of sequence identifiers (provided IDs or seq_0, seq_1, ...).

    Raises:
        ValueError: If ids length doesn't match sequences length.
    """
    if ids is not None:
        if len(ids) != len(sequences):
            raise ValueError(
                f"sequence_ids length ({len(ids)}) must match sequences length ({len(sequences)})"
            )
        return ids
    return [f"seq_{i}" for i in range(len(sequences))]
