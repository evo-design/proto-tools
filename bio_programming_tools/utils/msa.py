"""bio_programming_tools/utils/msa.py

Shared MSA utilities for structure prediction tools."""
from __future__ import annotations


def extract_msa_sequences(msa, query_index: int = 0) -> tuple[list[str], list[str]]:
    """Extract aligned sequences and IDs from an MSA, with query swapped to front.

    Args:
        msa: MSA object with ``num_sequences`` and ``iter_with_ids()``
        query_index (int): Index of the sequence to place first (default: 0)

    Returns:
        tuple[list[str], list[str]]: Tuple of (sequences, seq_ids), both with query at index 0
    """
    if query_index < 0 or query_index >= msa.num_sequences:
        raise IndexError(
            f"Query index {query_index} out of range [0, {msa.num_sequences})"
        )
    seq_ids = []
    sequences = []
    for seq_id, seq in msa.iter_with_ids():
        seq_ids.append(seq_id)
        sequences.append(seq)
    if query_index != 0:
        sequences[0], sequences[query_index] = sequences[query_index], sequences[0]
        seq_ids[0], seq_ids[query_index] = seq_ids[query_index], seq_ids[0]
    return sequences, seq_ids
