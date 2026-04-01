"""proto_tools/tools/structure_prediction/chai1/helpers.py.

Shared helpers for Chai-1 structure prediction. Provides utilities for
sequence hashing, MSA Parquet file writing, and FASTA generation.
"""

from __future__ import annotations

import hashlib
from typing import Any


def hash_sequence(seq: str) -> str:
    """Compute SHA-256 hash of a sequence.

    This matches the implementation in chai_lab.data.parsing.msas.aligned_pqt.hash_sequence,
    which is used to generate MSA filenames that Chai1's run_inference expects.

    Args:
        seq (str): Protein sequence string

    Returns:
        str: Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(seq.encode()).hexdigest()


def write_msa_pqt(
    aligned_sequences: list[Any],
    pqt_path: str,
    source_database: str = "uniref90",
    comments: list[Any] | None = None,
) -> None:
    """Write aligned sequences as Chai1-format Parquet (.aligned.pqt).

    Columns: sequence, source_database, pairing_key, comment.
    Query sequence (index 0) gets source_database="query".

    Args:
        aligned_sequences (list[Any]): List of aligned sequence strings. The first
            sequence is treated as the query.
        pqt_path (str): Path where the .pqt file will be written.
        source_database (str): Name of the source database for non-query sequences
            (default: "uniref90"). Valid values: "query", "uniref90", "uniprot",
            "bfd_uniclust", "mgnify".
        comments (list[Any] | None): Optional list of comment strings (e.g., sequence IDs from
            the original MSA). If None, uses synthetic ``seq_0``, ``seq_1``, etc.
    """
    import pandas as pd

    records = []
    for idx, seq in enumerate(aligned_sequences):
        comment = comments[idx] if comments else f"seq_{idx}"
        records.append(
            {
                "sequence": seq,
                "source_database": "query" if idx == 0 else source_database,
                "pairing_key": "",
                "comment": comment,
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(pqt_path, engine="pyarrow", index=False)


def complex_to_fasta(chains: list[dict[str, Any]]) -> str:
    """Convert a list of chain dicts to FASTA format for Chai1.

    Args:
        chains (list[dict[str, Any]]): List of chain dicts, each with 'entity_type' and 'sequence' keys.

    Returns:
        str: FASTA-formatted string
    """
    fasta_content = ""
    for i, chain in enumerate(chains):
        e_type = chain.get("entity_type", "protein")
        seq = chain.get("sequence", "")
        fasta_content += f">{e_type}|name={e_type}_{i + 1}\n"
        fasta_content += f"{seq.upper()}\n"
    return fasta_content
