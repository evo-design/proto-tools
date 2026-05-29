"""proto_tools/tools/structure_prediction/chai1/helpers.py.

Shared helpers for Chai-1 structure prediction. Provides utilities for
sequence hashing, MSA Parquet file writing, FASTA generation, and glycan-aware token counting.
"""

import hashlib
import re
from typing import Any

from proto_tools.entities.ligands import Fragment, count_heavy_atoms_for_ccd
from proto_tools.tools.structure_prediction.shared_data_models import Chain, count_structure_tokens


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
    pairing_keys: list[str] | None = None,
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
        pairing_keys (list[str] | None): Optional per-row pairing key strings.
            When supplied, rows whose pairing_key matches across chains are paired
            by chai_lab. The query row (index 0) should carry an empty pairing_key.
            When ``None``, all rows get an empty pairing_key (no cross-chain pairing).
    """
    import pandas as pd

    records = []
    for idx, seq in enumerate(aligned_sequences):
        comment = comments[idx] if comments else f"seq_{idx}"
        pairing_key = pairing_keys[idx] if pairing_keys else ""
        records.append(
            {
                "sequence": seq,
                "source_database": "query" if idx == 0 else source_database,
                "pairing_key": pairing_key,
                "comment": comment,
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(pqt_path, engine="pyarrow", index=False)


def complex_to_fasta(chains: list[Chain | Fragment]) -> str:
    """Convert a list of chains to FASTA format for Chai1.

    Args:
        chains (list[Chain | Fragment]): Biopolymer chains (``Chain``) and/or
            ligands (``Fragment``).

    Returns:
        str: FASTA-formatted string.
    """
    fasta_content = ""
    for i, chain in enumerate(chains):
        # SMILES is case-sensitive (lowercase letters denote aromatic atoms); only biopolymers get uppercased.
        body = chain.smiles if isinstance(chain, Fragment) else chain.sequence.upper()
        fasta_content += f">{chain.entity_type}|name={chain.entity_type}_{i + 1}\n"
        fasta_content += f"{body}\n"
    return fasta_content


def count_chai1_tokens(chains: list[Chain | Fragment]) -> int:
    """Token count for a Chai-1 complex: shared structure tokens plus glycan sugars.

    Glycans use Chai-1's string format (e.g. ``"MAN(6-1 FUC)(4-1 MAN)"``), each sugar
    contributing its heavy-atom count; every other chain is counted by
    :func:`count_structure_tokens`.
    """
    glycan_tokens = 0
    non_glycan: list[Chain | Fragment] = []
    for chain in chains:
        if not isinstance(chain, Fragment) and chain.entity_type == "glycan":
            glycan_tokens += _glycan_string_heavy_atom_count(chain.sequence)
        else:
            non_glycan.append(chain)
    return glycan_tokens + count_structure_tokens(non_glycan)


def _glycan_string_heavy_atom_count(glycan_string: str) -> int:
    """Sum heavy atoms across sugars in a Chai-1 glycan string.

    Mirrors chai-lab's parser at ``chai_lab/data/parsing/glycans.py``.

    Args:
        glycan_string (str): Chai-1 glycan string (e.g., ``"MAN(6-1 FUC)(4-1 MAN)"``).

    Returns:
        int: Total heavy-atom count across all sugars.

    Raises:
        ValueError: If the glycan string cannot be parsed.
    """
    total = 0
    i = 0
    s = glycan_string.strip()
    while i < len(s):
        c = s[i]
        if c in " ()":
            i += 1
            continue
        chunk = s[i : i + 3]
        if re.match(r"[1-6]-[1-6]", chunk):
            i += 3
            continue
        if re.match(r"[0-9A-Z]{3}", chunk):
            total += count_heavy_atoms_for_ccd(chunk)
            i += 3
            continue
        raise ValueError(f"Invalid glycan string: {glycan_string!r}")
    return total
