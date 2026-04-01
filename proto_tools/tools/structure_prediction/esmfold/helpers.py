"""
proto_tools/tools/structure_prediction/esmfold/helpers.py

Shared helpers for ESMFold structure prediction. Provides utilities for
batching complexes and relabeling chains in PDB output.
"""

from __future__ import annotations

import string
from typing import Any

CHAIN_IDS = list(string.ascii_uppercase)


def split_into_safe_batches(
    complexes: list[dict[str, Any]], max_residues: int
) -> list[list[dict[str, Any]]]:
    """
    Split complexes into sub-batches respecting GPU memory limits.

    Args:
        complexes (list[dict[str, Any]]): List of complex dicts, each with a "total_residues" key
        max_residues (int): Maximum total residues allowed per sub-batch

    Returns:
        list[list[dict[str, Any]]]: List of sub-batches, where each sub-batch is a list of complexes
    """
    batches = []
    current_batch = []
    current_residues = 0

    for item in complexes:
        item_residues = item["total_residues"]

        if item_residues > max_residues:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_residues = 0
            batches.append([item])
            continue

        if current_residues + item_residues > max_residues:
            batches.append(current_batch)
            current_batch = []
            current_residues = 0

        current_batch.append(item)
        current_residues += item_residues

    if current_batch:
        batches.append(current_batch)

    return batches


def relabel_chains(pdb_str: str, chain_lengths: list[int]) -> str:
    """
    Relabel single-chain PDB output into multiple chains (A, B, C, ...).

    ESMFold predicts multi-chain complexes by linking chains together, producing
    a single-chain PDB. This function splits the single chain back into separate
    chains with standard alphabetic labels (A, B, C, etc.).

    Args:
        pdb_str (str): PDB file content as a string (assumed to be a single chain)
        chain_lengths (list[int]): List of residue counts for each desired chain.
                      Total must match the number of residues in pdb_str.

    Returns:
        str: PDB content with chains relabeled and written back to string format
    """
    import io

    from Bio import PDB

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", io.StringIO(pdb_str))
    model = structure[0]

    original_chain = list(model.get_chains())[0]
    all_residues = list(original_chain.get_residues())

    new_chains = []
    start = 0

    for idx, length in enumerate(chain_lengths):
        new_chain = PDB.Chain.Chain(CHAIN_IDS[idx])
        for residue in all_residues[start : start + length]:
            new_chain.add(residue)
        new_chains.append(new_chain)
        start += length

    model.detach_child(original_chain.id)
    for chain in new_chains:
        model.add(chain)

    output = io.StringIO()
    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(output)

    return output.getvalue()
