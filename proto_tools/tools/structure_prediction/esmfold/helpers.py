"""proto_tools/tools/structure_prediction/esmfold/helpers.py.

Shared helpers for ESMFold structure prediction. Provides utilities for
batching complexes and relabeling chains in PDB output.
"""

import io
from typing import Any

from proto_tools.entities.complex import chain_label


def split_into_safe_batches(complexes: list[dict[str, Any]], max_residues: int) -> list[list[dict[str, Any]]]:
    """Split complexes into sub-batches respecting GPU memory limits.

    Args:
        complexes (list[dict[str, Any]]): List of complex dicts, each with a "total_residues" key
        max_residues (int): Maximum total residues allowed per sub-batch

    Returns:
        list[list[dict[str, Any]]]: List of sub-batches, where each sub-batch is a list of complexes
    """
    batches = []
    current_batch: list[dict[str, Any]] = []
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


def relabel_chains(pdb_str: str, chain_lengths: list[int], chain_ids: list[str] | None = None) -> str:
    """Relabel single-chain PDB output into multiple chains (A, B, C, ...).

    ESMFold predicts multi-chain complexes by linking chains together, producing
    a single-chain PDB. This function splits the single chain back into separate
    chains with standard alphabetic labels (A, B, C, etc.).

    Args:
        pdb_str (str): PDB file content as a string (assumed to be a single chain)
        chain_lengths (list[int]): List of residue counts for each desired chain.
                      Total must match the number of residues in pdb_str.
        chain_ids (list[str] | None): Per-chain labels (see ``resolve_chain_ids``)
                      so explicit input IDs are preserved. ``None`` falls back to
                      positional ``chain_label(idx)``. PDB is single-character, so
                      multi-character IDs are not supported here.

    Returns:
        str: PDB content with chains relabeled and written back to string format
    """
    if chain_ids is not None:
        multi_char = [cid for cid in chain_ids if len(cid) != 1]
        if multi_char:
            raise ValueError(f"PDB format requires single-character chain IDs; got {multi_char}.")

    from Bio import PDB

    parser = PDB.PDBParser(QUIET=True)  # type: ignore[attr-defined, no-untyped-call]
    structure = parser.get_structure("structure", io.StringIO(pdb_str))  # type: ignore[no-untyped-call]
    model = structure[0]

    original_chain = next(iter(model.get_chains()))
    all_residues = list(original_chain.get_residues())

    new_chains = []
    start = 0

    for idx, length in enumerate(chain_lengths):
        label = chain_ids[idx] if chain_ids is not None else chain_label(idx)
        new_chain = PDB.Chain.Chain(label)  # type: ignore[no-untyped-call]
        for residue in all_residues[start : start + length]:
            new_chain.add(residue)
        new_chains.append(new_chain)
        start += length

    model.detach_child(original_chain.id)
    for chain in new_chains:
        model.add(chain)

    output = io.StringIO()
    pdb_io = PDB.PDBIO()  # type: ignore[attr-defined, no-untyped-call]
    pdb_io.set_structure(structure)  # type: ignore[no-untyped-call]
    pdb_io.save(output)  # type: ignore[no-untyped-call]

    return output.getvalue()
