"""proto_tools/tools/structure_prediction/boltz2/helpers.py.

Shared helpers for Boltz2 structure prediction. Provides utilities for
MSA CSV file writing and YAML input generation.
"""

from __future__ import annotations

import csv
from typing import Any

from proto_tools.tools.structure_prediction.shared_data_models import CHAIN_IDS


def write_msa_csv(aligned_sequences: list[Any], csv_path: str) -> None:
    """Write aligned sequences as Boltz2-format CSV (sequence + key columns).

    Query sequence must be first (key=0). Boltz uses the key column
    for cross-chain MSA pairing.

    Args:
        aligned_sequences (list[Any]): List of aligned sequence strings. The first
            sequence is treated as the query.
        csv_path (str): Path where the CSV file will be written.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "key"])
        for idx, seq in enumerate(aligned_sequences):
            writer.writerow([seq, idx])


def complex_to_yaml(
    chains: list[dict[str, Any]],
    chain_msa_paths: dict[str, Any] | None = None,
) -> str:
    """Convert a list of chain dicts to Boltz2 YAML input format.

    Args:
        chains (list[dict[str, Any]]): List of chain dicts, each with 'entity_type' and 'sequence' keys.
        chain_msa_paths (dict[str, Any] | None): Optional dict mapping chain IDs (A, B, C, ...) to
            MSA CSV file paths. Protein chains without a path get msa="empty"
            (single-sequence mode). If None, all protein chains get msa="empty".

    Returns:
        str: YAML formatted string for Boltz2 input
    """
    import yaml

    yaml_entries = []

    for i, chain in enumerate(chains):
        e_type = chain.get("entity_type", "protein")
        seq = chain.get("sequence", "")
        entry = {"id": CHAIN_IDS[i]}

        if e_type in ["protein", "dna", "rna"]:
            entry["sequence"] = seq
        elif e_type == "ligand":
            entry["smiles"] = seq

        # Protein chains: use MSA path if available, otherwise single-sequence
        if e_type == "protein":
            chain_id = CHAIN_IDS[i]
            if chain_msa_paths and chain_id in chain_msa_paths:
                entry["msa"] = chain_msa_paths[chain_id]
            else:
                entry["msa"] = "empty"

        yaml_entries.append({e_type: entry})

    return str(
        yaml.dump(
            {"sequences": yaml_entries, "predict": {"structure": {"enabled": True}}},
            sort_keys=False,
            default_flow_style=False,
        )
    )
