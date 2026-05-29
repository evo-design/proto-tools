"""proto_tools/tools/structure_prediction/boltz2/helpers.py.

Shared helpers for Boltz2 structure prediction. Provides utilities for
MSA CSV file writing and YAML input generation.
"""

import csv
import hashlib
import os
import warnings
from logging import getLogger
from typing import Any

from proto_tools.entities.complex import chain_label
from proto_tools.entities.ligands import Fragment
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    resolve_chain_ids,
    unwrap_complex_msas,
)
from proto_tools.utils import extract_msa_sequences

logger = getLogger(__name__)


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


def build_chain_msa_paths(
    sp_complex: Complex,
    complex_msas: "ComplexMSAs | None",
    temp_dir: str,
    verbose: int = 0,
) -> dict[str, str]:
    """Map each protein chain ID to its MSA CSV, one sha256-named file per unique sequence.

    boltz requires identical chains to share a single MSA file. The ``key=<row_idx>``
    column written by ``write_msa_csv`` lets Boltz pair rows across chains by
    position; when rows are taxonomy-aligned by upstream preprocess, that
    pairing carries through automatically.
    """
    per_chain_msas, _is_paired = unwrap_complex_msas(complex_msas)
    chain_msa_paths: dict[str, str] = {}
    if per_chain_msas:
        msa_dir = os.path.join(temp_dir, "msas")
        os.makedirs(msa_dir, exist_ok=True)
        seq_to_csv: dict[str, str] = {}
        for ch_idx, chain in enumerate(sp_complex.chains):
            if not (isinstance(chain, Chain) and chain.entity_type == "protein"):
                continue
            msa = per_chain_msas.get(ch_idx)
            if msa is None:
                continue
            chain_id = chain.id if chain.id is not None else chain_label(ch_idx)
            seq = chain.sequence
            csv_path = seq_to_csv.get(seq)
            if csv_path is None:
                csv_path = os.path.join(msa_dir, f"{hashlib.sha256(seq.encode()).hexdigest()}.csv")
                sequences, _ids = extract_msa_sequences(msa, 0)
                write_msa_csv(sequences, csv_path)
                seq_to_csv[seq] = csv_path
            chain_msa_paths[chain_id] = csv_path
            if verbose:
                logger.info(f"Assigned MSA to chain {chain_id} ({len(msa)} sequences)")

    _, protein_chain_ids = sp_complex.extract_protein_chains()
    for chain_id in protein_chain_ids:
        if chain_id not in chain_msa_paths:
            warnings.warn(
                f"No homologs found for chain {chain_id} - setting msa='empty'.",
                UserWarning,
                stacklevel=2,
            )
    return chain_msa_paths


def complex_to_yaml(
    chains: list[Chain | Fragment],
    chain_msa_paths: dict[str, Any] | None = None,
    affinity_binder_chain_id: str | None = None,
) -> str:
    """Convert a list of chains to Boltz2 YAML input format.

    Args:
        chains (list[Chain | Fragment]): Biopolymer chains (``Chain``) and/or
            ligands (``Fragment``).
        chain_msa_paths (dict[str, Any] | None): Optional dict mapping chain IDs (A, B, C, ...) to
            MSA CSV file paths. Protein chains without a path get ``msa="empty"``
            (single-sequence mode). If None, all protein chains get ``msa="empty"``.
        affinity_binder_chain_id (str | None): If set, emit a ``properties.affinity.binder`` block for that ligand.

    Returns:
        str: YAML-formatted string for Boltz2 input.
    """
    import yaml

    yaml_entries = []

    chain_ids = resolve_chain_ids(chains)
    for i, chain in enumerate(chains):
        e_type = chain.entity_type
        entry: dict[str, Any] = {"id": chain_ids[i]}

        if isinstance(chain, Fragment):
            # Prefer CCD code: Boltz2 uses internal CCD parameterization,
            # avoiding RDKit↔Boltz SMILES canonicalization mismatches.
            if chain.ccd_code:
                entry["ccd"] = chain.ccd_code
            else:
                entry["smiles"] = chain.smiles
        else:
            entry["sequence"] = chain.sequence
            if e_type == "protein":
                chain_id = chain_ids[i]
                entry["msa"] = chain_msa_paths[chain_id] if chain_msa_paths and chain_id in chain_msa_paths else "empty"

        yaml_entries.append({e_type: entry})

    payload: dict[str, Any] = {"sequences": yaml_entries, "predict": {"structure": {"enabled": True}}}
    if affinity_binder_chain_id is not None:
        payload["properties"] = [{"affinity": {"binder": affinity_binder_chain_id}}]

    return str(yaml.dump(payload, sort_keys=False, default_flow_style=False))
