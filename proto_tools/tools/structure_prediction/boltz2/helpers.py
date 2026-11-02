"""proto_tools/tools/structure_prediction/boltz2/helpers.py.

Shared helpers for Boltz2 structure prediction. Provides utilities for
MSA CSV file writing and YAML input generation.
"""

import csv
import hashlib
import os
from logging import getLogger
from typing import Any

from proto_tools.entities.complex import chain_label
from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures.msa_conditioning import MSASubsampling
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    resolve_chain_ids,
    unwrap_complex_msas,
)
from proto_tools.utils import extract_msa_sequences

logger = getLogger(__name__)


def write_msa_csv(aligned_sequences: list[Any], csv_path: str, keys: list[int]) -> None:
    """Write aligned sequences as Boltz2-format CSV (sequence + key columns).

    The query sequence must be first. Boltz reads the key column as taxonomy and pairs
    rows that share a key across chains; ``key=-1`` keeps a row as unpaired depth.

    Args:
        aligned_sequences (list[Any]): Aligned sequence strings, query first.
        csv_path (str): Path where the CSV file will be written.
        keys (list[int]): Per-row pairing keys, parallel to ``aligned_sequences``.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "key"])
        for seq, key in zip(aligned_sequences, keys, strict=True):
            writer.writerow([seq, key])


def build_chain_msa_paths(
    sp_complex: Complex,
    complex_msas: "ComplexMSAs | None",
    temp_dir: str,
    verbose: int = 0,
) -> dict[str, str]:
    """Map each protein chain ID to its MSA CSV, one sha256-named file per unique sequence.

    boltz requires identical chains to share a single MSA file. Boltz reads the CSV ``key``
    column as taxonomy and pairs rows sharing a key across chains, so keys follow the bundle:
    a paired bundle's row-aligned rows get ``key=<row_idx>``, which pairs them by position,
    and an unpaired bundle gets ``key=-1`` everywhere so unrelated rows are never paired. The
    query gets ``-1`` in both cases: boltz always places it first, and a key would only add a
    duplicate query row. The deep per-chain unpaired MSA, when present, is appended with
    ``key=-1`` so Boltz keeps it as depth without pairing.
    """
    per_chain_msas, unpaired_per_chain, is_paired = unwrap_complex_msas(complex_msas)
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
                keys = [-1] + (list(range(1, len(sequences))) if is_paired else [-1] * (len(sequences) - 1))
                unpaired_msa = (unpaired_per_chain or {}).get(ch_idx)
                if unpaired_msa is not None:
                    # Append the deep unpaired rows for depth, skipping the query and
                    # any sequence already among the paired rows; key=-1 = no pairing.
                    u_seqs, _u_ids = extract_msa_sequences(unpaired_msa, 0)
                    seen = set(sequences)
                    for u_seq in u_seqs[1:]:
                        if u_seq in seen:
                            continue
                        seen.add(u_seq)
                        sequences.append(u_seq)
                        keys.append(-1)
                write_msa_csv(sequences, csv_path, keys=keys)
                seq_to_csv[seq] = csv_path
            chain_msa_paths[chain_id] = csv_path
            if verbose:
                logger.info(f"Assigned MSA to chain {chain_id} ({len(msa)} sequences)")

    _, protein_chain_ids = sp_complex.extract_protein_chains()
    for chain_id in protein_chain_ids:
        if chain_id not in chain_msa_paths:
            logger.info(
                "Chain %s has no supplied homolog alignment; using query-only input for this chain. "
                "Other chains may still use MSAs; inspect structure.msa_conditioning.",
                chain_id,
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


# Rows boltz randomly keeps per recycle when subsample_msa is on; passed as num_subsampled_msa.
BOLTZ2_SUBSAMPLED_ROWS = 1024
# boltz's featurizer keeps at most this many rows of the combined MSA, whatever max_msa_seqs is.
BOLTZ2_FEATURIZER_MAX_ROWS = 16384


def boltz2_msa_subsampling(*, max_msa_seqs: int, subsample_msa: bool, has_msa: bool) -> MSASubsampling:
    """Describe Boltz-2's MSA row selection for the conditioning report.

    boltz always keeps the first ``max_msa_seqs`` unique rows of each CSV, and at most 16384
    combined rows in its featurizer; ``subsample_msa`` then draws a random
    ``num_subsampled_msa`` of those per recycle.
    """
    if not has_msa:
        return MSASubsampling(mode="none")
    depth = min(max_msa_seqs, BOLTZ2_FEATURIZER_MAX_ROWS)
    if subsample_msa:
        return MSASubsampling(mode="stochastic", max_rows=min(depth, BOLTZ2_SUBSAMPLED_ROWS))
    return MSASubsampling(mode="deterministic", max_rows=depth)
