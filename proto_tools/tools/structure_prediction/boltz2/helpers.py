"""proto_tools/tools/structure_prediction/boltz2/helpers.py.

Shared helpers for Boltz2 structure prediction. Provides utilities for
MSA CSV file writing and YAML input generation.
"""

import csv
import hashlib
import os
from logging import getLogger
from typing import Any

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures.msa_conditioning import MSASubsampling
from proto_tools.tools.structure_prediction.msa_plan import SentMSA, protein_chain_msas, unpaired_depth_rows
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    resolve_chain_ids,
)
from proto_tools.utils import extract_msa_sequences

logger = getLogger(__name__)

# Boltz-2 is written a chain's CSV only from its primary alignment.
_UNPAIRED_ONLY_REASON = "only unpaired rows were supplied; Boltz-2 is sent a chain's primary alignment"


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
) -> tuple[dict[str, str], dict[int, SentMSA]]:
    """Map each protein chain ID to its MSA CSV, one sha256-named file per unique sequence.

    boltz requires identical chains to share a single MSA file. Boltz reads the CSV ``key``
    column as taxonomy and pairs rows sharing a key across chains, so keys follow the bundle:
    a paired bundle's row-aligned rows get ``key=<row_idx>``, which pairs them by position,
    and an unpaired bundle gets ``key=-1`` everywhere so unrelated rows are never paired. The
    query gets ``-1`` in both cases: boltz always places it first, and a key would only add a
    duplicate query row. The deep per-chain unpaired MSA, when present, is appended with
    ``key=-1`` so Boltz keeps it as depth without pairing.

    Returns:
        tuple[dict[str, str], dict[int, SentMSA]]: Chain ID to CSV path, and the rows written
            per protein chain index. Chains without a primary alignment get no CSV.
    """
    if complex_msas is None:
        return {}, {}
    is_paired = complex_msas.paired
    chain_msa_paths: dict[str, str] = {}
    sent: dict[int, SentMSA] = {}
    written: dict[str, tuple[str, SentMSA]] = {}
    for chain in protein_chain_msas(sp_complex.chains, complex_msas):
        if chain.primary is None:
            if chain.unpaired is not None:
                sent[chain.index] = SentMSA(ignored_reason=_UNPAIRED_ONLY_REASON)
            logger.info(
                "Chain %s has no supplied homolog alignment; using query-only input for this chain. "
                "Other chains may still use MSAs; inspect structure.msa_conditioning.",
                chain.chain_id,
            )
            continue
        seq = chain.chain.sequence
        if seq not in written:
            msa_dir = os.path.join(temp_dir, "msas")
            os.makedirs(msa_dir, exist_ok=True)
            csv_path = os.path.join(msa_dir, f"{hashlib.sha256(seq.encode()).hexdigest()}.csv")
            sequences, _ids = extract_msa_sequences(chain.primary, 0)
            keys = [-1] + (list(range(1, len(sequences))) if is_paired else [-1] * (len(sequences) - 1))
            depth = [row for row, _id in unpaired_depth_rows(sequences, chain.unpaired)]
            write_msa_csv(sequences + depth, csv_path, keys=keys + [-1] * len(depth))
            written[seq] = (csv_path, SentMSA(rows=len(sequences), unpaired_rows=len(depth), paired=is_paired))
        chain_msa_paths[chain.chain_id], sent[chain.index] = written[seq]
        if verbose:
            logger.info(f"Assigned MSA to chain {chain.chain_id} ({len(chain.primary)} sequences)")
    return chain_msa_paths, sent


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
