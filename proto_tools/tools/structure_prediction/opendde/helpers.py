"""proto_tools/tools/structure_prediction/opendde/helpers.py.

Shared helpers for OpenDDE structure prediction: build the OpenDDE inference
JSON job dict from a complex's chains, and write per-chain MSAs to A3M files.
"""

import os
from logging import getLogger
from typing import Any

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    resolve_chain_ids,
    unwrap_complex_msas,
    write_paired_a3m_with_uniprot_headers,
)

logger = getLogger(__name__)


def _ccd_code(code: str) -> str:
    """Return an OpenDDE-style ``CCD_``-prefixed code, tolerating an already-prefixed value."""
    return code if code.startswith("CCD_") else f"CCD_{code}"


def build_chain_msa_paths(
    sp_complex: Complex,
    complex_msas: "ComplexMSAs | None",
    temp_dir: str,
    verbose: int = 0,
) -> tuple[dict[str, str], dict[str, str]]:
    """Write per-chain A3M files for OpenDDE's ``unpairedMsaPath`` and ``pairedMsaPath``.

    OpenDDE takes one A3M path per protein chain for each of two roles and builds the
    MSA features itself:

    - ``unpairedMsaPath``: per-chain depth, block-diagonalized across the complex.
      Identical sequences share a single file.
    - ``pairedMsaPath``: taxonomy-paired rows for heterocomplexes. When the supplied
      MSAs are paired (``is_paired``), the primary per-chain MSA is row-aligned by
      species, so it is written with UniProt-style headers via
      ``write_paired_a3m_with_uniprot_headers`` — OpenDDE's ``_UNIPROT_REGEX`` then
      extracts a per-row species id and ``MSAPairingEngine.pair_chains_by_species``
      engages cross-chain pairing (same AF3/Protenix lineage). Single-sequence
      chains are omitted and fall back to single-sequence mode.

    Args:
        sp_complex (Complex): Complex whose protein chains are being aligned.
        complex_msas (ComplexMSAs | None): Per-chain MSAs keyed by chain index.
        temp_dir (str): Scratch directory for the written A3M files.
        verbose (int): Verbosity level (truthy = log per-chain assignment).

    Returns:
        tuple[dict[str, str], dict[str, str]]: ``(unpaired_paths, paired_paths)`` —
            each maps a protein chain ID to its A3M file path. ``paired_paths`` is
            empty unless the supplied MSAs are paired.
    """
    per_chain_msas, unpaired_per_chain, is_paired = unwrap_complex_msas(complex_msas)
    unpaired_paths: dict[str, str] = {}
    paired_paths: dict[str, str] = {}
    if not per_chain_msas:
        return unpaired_paths, paired_paths

    msa_dir = os.path.join(temp_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)
    chain_ids = resolve_chain_ids(sp_complex.chains)
    seq_to_a3m: dict[str, str] = {}
    for ch_idx, chain in enumerate(sp_complex.chains):
        if not (isinstance(chain, Chain) and chain.entity_type == "protein"):
            continue
        chain_id = chain_ids[ch_idx]

        # Unpaired depth: prefer the deep unpaired MSA, fall back to the primary set.
        unpaired_msa: MSA | None = (unpaired_per_chain or {}).get(ch_idx) or per_chain_msas.get(ch_idx)
        if unpaired_msa is not None:
            seq = chain.sequence
            a3m_path = seq_to_a3m.get(seq)
            if a3m_path is None:
                a3m_path = os.path.join(msa_dir, f"chain_{chain_id}_{ch_idx}.a3m")
                unpaired_msa.to_a3m_file(a3m_path, query_index=0)
                seq_to_a3m[seq] = a3m_path
            unpaired_paths[chain_id] = a3m_path

        # Paired set is taxonomy-aligned by row; emit it with pairing-engaging UniProt headers.
        if is_paired:
            paired_msa = per_chain_msas.get(ch_idx)
            if paired_msa is not None:
                paired_path = os.path.join(msa_dir, f"chain_{chain_id}_{ch_idx}.paired.a3m")
                write_paired_a3m_with_uniprot_headers(paired_msa, paired_path)
                paired_paths[chain_id] = paired_path

        if verbose and (chain_id in unpaired_paths or chain_id in paired_paths):
            logger.info(
                f"Assigned MSA to chain {chain_id} "
                f"(unpaired={'yes' if chain_id in unpaired_paths else 'no'}, "
                f"paired={'yes' if chain_id in paired_paths else 'no'})"
            )
    return unpaired_paths, paired_paths


def complex_to_opendde_json(
    chains: list[Chain | Fragment],
    name: str,
    seeds: list[int],
    chain_msa_paths: dict[str, str] | None = None,
    chain_paired_msa_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a complex's chains to a single OpenDDE inference JSON job dict.

    The OpenDDE JSON dialect is documented at
    https://github.com/aurekaresearch/OpenDDE/blob/main/docs/infer_json_format.md .
    Each entity is one ``sequences`` entry keyed by its OpenDDE type
    (``proteinChain`` / ``dnaSequence`` / ``rnaSequence`` / ``ligand``) with a
    ``count`` and single-element ``id`` list. Ligands prefer the CCD code, falling
    back to the SMILES string. Protein chains receive an ``unpairedMsaPath`` when a
    caller-supplied MSA is available; otherwise OpenDDE folds them single-sequence.

    Args:
        chains (list[Chain | Fragment]): Biopolymer chains and/or ligand fragments.
        name (str): Job name; drives OpenDDE's output path (``<out>/<name>/...``).
        seeds (list[int]): OpenDDE ``modelSeeds`` for this job.
        chain_msa_paths (dict[str, str] | None): Chain ID → unpaired A3M path for
            protein chains. Chains absent from the map fold single-sequence.
        chain_paired_msa_paths (dict[str, str] | None): Chain ID → paired A3M path
            (taxonomy-aligned) for heterocomplex pairing. Chains absent from the map
            get no ``pairedMsaPath``.

    Returns:
        dict[str, Any]: One OpenDDE job object (wrap in a list before writing).
    """
    chain_msa_paths = chain_msa_paths or {}
    chain_paired_msa_paths = chain_paired_msa_paths or {}
    chain_ids = resolve_chain_ids(chains)
    sequences: list[dict[str, Any]] = []

    for idx, chain in enumerate(chains):
        chain_id = chain_ids[idx]

        if isinstance(chain, Fragment):
            # OpenDDE reads a CCD code only when it is ``CCD_``-prefixed; any other
            # string is parsed as SMILES. Prefer the CCD code, fall back to raw SMILES.
            if chain.ccd_code:
                ligand_ref = _ccd_code(chain.ccd_code)
            elif chain.smiles:
                ligand_ref = chain.smiles
            else:
                raise ValueError(
                    f"Ligand chain {chain_id!r} has neither a CCD code nor a SMILES string; "
                    "OpenDDE requires one of them to place a ligand."
                )
            sequences.append({"ligand": {"ligand": ligand_ref, "count": 1, "id": [chain_id]}})
            continue

        entity_key = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}.get(
            chain.entity_type or ""
        )
        if entity_key is None:
            raise ValueError(f"OpenDDE does not support entity type {chain.entity_type!r} for chain {chain_id!r}.")
        entry: dict[str, Any] = {"sequence": chain.sequence, "count": 1, "id": [chain_id]}

        if chain.modifications:
            # OpenDDE expects CCD codes prefixed with ``CCD_``; proto_tools stores the bare code.
            if chain.entity_type == "protein":
                entry["modifications"] = [
                    {"ptmType": _ccd_code(mod.modification_code), "ptmPosition": mod.position}
                    for mod in chain.modifications
                ]
            else:
                entry["modifications"] = [
                    {"modificationType": _ccd_code(mod.modification_code), "basePosition": mod.position}
                    for mod in chain.modifications
                ]

        if chain.entity_type == "protein" and chain_id in chain_msa_paths:
            entry["unpairedMsaPath"] = chain_msa_paths[chain_id]
        if chain.entity_type == "protein" and chain_id in chain_paired_msa_paths:
            entry["pairedMsaPath"] = chain_paired_msa_paths[chain_id]

        sequences.append({entity_key: entry})

    return {"name": name, "modelSeeds": list(seeds), "sequences": sequences}
