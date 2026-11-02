"""proto_tools/tools/structure_prediction/opendde/helpers.py.

Shared helpers for OpenDDE structure prediction: build the OpenDDE inference
JSON job dict from a complex's chains, and write per-chain MSAs to A3M files.
"""

import os
from logging import getLogger
from typing import Any

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_prediction.msa_plan import SentMSA, protein_chain_msas, short_chain_reason
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    resolve_chain_ids,
    write_paired_a3m_with_uniprot_headers,
    write_query_only_a3m,
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
) -> tuple[dict[str, str], dict[str, str], dict[int, SentMSA]]:
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
      engages cross-chain pairing (same AF3/Protenix lineage).

    Whenever ``complex_msas`` is given, every protein chain gets an ``unpairedMsaPath``,
    query-only for chains without rows: OpenDDE searches its own MSA server for the whole
    complex if any chain lacks one, overwriting the supplied MSAs.

    Args:
        sp_complex (Complex): Complex whose protein chains are being aligned.
        complex_msas (ComplexMSAs | None): Per-chain MSAs keyed by chain index.
        temp_dir (str): Scratch directory for the written A3M files.
        verbose (int): Verbosity level (truthy = log per-chain assignment).

    Returns:
        tuple[dict[str, str], dict[str, str], dict[int, SentMSA]]:
            ``(unpaired_paths, paired_paths, sent)``. The paths map a protein chain ID to its A3M
            file (``paired_paths`` is empty unless the MSAs are paired); ``sent`` records the
            rows written per protein chain index.
    """
    unpaired_paths: dict[str, str] = {}
    paired_paths: dict[str, str] = {}
    sent: dict[int, SentMSA] = {}
    if complex_msas is None:
        return unpaired_paths, paired_paths, sent

    msa_dir = os.path.join(temp_dir, "msas")
    os.makedirs(msa_dir, exist_ok=True)
    # Identical chains share the first chain's unpaired a3m.
    written: dict[str, tuple[str, int]] = {}
    for chain in protein_chain_msas(sp_complex.chains, complex_msas):
        chain_id, ch_idx = chain.chain_id, chain.index
        # Unpaired depth: prefer the deep unpaired MSA, fall back to the primary set.
        unpaired_msa: MSA | None = chain.unpaired or chain.primary
        if unpaired_msa is not None:
            seq = chain.chain.sequence
            if seq not in written:
                a3m_path = os.path.join(msa_dir, f"chain_{chain_id}_{ch_idx}.a3m")
                unpaired_msa.to_a3m_file(a3m_path, query_index=0)
                written[seq] = (a3m_path, len(unpaired_msa))
            unpaired_paths[chain_id], file_rows = written[seq]
        else:
            # Every protein chain needs a path: OpenDDE treats any chain without one as a request
            # to search its own MSA server for the whole complex, overwriting the supplied MSAs.
            a3m_path = os.path.join(msa_dir, f"chain_{chain_id}_{ch_idx}.a3m")
            write_query_only_a3m(chain.chain.sequence, a3m_path)
            unpaired_paths[chain_id], file_rows = a3m_path, 1
            if verbose:
                logger.info("Assigned query-only MSA to chain %s, which has no alignment rows", chain_id)

        # Paired set is taxonomy-aligned by row; emit it with pairing-engaging UniProt headers.
        paired_msa = chain.primary if complex_msas.paired else None
        paired = paired_msa is not None
        if paired_msa is not None:
            paired_path = os.path.join(msa_dir, f"chain_{chain_id}_{ch_idx}.paired.a3m")
            write_paired_a3m_with_uniprot_headers(paired_msa, paired_path)
            paired_paths[chain_id] = paired_path
        sent[ch_idx] = SentMSA(
            rows=len(paired_msa) if paired_msa is not None else file_rows,
            # Without a deep unpaired MSA the unpaired file repeats the paired rows: no extra depth.
            unpaired_rows=file_rows - 1 if paired and chain.unpaired is not None else 0,
            paired=paired,
            ignored_reason=short_chain_reason(chain.chain, "OpenDDE"),
        )

        if verbose:
            logger.info(
                f"Assigned MSA to chain {chain_id} "
                f"(unpaired={'yes' if chain_id in unpaired_paths else 'no'}, "
                f"paired={'yes' if chain_id in paired_paths else 'no'})"
            )
    return unpaired_paths, paired_paths, sent


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
