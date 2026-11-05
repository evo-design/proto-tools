"""proto_tools/tools/structure_prediction/rf3/helpers.py.

Helpers for the RF3 wrapper: serialize a proto-tools ``Complex`` into the JSON
component schema accepted by ``rf3 fold``, and stage per-chain MSAs as ``.a3m``
files where RF3 can read them.
"""

import hashlib
import json
import os
from logging import getLogger
from typing import Any

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_prediction.msa_plan import SentMSA, protein_chain_msas
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    resolve_chain_ids,
)

logger = getLogger(__name__)


def build_chain_a3m_paths(
    sp_complex: Complex,
    complex_msas: "ComplexMSAs | None",
    temp_dir: str,
    verbose: int = 0,
) -> tuple[dict[str, str], dict[int, SentMSA]]:
    """Map each protein chain ID to an ``.a3m`` file written for RF3, and record what was sent.

    RF3's ``rf3 fold`` input schema expects each protein component's ``msa_path``
    to point to an A3M-format multiple sequence alignment (one of the standard
    HHsuite formats). Identical chains share a single file, keyed by SHA-256 of
    the sequence, mirroring the Boltz2 helper's dedup pattern.

    Args:
        sp_complex (Complex): The complex whose protein chains need MSAs.
        complex_msas (ComplexMSAs | None): Per-chain MSAs (keyed by chain index)
            and a ``paired`` flag, typically populated by ``preprocess()``.
        temp_dir (str): A directory where ``.a3m`` files can be written.
        verbose (int): Non-zero enables INFO log lines per chain.

    Returns:
        tuple[dict[str, str], dict[int, SentMSA]]: Chain ID (``"A"``, ``"B"``, …) to the
            ``.a3m`` file path on disk, and the rows written per protein chain index. Protein
            chains with no alignment are omitted and an INFO message identifies query-only
            input for those chains.
    """
    chain_a3m_paths: dict[str, str] = {}
    written: dict[str, tuple[str, MSA]] = {}
    sent_msas: dict[int, MSA] = {}
    sent: dict[int, SentMSA] = {}
    if complex_msas is None:
        return chain_a3m_paths, sent
    sequences: dict[int, str] = {}
    for chain in protein_chain_msas(sp_complex.chains, complex_msas):
        msa = chain.unpaired or chain.primary
        if msa is None:
            logger.info(
                "Chain %s has no supplied homolog alignment; using query-only input for this chain. "
                "Other chains may still use MSAs; inspect structure.msa_conditioning.",
                chain.chain_id,
            )
            continue
        # Feed the deep per-chain unpaired MSA when present for full per-chain depth. RF3
        # re-pairs across chains only by TaxID= in the a3m headers, which ColabFold rows lack.
        seq = sequences[chain.index] = chain.chain.sequence
        if seq not in written:
            msa_dir = os.path.join(temp_dir, "msas")
            os.makedirs(msa_dir, exist_ok=True)
            a3m_path = os.path.join(msa_dir, f"{hashlib.sha256(seq.encode()).hexdigest()}.a3m")
            msa.to_a3m_file(a3m_path)
            written[seq] = (a3m_path, msa)
        chain_a3m_paths[chain.chain_id], sent_msas[chain.index] = written[seq]
        if verbose:
            logger.info("Assigned MSA to chain %s (%d sequences)", chain.chain_id, len(sent_msas[chain.index]))
    # RF3 pairs rows only when at least two distinct chain sequences carry TaxID= headers.
    tagged = {sequences[i] for i, msa in sent_msas.items() if _has_taxonomy_headers(msa)}
    for index, msa in sent_msas.items():
        sent[index] = SentMSA(rows=len(msa), paired=len(tagged) >= 2 and _has_taxonomy_headers(msa))
    return chain_a3m_paths, sent


def complex_to_rf3_json(
    chains: list[Chain | Fragment],
    name: str = "complex",
    chain_msa_paths: dict[str, str] | None = None,
) -> str:
    """Convert a list of chains to the RF3 input JSON format.

    RF3's CLI consumes a JSON file containing a list of examples, where each
    example has a ``name`` and a list of ``components``. Each component is one
    of:

    * Protein / nucleic acid: ``{"seq": "...", "chain_id": "A", "msa_path": "..."}``
      (``chain_id`` and ``msa_path`` optional)
    * Ligand by SMILES: ``{"smiles": "..."}``
    * Ligand by CCD code: ``{"ccd_code": "..."}``

    Note: ``cyclic_chains`` is **not** read from the JSON wrapper by upstream
    (``rf3.data.InferenceInput.from_json_dict`` ignores it). Cyclization must
    be supplied via the Hydra CLI override ``cyclic_chains=[A,B]`` — this
    wrapper does that from :func:`run_rf3_prediction_on_complex`.

    Args:
        chains (list[Chain | Fragment]): Biopolymer chains (``Chain``) and/or
            ligands (``Fragment``).
        name (str): The ``name`` field of the single example wrapper. Default
            ``"complex"``.
        chain_msa_paths (dict[str, str] | None): Optional dict mapping chain
            IDs (A, B, C, …) to MSA paths. Protein chains without a path get no
            ``msa_path`` (single-sequence mode).

    Returns:
        str: JSON-encoded list with one example wrapper.

    Raises:
        ValueError: If a ``Fragment`` has neither ``ccd_code`` nor ``smiles``.
    """
    chain_ids = resolve_chain_ids(chains)
    components: list[dict[str, Any]] = []

    for i, chain in enumerate(chains):
        if isinstance(chain, Fragment):
            # Prefer CCD code: avoids RDKit↔upstream SMILES canonicalization mismatches.
            if chain.ccd_code:
                components.append({"ccd_code": chain.ccd_code})
            elif chain.smiles:
                components.append({"smiles": chain.smiles})
            else:
                raise ValueError(f"Ligand fragment at index {i} has neither ccd_code nor smiles")
        else:
            entry: dict[str, Any] = {"seq": chain.sequence, "chain_id": chain_ids[i]}
            if chain_msa_paths and chain_ids[i] in chain_msa_paths:
                entry["msa_path"] = chain_msa_paths[chain_ids[i]]
            components.append(entry)

    return json.dumps([{"name": name, "components": components}], indent=2)


def _has_taxonomy_headers(msa: MSA) -> bool:
    """Whether any non-query row carries the ``TaxID=`` header RF3 (atomworks) pairs on."""
    return any("TaxID=" in sequence_id for sequence_id in msa.sequence_ids[1:])
