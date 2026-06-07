"""proto_tools/tools/structure_prediction/rf3/helpers.py.

Helpers for the RF3 wrapper: serialize a proto-tools ``Complex`` into the JSON
component schema accepted by ``rf3 fold``, and stage per-chain MSAs as ``.a3m``
files where RF3 can read them.
"""

import hashlib
import json
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

logger = getLogger(__name__)


def build_chain_a3m_paths(
    sp_complex: Complex,
    complex_msas: "ComplexMSAs | None",
    temp_dir: str,
    verbose: int = 0,
) -> dict[str, str]:
    """Map each protein chain ID to an ``.a3m`` file written for RF3.

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
        dict[str, str]: Mapping from chain ID (``"A"``, ``"B"``, …) to the
            ``.a3m`` file path on disk. Protein chains with no MSA are omitted
            and a UserWarning is emitted (RF3 falls back to single-sequence
            mode for those chains).
    """
    per_chain_msas, unpaired_per_chain, _is_paired = unwrap_complex_msas(complex_msas)
    chain_a3m_paths: dict[str, str] = {}
    if per_chain_msas:
        msa_dir = os.path.join(temp_dir, "msas")
        os.makedirs(msa_dir, exist_ok=True)
        seq_to_a3m: dict[str, str] = {}
        for ch_idx, chain in enumerate(sp_complex.chains):
            if not (isinstance(chain, Chain) and chain.entity_type == "protein"):
                continue
            msa = per_chain_msas.get(ch_idx)
            if msa is None:
                continue
            # RF3 pairs by tax_id parsed from the a3m headers, so feed the deep
            # per-chain unpaired MSA when present: it carries full per-chain depth
            # plus the UniRef TaxID= headers RF3 uses to re-pair across chains.
            a3m_msa = (unpaired_per_chain or {}).get(ch_idx) or msa
            chain_id = chain.id if chain.id is not None else chain_label(ch_idx)
            seq = chain.sequence
            a3m_path = seq_to_a3m.get(seq)
            if a3m_path is None:
                a3m_path = os.path.join(msa_dir, f"{hashlib.sha256(seq.encode()).hexdigest()}.a3m")
                a3m_msa.to_a3m_file(a3m_path)
                seq_to_a3m[seq] = a3m_path
            chain_a3m_paths[chain_id] = a3m_path
            if verbose:
                logger.info("Assigned MSA to chain %s (%d sequences)", chain_id, len(a3m_msa))

    _, protein_chain_ids = sp_complex.extract_protein_chains()
    for chain_id in protein_chain_ids:
        if chain_id not in chain_a3m_paths:
            warnings.warn(
                f"No homologs found for chain {chain_id} - rf3 will fall back to single-sequence mode.",
                UserWarning,
                stacklevel=2,
            )
    return chain_a3m_paths


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
