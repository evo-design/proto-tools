"""tests/structure_prediction_tests/test_esmfold2.py.

Tests for ESMFold2 all-atom complex structure prediction.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from proto_tools.entities.complex import Chain, Complex
from proto_tools.entities.ligands import Fragment
from proto_tools.tools.structure_prediction.esmfold2 import (
    ESMFold2Config,
    ESMFold2Input,
    ESMFold2Metrics,
    ESMFold2Output,
    run_esmfold2,
)
from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import (
    _chain_to_payload,
    _rename_output_chains_to_input_ids,
)
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

# Ubiquitin (PDB 1UBQ), 76 aa: small, well-folded benchmark protein.
_UBIQUITIN_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Two short interacting helix peptides: heterodimer for iptm coverage.
_HELIX_A = "EELLKKLEELLKKLEELLKK"
_HELIX_B = "KKLLEEKKLLEEKKLLEEKK"


# ── Config validator (no GPU) ───────────────────────────────────────────────


def test_esmfold2_config_forbids_msa_with_fast_variant():
    """ESMFold2-Fast is single-sequence; pairing it with use_msa=True must raise."""
    with pytest.raises(ValidationError, match="does not support MSA conditioning"):
        ESMFold2Config(model_checkpoint="esmfold2-fast", use_msa=True)


# ── Serialization tests (no GPU) ─────────────────────────────────────────────


def test_esmfold2_payload_preserves_chain_ids_and_positional_fallbacks():
    """ESMFold2 payloads should expose A/B-style IDs for downstream chain selection."""
    explicit = Chain(id="T", sequence=_HELIX_A, entity_type="protein")
    fallback = Chain(sequence=_HELIX_B, entity_type="protein")
    ligand = Fragment(id="L", ccd_code="ATP")

    assert _chain_to_payload(explicit, 0)["id"] == "T"
    assert _chain_to_payload(fallback, 1)["id"] == "B"
    assert _chain_to_payload(ligand, 2)["id"] == "L"


def test_esmfold2_output_chain_ids_are_normalized_to_payload_ids():
    """Returned ESMFold2 structures should be selectable using input payload chain IDs."""
    structure = Mock()
    renamed = Mock()
    structure.get_chain_ids.return_value = ["1", "2"]
    structure.with_renamed_chains.return_value = renamed

    result = _rename_output_chains_to_input_ids(
        structure,
        [
            {"id": "A", "entity_type": "protein"},
            {"id": "B", "entity_type": "protein"},
        ],
    )

    assert result is renamed
    structure.with_renamed_chains.assert_called_once_with({"1": "A", "2": "B"})


# ── GPU integration tests ───────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_esmfold2_single_protein_basic():
    """Fold a single ubiquitin chain with the default fast checkpoint."""
    result = run_esmfold2(ESMFold2Input(complexes=[_UBIQUITIN_SEQUENCE]), ESMFold2Config(seed=0))

    assert result.success
    assert result.tool_id == "esmfold2-prediction"
    assert isinstance(result, ESMFold2Output)
    assert len(result.structures) == 1

    structure = result.structures[0]
    assert structure.structure_cif and len(structure.structure_cif) > 0
    assert isinstance(structure.metrics, ESMFold2Metrics)
    assert structure.metrics["plddt"] is not None
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_esmfold2_multi_chain_complex():
    """Two short helix peptides as a heterodimer (exercises the iptm path)."""
    heterodimer = Complex(
        chains=[
            Chain(sequence=_HELIX_A, entity_type="protein"),
            Chain(sequence=_HELIX_B, entity_type="protein"),
        ]
    )
    result = run_esmfold2(ESMFold2Input(complexes=[heterodimer]), ESMFold2Config(seed=0))

    assert result.success
    assert len(result.structures) == 1
    metrics = result.structures[0].metrics
    assert metrics["iptm"] is not None
    assert 0.0 <= metrics["iptm"] <= 1.0


@pytest.mark.uses_gpu
def test_esmfold2_ligand_ccd():
    """Protein + CCD-coded ligand (ATP) folds successfully."""
    complex_ = Complex(chains=[Chain(sequence=_UBIQUITIN_SEQUENCE, entity_type="protein"), Fragment(ccd_code="ATP")])
    result = run_esmfold2(ESMFold2Input(complexes=[complex_]), ESMFold2Config(seed=0))

    assert result.success
    assert len(result.structures) == 1
    assert result.structures[0].structure_cif
