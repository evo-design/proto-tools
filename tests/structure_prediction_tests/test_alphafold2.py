"""tests/structure_prediction_tests/test_alphafold2.py

Tests for AlphaFold2."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bio_programming_tools.entities.structures import is_valid_structure
from bio_programming_tools.tools.structure_prediction import (
    AlphaFold2Config,
    AlphaFold2Input,
    StructurePredictionComplex,
    run_alphafold2,
)
from bio_programming_tools.utils.tool_instance import ToolInstance

_HOMOOLIGOMER_SEQ = "MARFLGLYTWHK"

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _persistent_worker(request):
    if request.config.getoption("--cpu"):
        yield
        return
    with ToolInstance.scope():
        yield


# ── Input validation ──────────────────────────────────────────────────────────

def test_alphafold2_input_rejects_non_protein_sequence():
    """AlphaFold2 validator rejects sequences that aren't valid protein."""
    with pytest.raises(ValidationError, match="unsupported entity types"):
        AlphaFold2Input(complexes=["MKTL123"])


def test_alphafold2_input_accepts_x_for_unknown_residue():
    """'X' is a valid placeholder for unknown amino acids and must not raise."""
    inp = AlphaFold2Input(complexes=["MKTLX"])
    assert inp.complexes[0].chain_sequences[0] == "MKTLX"


def test_alphafold2_input_rejects_dna_entity_type():
    """AlphaFold2 only supports protein chains; DNA must be rejected."""
    with pytest.raises(ValidationError, match="unsupported entity types"):
        AlphaFold2Input(
            complexes=[StructurePredictionComplex(
                chains=[{"sequence": "ATCG", "entity_type": "dna"}]
            )]
        )


def test_alphafold2_input_rejects_chain_modifications():
    """AlphaFold2 does not allow chain modifications (ALLOWS_CHAIN_MODIFICATIONS=False)."""
    with pytest.raises(ValidationError, match="does not allow chain modifications"):
        AlphaFold2Input(
            complexes=[StructurePredictionComplex(
                chains=[{"sequence": "MVLSPADKTN", "entity_type": "protein", "modifications": [(4, "SEP")]}]
            )]
        )


# ── Config validation ─────────────────────────────────────────────────────────

def test_alphafold2_config_rejects_model_num_and_ensemble_together():
    """model_num != 1 combined with num_ensemble_models > 1 must raise."""
    with pytest.raises(ValidationError, match="mutually exclusive"):
        AlphaFold2Config(model_num=2, num_ensemble_models=3)


def test_alphafold2_config_rejects_num_recycles_below_zero():
    """num_recycles has ge=0; negative values must raise."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AlphaFold2Config(num_recycles=-1)


def test_alphafold2_config_rejects_num_recycles_above_48():
    """num_recycles has le=48; values above 48 must raise."""
    with pytest.raises(ValidationError, match="less than or equal to 48"):
        AlphaFold2Config(num_recycles=49)


def test_alphafold2_config_rejects_model_num_out_of_range():
    """model_num must be between 1 and 5 inclusive."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        AlphaFold2Config(model_num=0)
    with pytest.raises(ValidationError, match="less than or equal to 5"):
        AlphaFold2Config(model_num=6)


# ---------------------------------------------------------------------------
# Integration tests

@pytest.mark.uses_gpu
def test_homooligomer():
    """Verify the homo-oligomer code path (identical chains → copies)."""
    complexes = [StructurePredictionComplex(chains=[_HOMOOLIGOMER_SEQ, _HOMOOLIGOMER_SEQ])]

    inputs = AlphaFold2Input(complexes=complexes)
    config = AlphaFold2Config(use_msa=False, verbose=True)
    output = run_alphafold2(inputs, config)

    assert output.success
    assert len(output.structures) == 1

    structure = output.structures[0]
    assert is_valid_structure(structure.structure_cif)
    assert 0 <= structure.metrics["avg_plddt"] <= 1.0
    assert 0 <= structure.metrics["ptm"] <= 1.0
    assert structure.metrics["iptm"] is not None
