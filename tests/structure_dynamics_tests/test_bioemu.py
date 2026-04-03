"""tests/structure_dynamics_tests/test_bioemu.py.

Tests for BioEmu.
"""

from unittest.mock import patch

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_dynamics.bioemu import (
    BioEmuConfig,
    BioEmuInput,
    run_bioemu,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
)
from tests.tool_infra_tests.test_export_functionality import validate_output

_SAMPLE_SEQUENCE = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

_SAMPLE_PDB_CONTENT = (
    "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C\n"
    "END\n"
)


# ── Input validation ─────────────────────────────────────────────────────────


def test_input_rejects_multi_chain_complex():
    with pytest.raises(ValueError, match="single-chain"):
        BioEmuInput(
            complexes=[
                StructurePredictionComplex(
                    chains=[
                        {"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"},
                        {"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"},
                    ]
                )
            ]
        )


def test_input_rejects_non_protein_entity():
    with pytest.raises(ValueError, match="only supports: protein"):
        BioEmuInput(complexes=[StructurePredictionComplex(chains=[{"sequence": "ACGT", "entity_type": "dna"}])])


def test_input_rejects_invalid_amino_acids():
    with (
        patch(
            "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.return_invalid_protein_chars",
            return_value={"1", "2", "3"},
        ),
        pytest.raises(ValueError, match="Invalid protein characters"),
    ):
        BioEmuInput(
            complexes=[
                StructurePredictionComplex(
                    chains=[
                        {
                            "sequence": "MVLSPADKTNVKAAW123",
                            "entity_type": "protein",
                        }
                    ]
                )
            ]
        )


def test_input_warns_on_long_sequence(caplog):
    long_sequence = "A" * 600
    with (
        patch(
            "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.return_invalid_protein_chars",
            return_value=set(),
        ),
        caplog.at_level("WARNING"),
    ):
        BioEmuInput(
            complexes=[
                StructurePredictionComplex(
                    chains=[
                        {
                            "sequence": long_sequence,
                            "entity_type": "protein",
                        }
                    ]
                )
            ]
        )
    assert "500 residues" in caplog.text


# ── Config validation ────────────────────────────────────────────────────────


def test_config_rejects_num_samples_zero():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BioEmuConfig(num_samples=0)


def test_config_rejects_batch_size_zero():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BioEmuConfig(batch_size=0)


def test_config_rejects_invalid_model_name():
    with pytest.raises(ValueError, match="Input should be"):
        BioEmuConfig(model_name="invalid-model")


# ── Output assembly (mocked dispatch) ────────────────────────────────────────


@pytest.mark.slow
def test_multiple_complexes_produce_separate_ensembles():
    complex_ = StructurePredictionComplex(chains=[{"sequence": _SAMPLE_SEQUENCE, "entity_type": "protein"}])
    bioemu_input = BioEmuInput(complexes=[complex_, complex_])
    bioemu_config = BioEmuConfig(num_samples=10, verbose=False)

    with patch(
        "proto_tools.tools.structure_dynamics.bioemu.bioemu_sample.ToolInstance",
    ) as mock_cls:
        mock_cls.dispatch.return_value = {
            "results": [
                {
                    "pdb_frames": [_SAMPLE_PDB_CONTENT] * 3,
                    "num_frames": 3,
                    "num_residues": len(_SAMPLE_SEQUENCE),
                },
                {
                    "pdb_frames": [_SAMPLE_PDB_CONTENT] * 7,
                    "num_frames": 7,
                    "num_residues": len(_SAMPLE_SEQUENCE),
                },
            ]
        }
        result = run_bioemu(bioemu_input, bioemu_config)

    assert len(result.ensembles) == 2
    assert len(result.ensembles[0].structures) == 3
    assert len(result.ensembles[1].structures) == 7
    assert result.metadata["num_complexes"] == 2
    assert result.metadata["total_structures"] == 10


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_bioemu_sample_end_to_end():
    """Test BioEmu conformational ensemble sampling end-to-end."""
    sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
    inputs = BioEmuInput(
        complexes=[StructurePredictionComplex(chains=[{"sequence": sequence, "entity_type": "protein"}])]
    )
    config = BioEmuConfig(num_samples=5, verbose=False)

    result = run_bioemu(inputs, config)
    validate_output(result)

    assert result.tool_id == "bioemu-sample"
    assert len(result.ensembles) == 1
    assert len(result.ensembles[0].structures) >= 1
    assert result.metadata["num_complexes"] == 1
    assert result.metadata["model_name"] == "bioemu-v1.1"

    for structure in result.ensembles[0].structures:
        assert isinstance(structure, Structure)
        assert structure.structure_pdb is not None
        assert len(structure.structure_pdb) > 0
