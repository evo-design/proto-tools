"""Tests for Metal3D tool models and prediction."""

from pathlib import Path

import pytest

from proto_tools import (
    Metal3DPredictionConfig,
    Metal3DPredictionInput,
    Metal3DStructureInput,
    Structure,
    run_metal3d_prediction,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

EXAMPLE_PDB = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"
BENCHMARK_PDB = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"

# Exposed checkpoint variants, exercised end-to-end below. Each loads its own
# architecture (kernel 3 for the original weights, kernel 4 for dEVA cat/clean).
_CHECKPOINTS = ["metal3d-cat", "metal3d-clean", "metal3d-original"]

# A handful of metal-binding residues in chain A of pdl1.pdb (ASP/GLU/CYS/HIS).
_CANDIDATE_RESIDUES = {"A": [9, 14, 23, 52, 61]}

_persistent_tool = make_persistent_fixture("metal3d")


def test_metal3d_input_accepts_single_structure_with_candidate_residues() -> None:
    inputs = Metal3DPredictionInput(
        inputs={
            "structure": str(EXAMPLE_PDB),
            "candidate_residues": {"A": [1, 2, 3]},
        }
    )

    assert len(inputs.inputs) == 1
    assert inputs.inputs[0].candidate_residues is not None
    assert inputs.inputs[0].candidate_residues.chains == {"A": [1, 2, 3]}


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
@pytest.mark.parametrize("checkpoint", _CHECKPOINTS)
def test_metal3d_prediction(checkpoint: str) -> None:
    """Each checkpoint scores candidate residues end-to-end and returns valid metrics."""
    inputs = Metal3DPredictionInput(
        inputs=[Metal3DStructureInput(structure=str(EXAMPLE_PDB), candidate_residues=_CANDIDATE_RESIDUES)]
    )
    config = Metal3DPredictionConfig(model_checkpoint=checkpoint, device="cuda")

    output = run_metal3d_prediction(inputs, config)
    validate_output(output)
    assert_metrics_in_spec(output)

    assert output.success
    assert output.tool_id == "metal3d-prediction"
    assert len(output.results) == 1

    result = output.results[0]
    assert isinstance(result.found, bool)
    assert 0.0 <= result["pmetal"] <= 1.0
    assert result.annotated_structure.structure_format == "pdb"
    assert result.residue_probabilities
    assert all(0.0 <= rp["probability"] <= 1.0 for rp in result.residue_probabilities)


@pytest.mark.uses_gpu
def test_metal3d_prediction_whole_protein() -> None:
    """The default checkpoint scores all metal-binding residues when none are specified."""
    inputs = Metal3DPredictionInput(inputs=[Metal3DStructureInput(structure=str(EXAMPLE_PDB))])
    config = Metal3DPredictionConfig(device="cuda")

    output = run_metal3d_prediction(inputs, config)
    validate_output(output)
    assert_metrics_in_spec(output)

    assert output.success
    assert len(output.results) == 1
    result = output.results[0]
    assert isinstance(result.found, bool)
    assert 0.0 <= result["pmetal"] <= 1.0
    assert result.annotated_structure.structure_format == "pdb"
    assert result.residue_probabilities


@pytest.mark.benchmark("metal3d-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_metal3d_prediction_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark metal3d-prediction: all metal-binding residues of renin (~340 aa) with metal3d-cat (cold + warm)."""
    inputs = Metal3DPredictionInput(inputs=[Metal3DStructureInput(structure=Structure(structure=str(BENCHMARK_PDB)))])
    config = Metal3DPredictionConfig(model_checkpoint="metal3d-cat", device="cuda")

    output = benchmark_twice(request, "metal3d", lambda: run_metal3d_prediction(inputs, config))

    assert len(output.results) == 1
    result = output.results[0]
    assert isinstance(result.found, bool)
    assert 0.0 <= result["pmetal"] <= 1.0
    assert result.annotated_structure.structure_format == "pdb"
