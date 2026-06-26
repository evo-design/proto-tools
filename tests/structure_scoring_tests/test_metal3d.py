"""Tests for Metal3D tool models and registration."""

from pathlib import Path

import pytest

from proto_tools import (
    Metal3DPredictionConfig,
    Metal3DPredictionInput,
    Metal3DStructureInput,
    Structure,
    run_metal3d_prediction,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.conftest import benchmark_twice

EXAMPLE_PDB = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"
BENCHMARK_PDB = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


def test_metal3d_prediction_is_registered() -> None:
    spec = ToolRegistry.get("metal3d-prediction")
    assert spec.key == "metal3d-prediction"
    assert spec.uses_gpu is True
    assert spec.config_model is Metal3DPredictionConfig


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
