"""tests/structure_design_tests/test_rfdiffusion3.py

Tests for RFdiffusion3."""

import json

import pytest

from proto_tools.tools.structure_design import (
    RFdiffusion3Config,
    RFdiffusion3DesignSpec,
    RFdiffusion3Input,
    run_rfdiffusion3,
)
from tests.conftest import make_persistent_fixture

_persistent_tool = make_persistent_fixture("rfdiffusion3")


# ── Validation tests ────────────────────────────────────────────────────────

def test_rfdiffusion3_input_rejects_empty():
    """Must provide either design_specs or raw_json."""
    with pytest.raises(ValueError, match="Either 'design_specs'.*or 'raw_json'"):
        RFdiffusion3Input()


def test_rfdiffusion3_design_spec_rejects_empty():
    """Each spec needs at least contig, length, or another design parameter."""
    with pytest.raises(ValueError, match="At least one of.*must be provided"):
        RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec()])


# ── JSON spec generation ────────────────────────────────────────────────────

def test_rfdiffusion3_json_spec_generation():
    """Multiple design specs produce keyed JSON with correct fields."""
    inputs = RFdiffusion3Input(
        design_specs=[
            RFdiffusion3DesignSpec(length="100"),
            RFdiffusion3DesignSpec(contig="50-80"),
        ]
    )

    spec = json.loads(inputs.to_json_spec())

    assert "spec-0" in spec
    assert "spec-1" in spec
    assert spec["spec-0"]["length"] == "100"
    assert spec["spec-1"]["contig"] == "50-80"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.include_in_env_report(category="structure_design")
@pytest.mark.uses_gpu
def test_rfdiffusion3_unconditional_design():
    """Basic unconditional design produces a valid structure."""
    inputs = RFdiffusion3Input(
        design_specs=[RFdiffusion3DesignSpec(length="40")]
    )
    config = RFdiffusion3Config(
        n_batches=1,
        diffusion_batch_size=1,
        num_timesteps=50,
    )

    output = run_rfdiffusion3(inputs, config)

    assert len(output.output_structures) > 0
    assert output[0].structure is not None
    assert len(output[0].sequence) > 0
    assert output[0].spec_key == "spec-0"
    assert output[0].design_index == 0
