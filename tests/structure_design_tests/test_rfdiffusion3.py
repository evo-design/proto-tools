"""Test RFdiffusion3 structure design."""
from pathlib import Path

import pytest

from bio_programming_tools.tools.structure_design import (
    RFdiffusion3Config,
    RFdiffusion3DesignSpec,
    RFdiffusion3Input,
    run_rfdiffusion3,
)

DEFAULT_CHECKPOINT_DIR = Path.home() / ".foundry" / "checkpoints"

@pytest.mark.slow
@pytest.mark.uses_gpu
def test_rfdiffusion3_unconditional_design():
    """Test basic unconditional design."""
    # Simple unconditional design
    inputs = RFdiffusion3Input(
        design_specs=[RFdiffusion3DesignSpec(length="40")]
    )
    config = RFdiffusion3Config(
        n_batches=1,
        diffusion_batch_size=1,
        num_timesteps=50  # Use fewer steps for faster testing
    )

    output = run_rfdiffusion3(inputs, config)

    # Validate output
    assert len(output.output_structures) > 0
    assert output[0].structure is not None
    assert len(output[0].sequence) > 0
    assert output[0].spec_key == "spec-0"
    assert output[0].design_index == 0

@pytest.mark.uses_gpu
def test_rfdiffusion3_input_validation():
    """Test input validation."""
    # Should fail without design_specs or raw_json
    with pytest.raises(ValueError, match="Either 'design_specs'.*or 'raw_json'"):
        RFdiffusion3Input()

    # Should fail without contig/length/raw_spec
    with pytest.raises(ValueError, match="At least one of.*must be provided"):
        RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec()])

def test_rfdiffusion3_json_spec_generation():
    """Test JSON specification generation."""
    inputs = RFdiffusion3Input(
        design_specs=[
            RFdiffusion3DesignSpec(length="100"),
            RFdiffusion3DesignSpec(contig="50-80"),
        ]
    )

    import json
    spec = json.loads(inputs.to_json_spec())

    assert "spec-0" in spec
    assert "spec-1" in spec
    assert spec["spec-0"]["length"] == "100"
    assert spec["spec-1"]["contig"] == "50-80"
