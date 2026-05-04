"""tests/structure_design_tests/test_rfdiffusion3.py.

Tests for RFdiffusion3.
"""

import json

import pytest

from proto_tools.tools.structure_design import (
    RFdiffusion3Config,
    RFdiffusion3DesignSpec,
    RFdiffusion3Input,
    run_rfdiffusion3,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("rfdiffusion3")


# ── Validation tests ────────────────────────────────────────────────────────


def test_rfdiffusion3_input_rejects_empty():
    """Must provide either design_specs or raw_json."""
    with pytest.raises(ValueError, match=r"Either 'design_specs'.*or 'raw_json'"):
        RFdiffusion3Input()


def test_rfdiffusion3_design_spec_rejects_empty():
    """Each spec needs at least contig, length, or another design parameter."""
    with pytest.raises(ValueError, match=r"At least one of.*must be provided"):
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


def test_rfdiffusion3_dispatch_operation_is_design(monkeypatch):
    """Dispatch input must use operation='design' — the only value inference.py accepts."""
    from proto_tools.tools.structure_design.rfdiffusion3 import rfdiffusion3_sample as mod

    captured: dict = {}

    def fake(tool_id, input_data, **kw):
        captured.update(input_data)
        return {"designs": []}

    monkeypatch.setattr(mod.ToolInstance, "dispatch", fake)
    run_rfdiffusion3(RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="40")]), RFdiffusion3Config())

    assert captured["operation"] == "design"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_rfdiffusion3_unconditional_design():
    """Basic unconditional design produces a valid structure."""
    inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="40")])
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


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("rfdiffusion3-design")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_rfdiffusion3_design_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark rfdiffusion3-design: 8 unconditional 128-aa designs in one diffusion batch, 200 timesteps (cold + warm)."""
    inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="128")])
    config = RFdiffusion3Config(
        n_batches=1,
        diffusion_batch_size=8,  # 8 parallel designs per pass; mirrors typical mini-binder generation
        num_timesteps=200,
    )

    result = benchmark_twice(request, "rfdiffusion3", lambda: run_rfdiffusion3(inputs, config))
    validate_output(result)

    assert result.tool_id == "rfdiffusion3-design"
    assert len(result.output_structures) == 8
    for design in result.output_structures:
        assert design.structure is not None
        assert len(design.sequence) == 128
