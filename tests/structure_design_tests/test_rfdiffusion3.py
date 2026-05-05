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
    """Each spec needs at least one design parameter (contig, length, symmetry, ...)."""
    with pytest.raises(ValueError, match=r"At least one design parameter"):
        RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec()])


# ── Config: CLI kwargs assembly ─────────────────────────────────────────────


def test_rfdiffusion3_config_get_cli_kwargs():
    """Sampler knobs emit under inference_sampler.* dotted paths; top-level toggles stay flat.

    Conditional fields (n_recycle, cfg_features, cfg_t_max) only emit when explicitly
    set so upstream defaults stand for None.
    """
    # Defaults: conditionals omitted, required fields present
    default_kwargs = RFdiffusion3Config().get_cli_kwargs()
    for omitted in (
        "inference_sampler.n_recycle",
        "inference_sampler.cfg_features",
        "inference_sampler.cfg_t_max",
    ):
        assert omitted not in default_kwargs

    # Explicit override of every typed sampler + top-level toggle
    config = RFdiffusion3Config(
        cfg_scale=2.0,
        gamma_0=0.4,
        sampler_kind="symmetry",
        center_option="motif",
        use_classifier_free_guidance=True,
        n_recycle=3,
        cfg_features=["active_donor"],
        cfg_t_max=0.8,
        dump_trajectories=True,
        align_trajectory_structures=True,
        prevalidate_inputs=True,
    )
    kwargs = config.get_cli_kwargs()

    # Sampler knobs use dotted Hydra paths
    expected_dotted = {
        "inference_sampler.cfg_scale": 2.0,
        "inference_sampler.gamma_0": 0.4,
        "inference_sampler.kind": "symmetry",
        "inference_sampler.center_option": "motif",
        "inference_sampler.use_classifier_free_guidance": True,
        "inference_sampler.n_recycle": 3,
        "inference_sampler.cfg_features": ["active_donor"],
        "inference_sampler.cfg_t_max": 0.8,
    }
    for key, value in expected_dotted.items():
        assert kwargs[key] == value, f"{key} not emitted as dotted path"

    # Top-level toggles stay flat
    assert kwargs["dump_trajectories"] is True
    assert kwargs["align_trajectory_structures"] is True
    assert kwargs["prevalidate_inputs"] is True

    # Sampler keys MUST NOT also appear flat — Hydra would silently ignore them
    for flat in (
        "cfg_scale",
        "gamma_0",
        "kind",
        "center_option",
        "use_classifier_free_guidance",
        "n_recycle",
        "cfg_features",
        "cfg_t_max",
    ):
        assert flat not in kwargs


def test_rfdiffusion3_typed_fields_override_extras_on_collision():
    """Both Config.get_cli_kwargs and DesignSpec.to_dict resolve typed-vs-extras collisions for typed."""
    config = RFdiffusion3Config(cfg_scale=2.0, **{"inference_sampler.cfg_scale": 99.0})
    assert config.get_cli_kwargs()["inference_sampler.cfg_scale"] == 2.0

    spec = RFdiffusion3DesignSpec.model_validate({"input_structure": "typed.pdb", "input": "extra.pdb"})
    assert spec.to_dict()["input"] == "typed.pdb"


# ── DesignSpec: JSON spec emission ──────────────────────────────────────────


def test_rfdiffusion3_design_spec_typed_fields_propagate_to_json():
    """Every typed InputSpecification field lands in to_dict() under its upstream key.

    No ``length`` or ``contig`` is set — this also exercises the validator
    accepting specs whose only constraints are typed design fields.
    """
    spec = RFdiffusion3DesignSpec(
        symmetry="c3",
        select_buried="A1-50",
        select_partially_buried="A51-70",
        select_exposed="A71-100",
        select_hbond_donor={"A40": ["NE2"]},
        select_hbond_acceptor={"A45": ["OD1"]},
        redesign_motif_sidechains=True,
        plddt_enhanced=False,
        infer_ori_strategy="hotspots",
        ori_token=[1.0, 2.0, 3.0],
        is_non_loopy=True,
    )
    d = spec.to_dict()
    assert d["symmetry"] == "c3"
    assert d["select_buried"] == "A1-50"
    assert d["select_partially_buried"] == "A51-70"
    assert d["select_exposed"] == "A71-100"
    assert d["select_hbond_donor"] == {"A40": ["NE2"]}
    assert d["select_hbond_acceptor"] == {"A45": ["OD1"]}
    assert d["redesign_motif_sidechains"] is True
    assert d["plddt_enhanced"] is False
    assert d["infer_ori_strategy"] == "hotspots"
    assert d["ori_token"] == [1.0, 2.0, 3.0]
    assert d["is_non_loopy"] is True


def test_rfdiffusion3_design_spec_omits_unset_typed_fields():
    """None-valued typed fields must NOT leak into the JSON spec (preserve upstream defaults)."""
    d = RFdiffusion3DesignSpec(length="100").to_dict()
    for key in (
        "symmetry",
        "select_buried",
        "select_partially_buried",
        "select_exposed",
        "select_hbond_donor",
        "select_hbond_acceptor",
        "redesign_motif_sidechains",
        "plddt_enhanced",
        "infer_ori_strategy",
        "ori_token",
        "is_non_loopy",
    ):
        assert key not in d, f"{key} leaked into spec when None"


def test_rfdiffusion3_design_spec_ori_token_must_be_xyz():
    """ori_token is a 3-element [x, y, z] override — length 2 or 4 must fail."""
    RFdiffusion3DesignSpec(length="100", ori_token=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        RFdiffusion3DesignSpec(length="100", ori_token=[1.0, 2.0])
    with pytest.raises(ValueError):
        RFdiffusion3DesignSpec(length="100", ori_token=[1.0, 2.0, 3.0, 4.0])


# ── Schema flags + cache-key invariants ─────────────────────────────────────


def test_rfdiffusion3_config_schema_visibility_flags():
    """depends_on relationships render correctly in the generated JSON Schema."""
    properties = RFdiffusion3Config.model_json_schema()["properties"]

    cfg_dep = {"field": "use_classifier_free_guidance", "value": [True]}
    for f in ("cfg_scale", "cfg_features", "cfg_t_max"):
        assert properties[f]["x-depends-on"] == cfg_dep, f"{f} missing CFG depends_on"

    assert properties["align_trajectory_structures"]["x-depends-on"] == {
        "field": "dump_trajectories",
        "value": [True],
    }


def test_rfdiffusion3_config_cache_key_invariants():
    """Path / debug-output fields are excluded; n_recycle is included (changes outputs)."""
    base = RFdiffusion3Config().cache_key()

    # Excluded — purely IO / debug knobs
    assert (
        RFdiffusion3Config(
            input_dir="in",
            output_dir="out",
            dump_trajectories=True,
            align_trajectory_structures=True,
            prevalidate_inputs=True,
        ).cache_key()
        == base
    )

    # Included — n_recycle changes the model output
    assert RFdiffusion3Config(n_recycle=5).cache_key() != base


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
