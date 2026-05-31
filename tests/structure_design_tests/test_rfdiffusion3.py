"""tests/structure_design_tests/test_rfdiffusion3.py.

Tests for RFdiffusion3.
"""

import json
import subprocess
from importlib import util
from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_design import (
    RFdiffusion3Config,
    RFdiffusion3DesignSpec,
    RFdiffusion3Input,
    run_rfdiffusion3,
)
from tests._structure_fixtures import synthetic_cif
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


def test_rfdiffusion3_design_spec_selections_require_input_structure():
    """contig, unindex, and select_* fields all require ``input_structure``."""
    pattern = r"require 'input_structure'"

    # select_* with only length → reject (length doesn't establish an atom array).
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(length="100", select_hotspots="A24,A35")
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(length="100", select_hbond_donor={"A40": ["NE2"]})

    # contig alone → reject (upstream rfd3 needs atoms to resolve contig against).
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(contig="A1-100")
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(contig="50-80")

    # contig + select_* without input → reject (contig is not an atom source).
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(contig="A1-100", select_hotspots="A24")
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(contig="50-80", select_hbond_donor={"A40": ["NE2"]})

    # unindex without input → reject (same upstream rule as contig and select_*).
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3DesignSpec(unindex="A244,A274")

    # Happy path 1: input_structure alone with select_*
    spec = RFdiffusion3DesignSpec(input_structure=synthetic_cif(["A"]), select_hotspots="A24,A35,A50")
    assert spec.select_hotspots == "A24,A35,A50"

    # Happy path 2: input_structure + contig + select_* (full motif scaffolding shape)
    spec = RFdiffusion3DesignSpec(
        input_structure=synthetic_cif(["A"]),
        contig="A1-100",
        select_hbond_donor={"A40": ["NE2"]},
    )
    assert spec.contig == "A1-100"
    assert spec.select_hbond_donor == {"A40": ["NE2"]}

    # Happy path 3: length alone (unconditional design — the only no-input path).
    spec = RFdiffusion3DesignSpec(length="40")
    assert spec.length == "40"


def test_rfdiffusion3_config_gamma_0_symmetry_constraint():
    """gamma_0 must be > 0.5 when sampler_kind='symmetry' (upstream rfd3 asserts this)."""
    pattern = r"gamma_0 must be > 0\.5 when sampler_kind='symmetry'"

    # Reject: low gamma_0 with symmetry sampler, including the strict boundary at 0.5.
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3Config(sampler_kind="symmetry", gamma_0=0.35)
    with pytest.raises(ValueError, match=pattern):
        RFdiffusion3Config(sampler_kind="symmetry", gamma_0=0.5)

    # Happy path: default gamma_0=0.6 is fine with symmetry.
    assert RFdiffusion3Config(sampler_kind="symmetry").gamma_0 == 0.6

    # Happy path: low gamma_0 is fine with the default sampler (constraint is symmetry-specific).
    assert RFdiffusion3Config(sampler_kind="default", gamma_0=0.1).gamma_0 == 0.1


# ── Config: CLI kwargs assembly ─────────────────────────────────────────────


def test_rfdiffusion3_config_get_cli_kwargs():
    """Typed sampler knobs emit under inference_sampler.* dotted paths; top-level toggles stay flat.

    Niche sampler internals (noise schedule, motif realignment, CFG sub-features) are not typed
    fields; they reach the CLI only via the inference_sampler.* passthrough (model_extra).
    """
    config = RFdiffusion3Config(
        sampler_kind="symmetry",
        cfg_scale=2.0,
        gamma_0=0.7,
        center_option="motif",
        use_classifier_free_guidance=True,
        dump_trajectories=True,
        align_trajectory_structures=True,
        prevalidate_inputs=True,
    )
    kwargs = config.get_cli_kwargs()

    # Typed sampler knobs use dotted Hydra paths.
    expected_dotted = {
        "inference_sampler.cfg_scale": 2.0,
        "inference_sampler.gamma_0": 0.7,
        "inference_sampler.kind": "symmetry",
        "inference_sampler.center_option": "motif",
        "inference_sampler.use_classifier_free_guidance": True,
    }
    for key, value in expected_dotted.items():
        assert kwargs[key] == value, f"{key} not emitted as dotted path"

    # Top-level toggles stay flat.
    assert kwargs["dump_trajectories"] is True
    assert kwargs["align_trajectory_structures"] is True
    assert kwargs["prevalidate_inputs"] is True

    # Typed sampler keys MUST NOT also appear flat — Hydra would silently ignore them.
    for flat in ("cfg_scale", "gamma_0", "kind", "center_option", "use_classifier_free_guidance"):
        assert flat not in kwargs

    # Demoted sampler internals stay reachable only via the inference_sampler.* passthrough.
    passthrough = RFdiffusion3Config(**{"inference_sampler.noise_scale": 1.003}).get_cli_kwargs()
    assert passthrough["inference_sampler.noise_scale"] == 1.003


def test_rfdiffusion3_typed_fields_override_extras_on_collision(tmp_path):
    """Both Config.get_cli_kwargs and DesignSpec.to_dict resolve typed-vs-extras collisions for typed."""
    config = RFdiffusion3Config(cfg_scale=2.0, **{"inference_sampler.cfg_scale": 99.0})
    assert config.get_cli_kwargs()["inference_sampler.cfg_scale"] == 2.0

    target = tmp_path / "typed.pdb"
    target.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n", encoding="utf-8")
    spec = RFdiffusion3DesignSpec.model_validate({"input_structure": str(target), "input": "extra.pdb"})
    # input_structure is materialized to a real file path by to_json_spec, overriding the extra "input".
    emitted = json.loads(RFdiffusion3Input(design_specs=[spec]).to_json_spec(input_dir=tmp_path))["spec-0"]
    assert emitted["input"] != "extra.pdb"
    assert Path(emitted["input"]).is_file()
    assert Path(emitted["input"]).read_text() == target.read_text()


# ── DesignSpec: JSON spec emission ──────────────────────────────────────────


def test_rfdiffusion3_design_spec_typed_fields_propagate_to_json():
    """Every typed InputSpecification field lands in to_dict() under its upstream key."""
    pdb_content = "ATOM      1  N   ALA A   1       0.000   0.000   0.000\nEND\n"

    spec = RFdiffusion3DesignSpec(
        input_structure=pdb_content,
        contig="A1-100",
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
    assert "input" not in d  # input_structure is materialized to a path by to_json_spec, not inlined in to_dict
    assert d["symmetry"] == {"id": "C3"}  # string normalized to SymmetryConfig dict
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


def test_rfdiffusion3_design_spec_rejects_literal_nul_chain_break():
    """RFdiffusion3 chain breaks use /0, never a Python literal NUL byte."""
    with pytest.raises(ValueError, match="/0"):
        RFdiffusion3DesignSpec(input_structure=synthetic_cif(["A"]), contig="50,\0,A1-10")


def test_rfdiffusion3_design_spec_rejects_backslash_zero_chain_break():
    r"""RFdiffusion3 chain breaks use /0, not the text token "\\0"."""
    with pytest.raises(ValueError, match="/0"):
        RFdiffusion3DesignSpec(input_structure=synthetic_cif(["A"]), contig=r"50,\0,A1-10")


def test_rfdiffusion3_design_spec_symmetry_serializes_to_id_dict():
    """A group-id str wraps to {"id": "<UPPER>"} (the rfd3 SymmetryConfig); a dict passes through."""
    assert RFdiffusion3DesignSpec(length="100", symmetry="c3").to_dict()["symmetry"] == {"id": "C3"}
    assert RFdiffusion3DesignSpec(length="100", symmetry="D2").to_dict()["symmetry"] == {"id": "D2"}
    # dict passes through unchanged (advanced SymmetryConfig)
    assert RFdiffusion3DesignSpec(length="100", symmetry={"id": "C3"}).to_dict()["symmetry"] == {"id": "C3"}
    rich = {"id": "C5", "extra": 1}
    assert RFdiffusion3DesignSpec(length="100", symmetry=rich).to_dict()["symmetry"] == rich


# ── Cache-key invariants ────────────────────────────────────────────────────


def test_rfdiffusion3_config_cache_key_invariants():
    """Path / debug-output fields are excluded; sampler knobs are included (change outputs)."""
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

    # Included — num_timesteps changes the model output
    assert RFdiffusion3Config(num_timesteps=50).cache_key() != base


# ── JSON spec generation ────────────────────────────────────────────────────


def test_rfdiffusion3_json_spec_generation(tmp_path, monkeypatch):
    """Multiple design specs produce keyed JSON with correct fields."""
    target = tmp_path / "target.pdb"
    target.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    inputs = RFdiffusion3Input(
        design_specs=[
            RFdiffusion3DesignSpec(length="100"),
            RFdiffusion3DesignSpec(input_structure="target.pdb", contig="50-80"),
        ]
    )

    spec = json.loads(inputs.to_json_spec(input_dir=tmp_path))

    assert "spec-0" in spec
    assert "spec-1" in spec
    assert spec["spec-0"]["length"] == "100"
    assert spec["spec-1"]["contig"] == "50-80"
    # input_structure is written to a real file under input_dir; "input" is that path (rfd3 needs a path).
    input_path = Path(spec["spec-1"]["input"])
    assert input_path.is_file() and input_path.suffix == ".pdb"
    assert input_path.read_text() == target.read_text()
    assert "\n" not in spec["spec-1"]["input"]  # a path, never inline multi-line content


def test_rfdiffusion3_input_structure_accepts_structure_object(tmp_path):
    """A Structure object is materialized to a file; "input" is that path (rfd3 needs a path, not content)."""
    structure = Structure(structure=synthetic_cif(["A"]))
    inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(input_structure=structure, length="40")])
    spec = json.loads(inputs.to_json_spec(input_dir=tmp_path))
    input_path = Path(spec["spec-0"]["input"])
    assert input_path.is_file() and input_path.suffix == ".cif"  # synthetic_cif → cif format
    assert input_path.read_text() == structure.structure
    assert spec["spec-0"]["input"] != structure.structure  # a path, not inline content


def test_rfdiffusion3_raw_json_resolves_relative_input_paths(tmp_path, monkeypatch):
    target = tmp_path / "target.cif"
    target.write_text("data_target\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    inputs = RFdiffusion3Input(raw_json=json.dumps({"spec-0": {"input": "target.cif", "contig": "50-80"}}))

    spec = json.loads(inputs.to_json_spec())

    assert spec["spec-0"]["input"] == str(target)


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


def test_rfdiffusion3_output_chain_ids_canonicalized(monkeypatch):
    """Symmetric outputs carry transformation-suffixed chain IDs; normalize to positional.

    rfd3 suffixes chain IDs with the transformation index for multi-transformation
    (symmetric) designs (``A1, A2, A3`` for a C3 trimer); the wrapper remaps these
    to positional labels by emission order.
    """
    from proto_tools.tools.structure_design.rfdiffusion3 import rfdiffusion3_sample as mod

    def fake(tool_id, input_data, **kw):
        return {
            "designs": [
                {
                    "structure_content": synthetic_cif(["A1", "A2", "A3"]),
                    "spec_key": "spec-0",
                    "design_index": 0,
                    "metadata": {},
                }
            ]
        }

    monkeypatch.setattr(mod.ToolInstance, "dispatch", fake)
    output = run_rfdiffusion3(
        RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="40")]), RFdiffusion3Config()
    )

    design = output.designed_structures[0].structures[0]
    assert design.structure.get_chain_ids() == ["A", "B", "C"]
    # complex is the primary downstream interface — it must carry the canonicalized IDs too.
    assert [c.id for c in design.complex.chains] == ["A", "B", "C"]


def test_rfdiffusion3_standalone_failure_reports_full_subprocess_output(monkeypatch, tmp_path):
    """Subprocess failures should preserve the full captured error, not only a tail."""
    standalone_dir = Path(__file__).resolve().parents[2] / "proto_tools/tools/structure_design/rfdiffusion3/standalone"
    monkeypatch.syspath_prepend(str(standalone_dir))
    spec = util.spec_from_file_location("rfdiffusion3_standalone_inference", standalone_dir / "inference.py")
    mod = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    model = mod.RFdiffusion3Model()
    model._loaded = True
    model.rfd3_executable = "rfd3"
    stderr = "\n".join(f"stderr line {i}" for i in range(20))
    stdout = "\n".join(f"stdout line {i}" for i in range(20))

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=args[0],
            stderr=stderr,
            output=stdout,
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        model(
            input_json_path=str(tmp_path / "input.json"),
            output_dir=str(tmp_path / "out"),
            device="cpu",
        )

    message = str(excinfo.value)
    assert "stderr line 0" in message
    assert "stderr line 19" in message
    assert "stdout line 0" in message
    assert "stdout line 19" in message


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_rfdiffusion3_unconditional_design():
    """Basic unconditional design produces a valid 2-chain structure (C2 symmetry)."""
    # C2 symmetry yields 2 chains, exercising the per-chain list[str] output shape.
    # sampler_kind="symmetry" is required upstream when the spec sets a symmetry group.
    inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="40", symmetry="c2")])
    config = RFdiffusion3Config(
        n_batches=1,
        diffusion_batch_size=1,
        num_timesteps=50,
        sampler_kind="symmetry",
    )

    output = run_rfdiffusion3(inputs, config)

    # One bundle per input spec (this test passes a single spec).
    assert len(output.designed_structures) == 1
    bundle = output[0]
    assert bundle.spec_key == "spec-0"
    assert len(bundle.structures) > 0
    first_design = bundle[0]
    assert first_design.structure is not None
    # C2 symmetry → 2 designed chains of 40 residues each.
    assert len(first_design.complex.chains) == 2
    assert all(len(c.sequence) == 40 for c in first_design.complex.chains)


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
    # Single input spec → single bundle holding all 8 designs.
    assert len(result.designed_structures) == 1
    bundle = result[0]
    assert len(bundle.structures) == 8
    for design in bundle.structures:
        assert design.structure is not None
        # Single-chain unconditional design → one designed chain of 128 residues.
        assert len(design.complex.chains) == 1
        assert len(design.complex.chains[0].sequence) == 128
