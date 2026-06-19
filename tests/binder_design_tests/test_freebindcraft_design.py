"""Tests for the FreeBindCraft (PyRosetta-free) binder-design tool."""

import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest

from proto_tools.tools import (
    FreeBindCraftConfig,
    FreeBindCraftDesign,
    FreeBindCraftInput,
    FreeBindCraftMetrics,
    run_freebindcraft_design,
)
from proto_tools.tools.binder_design.freebindcraft.freebindcraft_design import (
    _HARDCODED_INTERNAL_SETTINGS,
    _USER_FACING_UPSTREAM_KEYS,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_FIXTURE_PDB = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
# FreeBindCraft is a drop-in fork: its hallucination/MPNN settings match upstream BindCraft byte-for-byte.
_UPSTREAM_SNAPSHOT = Path(__file__).parent.parent / "dummy_data" / "bindcraft_default_4stage_multimer.json"
_UPSTREAM_INFRA_KEYS = frozenset({"af_params_dir", "dssp_path", "dalphaball_path"})
_UPSTREAM_KNOBS = set(json.loads(_UPSTREAM_SNAPSHOT.read_text())) - _UPSTREAM_INFRA_KEYS

# PyRosetta-only metrics FreeBindCraft emits as placeholders — must never be surfaced as real.
_PLACEHOLDER_METRICS = frozenset(
    {
        "dG",
        "dG_per_dSASA",
        "binder_energy_score",
        "packstat",
        "n_interface_hbonds",
        "interface_hbonds_pct",
        "n_interface_unsat_hbonds",
        "interface_unsat_hbonds_pct",
    }
)

_persistent_tool = make_persistent_fixture("freebindcraft")


# ── Validation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("lengths", "match"),
    [((100, 50), "min <= max"), ((0, 100), "positive"), ((-5, 50), "positive")],
)
def test_freebindcraft_input_rejects_invalid_length_range(lengths: tuple[int, int], match: str) -> None:
    """``_validate_lengths`` rejects inverted, zero, and negative ranges."""
    with pytest.raises(ValueError, match=match):
        FreeBindCraftInput(target_pdb=str(_FIXTURE_PDB), binder_lengths=lengths)


@pytest.mark.parametrize("value", [0, -1])
def test_max_trajectories_rejects_invalid(value: int) -> None:
    """``max_trajectories`` must be False (unlimited) or a positive int; 0 and negatives are rejected."""
    with pytest.raises(ValueError, match="max_trajectories"):
        FreeBindCraftConfig(max_trajectories=value)


def test_max_trajectories_true_coerces_to_one() -> None:
    """``max_trajectories=True`` collapses to int 1; ``False`` stays unlimited."""
    assert FreeBindCraftConfig(max_trajectories=True).max_trajectories == 1
    assert FreeBindCraftConfig(max_trajectories=False).max_trajectories is False


# ── Upstream fidelity ─────────────────────────────────────────────────────────


def test_user_facing_config_defaults_match_pinned_upstream() -> None:
    """User-facing Config defaults equal the upstream BindCraft snapshot byte-for-byte."""
    upstream = json.loads(_UPSTREAM_SNAPSHOT.read_text())
    cfg = FreeBindCraftConfig().model_dump()
    skip = _UPSTREAM_INFRA_KEYS | _HARDCODED_INTERNAL_SETTINGS.keys()
    mismatches = [f"{k}: ours={cfg[k]!r} upstream={v!r}" for k, v in upstream.items() if k not in skip and cfg[k] != v]
    assert not mismatches, "Default-value drift from upstream:\n" + "\n".join(mismatches)


def test_upstream_keys_partition_into_user_facing_or_hardcoded() -> None:
    """Every upstream advanced key is either on the Config or in the hardcoded dict."""
    user_facing = set(_USER_FACING_UPSTREAM_KEYS)
    hardcoded = set(_HARDCODED_INTERNAL_SETTINGS)
    assert user_facing.isdisjoint(hardcoded)
    assert user_facing | hardcoded == _UPSTREAM_KNOBS


# ── PyRosetta-free metric honesty ─────────────────────────────────────────────


def test_metric_spec_excludes_pyrosetta_placeholder_metrics() -> None:
    """PyRosetta-only placeholder metrics are not exposed in the typed metric spec."""
    spec_keys = set(FreeBindCraftMetrics.metric_spec)
    leaked = spec_keys & _PLACEHOLDER_METRICS
    assert not leaked, f"PyRosetta-free output must not surface placeholder metrics: {sorted(leaked)}"
    # The headline ranking metric is an AlphaFold2 confidence value, not a Rosetta energy.
    assert FreeBindCraftMetrics().primary_metric == "avg_iptm"
    assert "shape_complementarity" in spec_keys  # real (sc-rs), kept


# ── Dispatch payload (mocked) ────────────────────────────────────────────────


def test_dispatch_payload_carries_user_config_plus_hardcoded_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch payload merges user-set Config fields + hardcoded internals + filter_overrides."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, **kwargs):  # type: ignore[no-untyped-def]
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        # target_pdb is a tempfile that the wrapper cleans up after dispatch returns;
        # snapshot its content here while it still exists.
        captured["target_pdb_content"] = Path(payload["target_pdb"]).read_text()
        return {"designs": [], "n_trajectories_run": 0, "n_designs_accepted": 0}

    monkeypatch.setattr(
        "proto_tools.tools.binder_design.freebindcraft.freebindcraft_design.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = FreeBindCraftInput(
        target_pdb=str(_FIXTURE_PDB),
        target_chain="A",
        binder_lengths=(60, 70),
        binder_name="smoke",
    )
    config = FreeBindCraftConfig(
        soft_iterations=10,
        max_trajectories=1,
        filter_overrides={"Average_pLDDT": {"threshold": 0.5}},
    )
    run_freebindcraft_design(inputs, config)

    assert captured["toolkit"] == "freebindcraft"
    payload = captured["payload"]
    assert payload["operation"] == "design"
    # target_pdb is materialized to a tempfile by the wrapper; assert content shape
    # rather than the original input path.
    assert payload["target_pdb"].endswith(".pdb")
    assert "ATOM" in captured["target_pdb_content"]
    assert payload["binder_lengths"] == [60, 70]
    assert payload["filter_overrides"] == {"Average_pLDDT": {"threshold": 0.5}}

    advanced = payload["advanced_settings"]
    assert set(advanced) == _UPSTREAM_KNOBS, "advanced_settings missing or extra upstream keys"
    assert advanced["soft_iterations"] == 10, "user-set value didn't flow through"
    for k, v in _HARDCODED_INTERNAL_SETTINGS.items():
        assert advanced[k] == v, f"hardcoded {k!r} not merged into advanced_settings"


# ── Standalone output parsers (run in the tool subprocess, not via dispatch) ──


@pytest.fixture(scope="module")
def standalone():
    """Import the freebindcraft standalone inference module with ``standalone_helpers`` stubbed.

    The module's helpers are only needed for live dispatch; the pure parsers exercised
    here (``_build_filters``, ``_parse_outputs``) don't touch them.
    """
    stub = types.ModuleType("standalone_helpers")
    stub.get_jax_memory_stats = lambda **_: {}  # type: ignore[attr-defined]
    stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
    stub.get_subprocess_device_env = lambda _device: dict(os.environ)  # type: ignore[attr-defined]
    stub.resolve_weights_dir = lambda _toolkit: "weights"  # type: ignore[attr-defined]  # unused by the parsers under test
    saved = sys.modules.get("standalone_helpers")
    sys.modules["standalone_helpers"] = stub
    # Module-level guard only needs a truthy value; the parsers tested here never read it.
    os.environ.setdefault("VENV_PATH", str(Path(__file__).parent / "_fake_venv"))
    inference_path = (
        Path(__file__).parent.parent.parent / "proto_tools/tools/binder_design/freebindcraft/standalone/inference.py"
    )
    spec = importlib.util.spec_from_file_location("freebindcraft_inference", inference_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if saved is not None:
        sys.modules["standalone_helpers"] = saved
    else:
        sys.modules.pop("standalone_helpers", None)


def test_filter_overrides_merge_preserves_default_keys(standalone) -> None:  # type: ignore[no-untyped-def]
    """A partial filter override merges per-metric, keeping the default ``higher`` key.

    Guards the runtime foot-gun the mocked dispatch test can't see: a full replace would
    drop ``higher``, which upstream's filter check reads unconditionally.
    """
    filters = standalone._build_filters({"filter_overrides": {"Average_pLDDT": {"threshold": 0.5}}})
    assert filters["Average_pLDDT"]["threshold"] == 0.5
    assert filters["Average_pLDDT"]["higher"] is True


def test_parse_outputs_surfaces_early_stop_accepted_designs(standalone, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Budget-capped runs (no final_design_stats.csv) still return Accepted/ binders via mpnn stats."""
    (tmp_path / "Accepted").mkdir()
    (tmp_path / "Accepted" / "binder_l60_s7_mpnn1_model2.pdb").write_text("ATOM\n")
    (tmp_path / "mpnn_design_stats.csv").write_text(
        "Design,Sequence,Seed,Average_pLDDT,Average_ipSAE,Average_Unrelaxed_Clashes\n"
        "binder_l60_s7_mpnn1,MKLV,7,0.88,0.66,5\n"
    )
    (tmp_path / "trajectory_stats.csv").write_text("header\nrow1\n")

    out = standalone._parse_outputs(tmp_path)

    assert out["n_designs_accepted"] == 1
    design = out["designs"][0]
    assert design["design_name"] == "binder_l60_s7_mpnn1"
    assert design["binder_sequence"] == "MKLV"
    assert design["seed"] == 7
    assert design["metrics"]["avg_ipsae"] == 0.66
    assert design["metrics"]["unrelaxed_clashes"] == 5.0


# ── Tool registration ────────────────────────────────────────────────────────


def test_freebindcraft_design_registered_with_expected_metadata() -> None:
    """``@tool`` wires the right registry metadata + a callable example_input."""
    spec = next((s for s in ToolRegistry.list_all() if s.key == "freebindcraft-design"), None)
    assert spec is not None
    assert spec.category == "binder_design"
    assert spec.uses_gpu is True
    assert spec.cacheable is False
    assert spec.device_count == "1"
    example = spec.example_input()
    assert isinstance(example, FreeBindCraftInput)
    assert example.number_of_final_designs == 1, "example_input should be a one-off, not a multi-day run"


# ── Integration: real-world end-to-end ────────────────────────────────────────
# Each test runs the full hallucination + MPNN + AF2 + OpenMM/FreeSASA pipeline,
# PyRosetta-free. Cost: ~10-30 min on H100. CPU-skipped automatically.


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_freebindcraft_one_off_sample_full_pipeline() -> None:
    """End-to-end one-off sample on renin at minimum trajectory budget."""
    inputs = FreeBindCraftInput(
        target_pdb=str(_FIXTURE_PDB),
        target_chain="A",
        target_hotspot_residues="56",
        binder_lengths=(60, 70),
        binder_name="oneoff",
        number_of_final_designs=1,
    )
    config = FreeBindCraftConfig(
        max_trajectories=1,
        soft_iterations=10,
        temporary_iterations=5,
        hard_iterations=2,
        greedy_iterations=2,
        num_seqs=2,
        max_mpnn_sequences=1,
        seed=42,
    )
    result = run_freebindcraft_design(inputs, config)
    validate_output(result)

    assert result.tool_id == "freebindcraft-design"
    assert result.n_trajectories_run >= 1
    assert result.n_designs_accepted == len(result.designs)
    # Acceptance is stochastic at 1 trajectory; only assert on what runs through.
    for design in result.designs:
        assert isinstance(design, FreeBindCraftDesign)
        assert 60 <= len(design.binder_sequence) <= 70
        assert design.structure is not None
        assert design.metrics.primary_value is not None
        # PyRosetta-free run must never report Rosetta-only placeholder metrics as real.
        assert _PLACEHOLDER_METRICS.isdisjoint(dict(design.metrics.items()))
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_freebindcraft_filter_override_rejects_with_impossible_threshold() -> None:
    """``filter_overrides`` flows into the dispatch payload and reaches upstream's filter check."""
    inputs = FreeBindCraftInput(
        target_pdb=str(_FIXTURE_PDB),
        target_chain="A",
        target_hotspot_residues="56",
        binder_lengths=(60, 70),
        binder_name="strict",
        number_of_final_designs=1,
    )
    config = FreeBindCraftConfig(
        max_trajectories=1,
        soft_iterations=10,
        temporary_iterations=5,
        hard_iterations=2,
        greedy_iterations=2,
        num_seqs=2,
        max_mpnn_sequences=1,
        seed=42,
        filter_overrides={"Average_pLDDT": {"threshold": 0.999, "higher": True}},
    )
    result = run_freebindcraft_design(inputs, config)
    validate_output(result)

    assert result.n_trajectories_run >= 1
    assert result.n_designs_accepted == 0, "Impossible pLDDT threshold should reject every design"
    assert result.designs == []
