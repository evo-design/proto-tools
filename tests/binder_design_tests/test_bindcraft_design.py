"""Tests for BindCraft binder-design tool."""

import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest

from proto_tools.tools import (
    BindCraftConfig,
    BindCraftDesign,
    BindCraftInput,
    run_bindcraft_design,
)
from proto_tools.tools.binder_design.bindcraft.bindcraft_design import (
    _HARDCODED_INTERNAL_SETTINGS,
    _USER_FACING_UPSTREAM_KEYS,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_FIXTURE_PDB = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
_UPSTREAM_SNAPSHOT = Path(__file__).parent.parent / "dummy_data" / "bindcraft_default_4stage_multimer.json"
_UPSTREAM_INFRA_KEYS = frozenset({"af_params_dir", "dssp_path", "dalphaball_path"})
_UPSTREAM_KNOBS = set(json.loads(_UPSTREAM_SNAPSHOT.read_text())) - _UPSTREAM_INFRA_KEYS

_persistent_tool = make_persistent_fixture("bindcraft")


# ── Validation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("lengths", "match"),
    [((100, 50), "min <= max"), ((0, 100), "positive"), ((-5, 50), "positive")],
)
def test_bindcraft_input_rejects_invalid_length_range(lengths: tuple[int, int], match: str) -> None:
    """``_validate_lengths`` rejects inverted, zero, and negative ranges."""
    with pytest.raises(ValueError, match=match):
        BindCraftInput(target_pdb=str(_FIXTURE_PDB), binder_lengths=lengths)


@pytest.mark.parametrize("value", [0, -1])
def test_max_trajectories_rejects_invalid(value: int) -> None:
    """``max_trajectories`` must be False or a positive int (``True`` bool-coerces to 1, allowed)."""
    with pytest.raises(ValueError, match="max_trajectories"):
        BindCraftConfig(max_trajectories=value)


# ── Upstream fidelity ─────────────────────────────────────────────────────────


def test_user_facing_config_defaults_match_pinned_upstream() -> None:
    """User-facing Config defaults equal the upstream snapshot byte-for-byte."""
    upstream = json.loads(_UPSTREAM_SNAPSHOT.read_text())
    cfg = BindCraftConfig().model_dump()
    # We spell omit_AAs differently; compare it in the upstream form the payload sends.
    cfg["omit_AAs"] = ",".join(cfg.pop("excluded_amino_acids") or [])
    skip = _UPSTREAM_INFRA_KEYS | _HARDCODED_INTERNAL_SETTINGS.keys()
    mismatches = [f"{k}: ours={cfg[k]!r} upstream={v!r}" for k, v in upstream.items() if k not in skip and cfg[k] != v]
    assert not mismatches, "Default-value drift from upstream:\n" + "\n".join(mismatches)


def test_upstream_keys_partition_into_user_facing_or_hardcoded() -> None:
    """Every upstream advanced key is either on the Config or in the hardcoded dict."""
    user_facing = set(_USER_FACING_UPSTREAM_KEYS) | {"omit_AAs"}
    hardcoded = set(_HARDCODED_INTERNAL_SETTINGS)
    assert user_facing.isdisjoint(hardcoded)
    assert user_facing | hardcoded == _UPSTREAM_KNOBS


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
        "proto_tools.tools.binder_design.bindcraft.bindcraft_design.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = BindCraftInput(
        target_pdb=str(_FIXTURE_PDB),
        target_chain="A",
        binder_lengths=(60, 70),
        binder_name="smoke",
    )
    config = BindCraftConfig(
        soft_iterations=10,
        max_trajectories=1,
        filter_overrides={"Average_pLDDT": {"threshold": 0.5}},
    )
    run_bindcraft_design(inputs, config)

    assert captured["toolkit"] == "bindcraft"
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


# ── Standalone filter building (runs in the tool subprocess, not via dispatch) ──


@pytest.fixture(scope="module")
def standalone():
    """Import the bindcraft standalone inference module with ``standalone_helpers`` stubbed.

    The module's helpers are only needed for live dispatch; ``_build_filters``
    exercised here never touches them.
    """
    stub = types.ModuleType("standalone_helpers")
    stub.get_jax_memory_stats = lambda **_: {}  # type: ignore[attr-defined]
    stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
    stub.get_subprocess_device_env = lambda _device: dict(os.environ)  # type: ignore[attr-defined]
    stub.resolve_weights_dir = lambda _toolkit: "weights"  # type: ignore[attr-defined]
    saved = sys.modules.get("standalone_helpers")
    sys.modules["standalone_helpers"] = stub
    # Module-level guard only needs a truthy value; _build_filters never reads it.
    os.environ.setdefault("VENV_PATH", str(Path(__file__).parent / "_fake_venv"))
    inference_path = (
        Path(__file__).parent.parent.parent / "proto_tools/tools/binder_design/bindcraft/standalone/inference.py"
    )
    spec = importlib.util.spec_from_file_location("bindcraft_inference", inference_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    if saved is not None:
        sys.modules["standalone_helpers"] = saved
    else:
        sys.modules.pop("standalone_helpers", None)


def test_filter_overrides_replace_whole_entries_and_keep_other_defaults(standalone) -> None:  # type: ignore[no-untyped-def]
    """An override replaces its metric's entry outright; every other default filter survives.

    Upstream's filter check reads ``higher`` unconditionally, so a caller overriding one
    metric must pass both keys — a ``threshold``-only override drops ``higher``.
    """
    defaults = standalone._build_filters({})
    assert defaults["Average_pLDDT"] == {"threshold": 0.8, "higher": True}

    filters = standalone._build_filters({"filter_overrides": {"Average_pLDDT": {"threshold": 0.5, "higher": True}}})
    assert filters["Average_pLDDT"] == {"threshold": 0.5, "higher": True}
    # Untouched metrics keep their defaults rather than being dropped by the merge.
    assert {k: v for k, v in filters.items() if k != "Average_pLDDT"} == {
        k: v for k, v in defaults.items() if k != "Average_pLDDT"
    }

    partial = standalone._build_filters({"filter_overrides": {"Average_pLDDT": {"threshold": 0.5}}})
    assert partial["Average_pLDDT"] == {"threshold": 0.5}, "override replaces the entry; it does not merge into it"


# ── Tool registration ────────────────────────────────────────────────────────


def test_bindcraft_design_registered_with_expected_metadata() -> None:
    """``@tool`` wires the right registry metadata + a callable example_input."""
    spec = next((s for s in ToolRegistry.list_all() if s.key == "bindcraft-design"), None)
    assert spec is not None
    assert spec.category == "binder_design"
    assert spec.uses_gpu is True
    assert spec.cacheable is False
    assert spec.device_count == "1"
    example = spec.example_input()
    assert isinstance(example, BindCraftInput)
    assert example.number_of_final_designs == 1, "example_input should be a one-off, not a multi-day run"


# ── Integration: real-world end-to-end ────────────────────────────────────────
# Each test below runs the full hallucination + MPNN + AF2 + PyRosetta pipeline.
# Cost: ~10-30 min on H100. CPU-skipped automatically.


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_bindcraft_one_off_sample_full_pipeline() -> None:
    """End-to-end one-off sample on renin at minimum trajectory budget."""
    inputs = BindCraftInput(
        target_pdb=str(_FIXTURE_PDB),
        target_chain="A",
        target_hotspot_residues="56",
        binder_lengths=(60, 70),
        binder_name="oneoff",
        number_of_final_designs=1,
    )
    config = BindCraftConfig(
        max_trajectories=1,
        soft_iterations=10,
        temporary_iterations=5,
        hard_iterations=2,
        greedy_iterations=2,
        num_seqs=2,
        max_mpnn_sequences=1,
        seed=42,
    )
    result = run_bindcraft_design(inputs, config)
    validate_output(result)

    assert result.tool_id == "bindcraft-design"
    assert result.n_trajectories_run >= 1
    assert result.n_designs_accepted == len(result.designs)
    # Acceptance is stochastic at 1 trajectory; only assert on what runs through.
    for design in result.designs:
        assert isinstance(design, BindCraftDesign)
        assert 60 <= len(design.binder_sequence) <= 70
        assert design.structure is not None
        assert design.metrics.primary_value is not None
    assert_metrics_in_spec(result)
