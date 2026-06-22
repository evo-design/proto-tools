"""tests/sequence_scoring_tests/test_deeppbs_specificity.py.

Tests for DeepPBS specificity tool wrapper, canonicalization, and fallback.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

import proto_tools.utils.standalone_helpers_source as _shs
from proto_tools.tools.sequence_scoring.deeppbs_specificity import (
    DeepPBSSpecificityConfig,
    DeepPBSSpecificityInput,
    run_deeppbs_specificity,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.tool_infra_tests.test_export_functionality import validate_output

# The standalone run.py imports the worker-injected ``standalone_helpers`` package,
# which only sits on sys.path inside a tool venv. Add its source dir so the pure
# canonicalization/runner logic can be imported and unit-tested on the host.
sys.path.insert(0, str(Path(next(iter(_shs.__path__)))))
_run = importlib.import_module("proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run")

# ── Registration ──────────────────────────────────────────────────────────────


def test_tool_key_registered():
    """Tool key is visible in ToolRegistry under sequence_scoring."""
    spec = ToolRegistry.get("deeppbs-specificity")
    assert spec.key == "deeppbs-specificity"
    assert spec.category == "sequence_scoring"


# ── Validation ────────────────────────────────────────────────────────────────


def test_input_accepts_single_path_string():
    """Single string path is normalized to a list."""
    payload = DeepPBSSpecificityInput(pdb_paths="one.pdb")
    assert payload.pdb_paths == ["one.pdb"]


def test_input_rejects_empty_list():
    """Empty pdb_paths is rejected."""
    with pytest.raises(ValidationError, match="pdb_paths must contain at least one path"):
        DeepPBSSpecificityInput(pdb_paths=[])


def test_input_rejects_extra_fields():
    """Unknown input fields are rejected."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DeepPBSSpecificityInput(pdb_paths=["one.pdb"], extra_field="x")


def test_config_exposes_no_path_fields():
    """Repo/X3DNA/config paths are auto-provisioned into the cache, so no path field is exposed."""
    path_fields = {
        "deeppbs_repo_path",
        "process_config_path",
        "prediction_config_path",
        "x3dna_bin_path",
        "x3dna_home",
    }
    assert not (path_fields & set(DeepPBSSpecificityConfig.model_fields))
    assert DeepPBSSpecificityConfig().allow_fallback is False


def test_config_cloud_unsupported():
    """DeepPBS cannot run on device='cloud' (needs local repo/X3DNA)."""
    assert DeepPBSSpecificityConfig().cloud_unsupported_reason() is not None


# ── Dispatch (mocked) ─────────────────────────────────────────────────────────


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_run_calls_dispatch_and_returns_output(mock_dispatch):
    """run_deeppbs_specificity returns canonicalized results."""
    mock_dispatch.return_value = {
        "results": [
            {
                "input_name": "candidate_0",
                "source_method": "deeppbs",
                "output_npz_path": "candidate_0.npz",
                "predicted_ppm": [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]],
                "true_sequence": [0, 1],
                "mask": [1, 1],
                "dna_mask": [1, 1],
                "chain_labels": [0, 1],
                "used_fallback": False,
                "fallback_reason": None,
            }
        ]
    }

    output = run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
        DeepPBSSpecificityConfig(),
    )
    validate_output(output)

    assert len(output.results) == 1
    assert output.results[0].source_method == "deeppbs"
    assert output.results[0].true_sequence == [0, 1]
    assert output.results[0].used_fallback is False
    mock_dispatch.assert_called_once()


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_run_threads_device_and_fallback_flag_into_payload(mock_dispatch):
    """The wrapper forwards config.device and allow_fallback in the dispatch payload (cpu-bug regression)."""
    mock_dispatch.return_value = {
        "results": [
            {
                "input_name": "candidate_0",
                "source_method": "deeppbs",
                "output_npz_path": "candidate_0.npz",
                "predicted_ppm": [[0.25, 0.25, 0.25, 0.25]],
                "true_sequence": [0],
                "mask": [1],
                "dna_mask": [1],
                "chain_labels": [0],
                "used_fallback": False,
                "fallback_reason": None,
            }
        ]
    }

    run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
        DeepPBSSpecificityConfig(device="cpu"),
    )
    payload = mock_dispatch.call_args.args[1]
    assert payload["device"] == "cpu"
    assert payload["allow_fallback"] is False


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_run_surfaces_fallback_fields(mock_dispatch):
    """Fallback results round-trip used_fallback / fallback_reason through the wrapper."""
    mock_dispatch.return_value = {
        "results": [
            {
                "input_name": "candidate_0",
                "source_method": "deeppbs",
                "output_npz_path": "candidate_0.npz",
                "predicted_ppm": [[0.25, 0.25, 0.25, 0.25]],
                "true_sequence": [0],
                "mask": [1],
                "dna_mask": [1],
                "chain_labels": [0],
                "used_fallback": True,
                "fallback_reason": "DeepPBS dependency missing: x3dna-dssr not found in PATH.",
            }
        ]
    }

    output = run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
        DeepPBSSpecificityConfig(),
    )
    validate_output(output)
    assert output.results[0].used_fallback is True
    assert "x3dna-dssr" in output.results[0].fallback_reason


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_dispatch_errors_propagate(mock_dispatch):
    """Dispatch failures propagate (decorator owns error policy; raises by default)."""
    mock_dispatch.side_effect = RuntimeError("DeepPBS failed")
    with pytest.raises(RuntimeError, match="DeepPBS failed"):
        run_deeppbs_specificity(
            DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
            DeepPBSSpecificityConfig(),
        )


# ── Canonicalization (pure standalone logic) ──────────────────────────────────


def test_canonicalize_prediction_transforms_to_schema(tmp_path):
    """Raw DeepPBS arrays are converted to canonical DNA schema."""
    from proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run import (
        _canonicalize_prediction,
    )

    raw_path = tmp_path / "raw_predict.npz"
    out_path = tmp_path / "canonical.npz"

    pred = np.array([[0.70, 0.10, 0.10, 0.10], [0.10, 0.70, 0.10, 0.10]], dtype=np.float64)
    seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    np.savez_compressed(raw_path, P=pred, Seq=seq)

    payload = _canonicalize_prediction(str(raw_path), str(out_path))

    # Concatenates forward + mirrored strand, so length doubles.
    assert len(payload["predicted_ppm"]) == 4
    assert len(payload["predicted_ppm"][0]) == 4
    assert len(payload["true_sequence"]) == 4
    assert payload["mask"] == [1, 1, 1, 1]
    assert payload["dna_mask"] == [1, 1, 1, 1]
    assert out_path.exists()
    # Each row of the PPM is normalized to sum to 1.
    for row in payload["predicted_ppm"]:
        assert abs(sum(row) - 1.0) < 1e-9


# ── Fallback (pure standalone logic) ──────────────────────────────────────────


def test_fallback_canonical_from_pdb_builds_uniform_ppm(tmp_path):
    """Fallback derives a uniform PPM and DNA truth from PDB DNA residues."""
    from proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run import (
        _fallback_canonical_from_pdb,
    )

    pdb_path = tmp_path / "complex.pdb"
    # Two DNA residues (DA, DC) on chain B; a protein residue (ALA) is ignored.
    pdb_path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  P    DA B   1       1.000   1.000   1.000  1.00  0.00           P\n"
        "ATOM      3  P    DC B   2       2.000   2.000   2.000  1.00  0.00           P\n"
    )
    out_path = tmp_path / "fallback.npz"

    payload = _fallback_canonical_from_pdb(str(out_path), str(pdb_path))

    assert payload["true_sequence"] == [0, 1]  # DA -> 0, DC -> 1
    assert payload["chain_labels"] == [0, 0]
    assert payload["mask"] == [1, 1]
    assert payload["dna_mask"] == [1, 1]
    assert payload["predicted_ppm"] == [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    assert out_path.exists()


def test_fallback_canonical_handles_no_dna_residues(tmp_path):
    """Fallback degrades gracefully when no DNA residues are present."""
    from proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run import (
        _fallback_canonical_from_pdb,
    )

    pdb_path = tmp_path / "protein_only.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    out_path = tmp_path / "fallback.npz"

    payload = _fallback_canonical_from_pdb(str(out_path), str(pdb_path))

    assert payload["true_sequence"] == [0]
    assert payload["predicted_ppm"] == [[0.25, 0.25, 0.25, 0.25]]


# ── Runner: device threading & fail-fast (pure standalone logic) ──────────────


def _fake_deeppbs_repo(tmp_path) -> Path:
    """Create a minimal DeepPBS repo skeleton so the runner's existence checks pass."""
    repo = tmp_path / "DeepPBS"
    (repo / "run" / "process" / "pred_configs").mkdir(parents=True)
    (repo / "run" / "process_co_crystal.py").write_text("")
    (repo / "run" / "predict.py").write_text("")
    (repo / "run" / "process" / "process_config.json").write_text("{}")
    (repo / "run" / "process" / "pred_configs" / "pred_config_deeppbs.json").write_text("{}")
    return repo


def _dna_pdb(tmp_path) -> Path:
    pdb = tmp_path / "complex.pdb"
    pdb.write_text("ATOM      2  P    DA B   1       1.000   1.000   1.000  1.00  0.00           P\n")
    return pdb


def test_runner_threads_configured_device(tmp_path, monkeypatch):
    """config.device reaches get_subprocess_device_env (regression for the cpu device bug)."""
    repo = _fake_deeppbs_repo(tmp_path)
    pdb = _dna_pdb(tmp_path)
    monkeypatch.setenv("PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR", str(repo))

    seen = {}

    def fake_env(device):
        seen["device"] = device
        return {"PATH": ""}  # no x3dna binaries -> hits the missing-deps branch

    monkeypatch.setattr(_run, "get_subprocess_device_env", fake_env)

    result = _run.run_deeppbs_specificity(
        {
            "pdb_paths": [str(pdb)],
            "device": "cpu",
            "allow_fallback": True,
            "output_directory": str(tmp_path / "out"),
        }
    )

    assert seen["device"] == "cpu"
    assert result["results"][0]["used_fallback"] is True


def test_runner_raises_when_fallback_disabled(tmp_path, monkeypatch):
    """With allow_fallback False (default), a missing dependency raises instead of faking a PPM."""
    repo = _fake_deeppbs_repo(tmp_path)
    pdb = _dna_pdb(tmp_path)
    monkeypatch.setenv("PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR", str(repo))
    monkeypatch.setattr(_run, "get_subprocess_device_env", lambda device: {"PATH": ""})

    with pytest.raises(RuntimeError, match="DeepPBS dependency missing"):
        _run.run_deeppbs_specificity(
            {
                "pdb_paths": [str(pdb)],
                "output_directory": str(tmp_path / "out"),
            }
        )


def test_x3dna_env_root_binaries_are_found(tmp_path, monkeypatch):
    """Binaries under $X3DNA/bin satisfy the dependency check (no false 'missing' report)."""
    repo = _fake_deeppbs_repo(tmp_path)
    pdb = _dna_pdb(tmp_path)
    x3dna_root = tmp_path / "x3dna"
    bindir = x3dna_root / "bin"
    bindir.mkdir(parents=True)
    for name in ("x3dna-dssr", "analyze"):
        binary = bindir / name
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    monkeypatch.setenv("PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR", str(repo))
    monkeypatch.setenv("X3DNA", str(x3dna_root))
    monkeypatch.setattr(_run, "get_subprocess_device_env", lambda device: {"PATH": ""})

    result = _run.run_deeppbs_specificity(
        {
            "pdb_paths": [str(pdb)],
            "allow_fallback": True,
            "output_directory": str(tmp_path / "out"),
        }
    )

    # Got past the dependency check via $X3DNA/bin; any fallback is from the stub repo,
    # not a "dependency missing" report.
    assert "dependency missing" not in (result["results"][0]["fallback_reason"] or "")


# ── Integration (real DeepPBS env; skipped by default) ────────────────────────


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_deeppbs_specificity_runs_on_structure():
    """Run DeepPBS end-to-end on a real protein-DNA PDB (requires local env + X3DNA)."""
    import os

    pdb = os.environ.get("DEEPPBS_TEST_PDB")
    if not pdb or not os.path.exists(pdb):
        pytest.skip("Set DEEPPBS_TEST_PDB to a protein-DNA PDB to run this integration test")

    output = run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=[pdb]),
        DeepPBSSpecificityConfig(),
    )
    validate_output(output)
    assert len(output.results) == 1
    result = output.results[0]
    # PPM is L x 4 in ACGT order; rows normalized.
    assert all(len(row) == 4 for row in result.predicted_ppm)
    assert os.path.exists(result.output_npz_path)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("deeppbs-specificity")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_deeppbs_specificity_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark deeppbs-specificity on a real protein-DNA structure (cold + warm).

    Requires a local DeepPBS repository, an X3DNA install, and the standalone
    env. Skips cleanly when ``DEEPPBS_TEST_PDB`` is unset or the env is absent.
    """
    import os

    pdb = os.environ.get("DEEPPBS_TEST_PDB")
    if not pdb or not os.path.exists(pdb):
        pytest.skip("Set DEEPPBS_TEST_PDB to a protein-DNA PDB to run this benchmark")

    inputs = DeepPBSSpecificityInput(pdb_paths=[pdb])
    config = DeepPBSSpecificityConfig()

    # Cold + warm to exercise persistent-worker reuse.
    for _ in range(2):
        output = run_deeppbs_specificity(inputs, config)
        validate_output(output)
        assert len(output.results) == 1
        assert all(len(row) == 4 for row in output.results[0].predicted_ppm)
