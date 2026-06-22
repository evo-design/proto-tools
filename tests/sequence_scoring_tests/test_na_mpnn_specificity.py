"""Tests for the NA-MPNN specificity prediction tool."""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import proto_tools.utils.standalone_helpers_source as _shs
from proto_tools.tools.sequence_scoring.na_mpnn_specificity import (
    NAMPNNSpecificityConfig,
    NAMPNNSpecificityInput,
    NAMPNNSpecificityOutput,
    run_na_mpnn_specificity,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

# The standalone run.py imports the worker-injected ``standalone_helpers`` package,
# which only sits on sys.path inside a tool venv. Add its source dir so the pure
# canonicalization logic can be imported and unit-tested on the host.
sys.path.insert(0, str(Path(next(iter(_shs.__path__)))))
_standalone_run = importlib.import_module("proto_tools.tools.sequence_scoring.na_mpnn_specificity.standalone.run")


# -- Input validation ------------------------------------------------------------------


def test_input_normalizes_single_path_string():
    """A bare string path is normalized into a single-element list (custom validator)."""
    payload = NAMPNNSpecificityInput(pdb_paths="dummy/one.pdb")
    assert payload.pdb_paths == ["dummy/one.pdb"]
    assert len(payload) == 1


def test_input_rejects_empty_list():
    """An empty pdb_paths list is rejected (min_length=1)."""
    with pytest.raises(ValueError):
        NAMPNNSpecificityInput(pdb_paths=[])


# -- Cloud gating ----------------------------------------------------------------------


def test_cloud_unsupported_reason_is_always_set():
    """NA-MPNN needs local repo + checkpoint, so cloud is unconditionally rejected."""
    reason = NAMPNNSpecificityConfig().cloud_unsupported_reason()
    assert reason is not None
    assert "cloud" in reason


# -- Registration ----------------------------------------------------------------------


def test_tool_key_registered():
    """The tool is discoverable in the registry under its expected key/category."""
    spec = ToolRegistry.get("na-mpnn-specificity")
    assert spec.key == "na-mpnn-specificity"
    assert spec.category == "sequence_scoring"
    assert spec.uses_gpu is True


# -- Canonicalization (pure numpy, no model) -------------------------------------------


def test_canonicalize_npz_transforms_to_acgt(tmp_path):
    """Raw NA-MPNN arrays are converted to the canonical DNA-only A,C,G,T schema."""
    raw_path = tmp_path / "raw.npz"
    out_path = tmp_path / "canonical.npz"

    predicted = np.zeros((4, 30), dtype=np.float64)
    predicted[:, 21] = [0.8, 0.1, 0.1, 0.3]  # A
    predicted[:, 22] = [0.1, 0.7, 0.1, 0.2]  # C
    predicted[:, 23] = [0.05, 0.1, 0.7, 0.2]  # G
    predicted[:, 24] = [0.05, 0.1, 0.1, 0.3]  # T

    np.savez_compressed(
        raw_path,
        predicted_ppm=predicted,
        true_sequence=np.array([21, 22, 23, 24], dtype=np.int64),
        mask=np.array([1, 1, 0, 1], dtype=np.int64),
        dna_mask=np.array([1, 1, 1, 0], dtype=np.int64),
        chain_labels=np.array([0, 0, 1, 1], dtype=np.int64),
        restype_to_int=np.array({"DA": 21, "DC": 22, "DG": 23, "DT": 24}, dtype=object),
    )

    payload = _standalone_run._canonicalize_npz(str(raw_path), str(out_path))

    # Only rows where mask==1 and dna_mask==1 survive: indices 0 and 1.
    assert len(payload["predicted_ppm"]) == 2
    assert len(payload["predicted_ppm"][0]) == 4
    # Each canonical row is renormalized over A,C,G,T.
    assert pytest.approx(sum(payload["predicted_ppm"][0]), abs=1e-9) == 1.0
    assert payload["true_sequence"] == [0, 1]
    assert payload["mask"] == [1, 1]
    assert payload["dna_mask"] == [1, 1]
    assert out_path.exists()


def test_canonicalize_npz_masks_unknown_true_base(tmp_path):
    """Unknown truth tokens are masked out (mask=0), not backfilled from the prediction."""
    raw_path = tmp_path / "raw.npz"
    out_path = tmp_path / "canonical.npz"

    predicted = np.zeros((1, 30), dtype=np.float64)
    predicted[0, 21] = 0.1  # A
    predicted[0, 22] = 0.1  # C
    predicted[0, 23] = 0.7  # G (would have been the argmax backfill)
    predicted[0, 24] = 0.1  # T

    np.savez_compressed(
        raw_path,
        predicted_ppm=predicted,
        true_sequence=np.array([99], dtype=np.int64),  # unknown token
        mask=np.array([1], dtype=np.int64),
        dna_mask=np.array([1], dtype=np.int64),
        chain_labels=np.array([0], dtype=np.int64),
    )

    payload = _standalone_run._canonicalize_npz(str(raw_path), str(out_path))
    assert payload["mask"] == [0]
    assert payload["true_sequence"] == [0]  # placeholder; ignored because mask=0


# -- Path resolution -------------------------------------------------------------------


def test_resolve_repo_from_weights_cache(tmp_path, monkeypatch):
    """The NA-MPNN repo resolves from the managed weights cache (PROTO_*_WEIGHTS_DIR)."""
    repo = tmp_path / "cached_na_mpnn"
    (repo / "inference").mkdir(parents=True)
    (repo / "inference" / "run.py").write_text("")
    monkeypatch.setenv("PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR", str(repo))

    resolved = _standalone_run._resolve_na_mpnn_repo()
    assert Path(resolved).samefile(repo)


def test_resolve_repo_missing_raises(tmp_path, monkeypatch):
    """With nothing provisioned, repo resolution raises an actionable error."""
    monkeypatch.delenv("PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path / "empty_cache"))
    with pytest.raises(FileNotFoundError, match="could not locate an NA-MPNN checkout"):
        _standalone_run._resolve_na_mpnn_repo()


def test_resolve_checkpoint_prefers_nested_specificity_model(tmp_path, monkeypatch):
    """The checkpoint resolves from the in-repo models/specificity_model/ dir, not the design one."""
    repo = tmp_path / "cached_na_mpnn"
    (repo / "models" / "specificity_model").mkdir(parents=True)
    (repo / "models" / "design_model").mkdir(parents=True)
    spec_ckpt = repo / "models" / "specificity_model" / "s_70114.pt"
    spec_ckpt.write_text("")
    (repo / "models" / "design_model" / "s_19137.pt").write_text("")
    monkeypatch.setenv("PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR", str(repo))

    resolved = _standalone_run._resolve_checkpoint()
    assert Path(resolved).samefile(spec_ckpt)


# -- Dispatch (mocked) -----------------------------------------------------------------


@patch("proto_tools.tools.sequence_scoring.na_mpnn_specificity.na_mpnn_specificity.ToolInstance.dispatch")
def test_run_returns_canonicalized_output(mock_dispatch):
    """run_na_mpnn_specificity wraps dispatch results into a typed, exportable output."""
    mock_dispatch.return_value = {
        "results": [
            {
                "input_name": "candidate_0",
                "source_method": "na_mpnn",
                "output_npz_path": "dummy/candidate_0.npz",
                "predicted_ppm": [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]],
                "true_sequence": [0, 1],
                "mask": [1, 1],
                "dna_mask": [1, 1],
                "chain_labels": [0, 1],
            }
        ]
    }

    output = run_na_mpnn_specificity(
        NAMPNNSpecificityInput(pdb_paths=["dummy/candidate_0.pdb"]),
        NAMPNNSpecificityConfig(),
    )
    validate_output(output)

    assert isinstance(output, NAMPNNSpecificityOutput)
    assert len(output.results) == 1
    assert output.results[0].source_method == "na_mpnn"
    assert output.results[0].predicted_ppm[0] == [0.7, 0.1, 0.1, 0.1]
    assert output.results[0].chain_labels == [0, 1]
    assert output.results[0].output_npz_path == "dummy/candidate_0.npz"
    mock_dispatch.assert_called_once()


@patch("proto_tools.tools.sequence_scoring.na_mpnn_specificity.na_mpnn_specificity.ToolInstance.dispatch")
def test_run_validates_result_cardinality(mock_dispatch):
    """A result count that doesn't match the inputs is a hard error."""
    mock_dispatch.return_value = {"results": []}
    with pytest.raises(ValueError, match="Expected 1 NA-MPNN results"):
        run_na_mpnn_specificity(
            NAMPNNSpecificityInput(pdb_paths=["dummy/candidate_0.pdb"]),
            NAMPNNSpecificityConfig(),
        )


# -- Integration (requires real NA-MPNN env + checkpoint) ------------------------------


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_na_mpnn_specificity_basic_execution():
    """End-to-end NA-MPNN specificity on a real protein-DNA complex (needs weights)."""
    pdb_file = Path(__file__).parent.parent / "dummy_data" / "protein_dna_complex.pdb"
    if not pdb_file.exists():
        pytest.skip("No protein-DNA dummy PDB available for NA-MPNN integration test")

    output = run_na_mpnn_specificity(
        NAMPNNSpecificityInput(pdb_paths=[str(pdb_file)]),
        NAMPNNSpecificityConfig(),
    )
    assert output.success
    assert output.tool_id == "na-mpnn-specificity"
    assert len(output.results) == 1
    ppm = output.results[0].predicted_ppm
    assert all(len(row) == 4 for row in ppm)


@pytest.mark.benchmark("na-mpnn-specificity")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_na_mpnn_specificity_benchmark(request):
    """Benchmark na-mpnn-specificity on a real protein-DNA complex (cold + warm)."""
    pdb_file = Path(__file__).parent.parent / "dummy_data" / "protein_dna_complex.pdb"
    if not pdb_file.exists():
        pytest.skip("No protein-DNA dummy PDB available for NA-MPNN benchmark")

    inputs = NAMPNNSpecificityInput(pdb_paths=[str(pdb_file)])
    config = NAMPNNSpecificityConfig()

    result = benchmark_twice(request, "na_mpnn_specificity", lambda: run_na_mpnn_specificity(inputs, config))
    validate_output(result)

    assert result.tool_id == "na-mpnn-specificity"
    assert len(result.results) == 1
    assert all(len(row) == 4 for row in result.results[0].predicted_ppm)
