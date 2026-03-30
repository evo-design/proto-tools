"""tests/gene_annotation_tests/test_crispr_tracr.py

Tests for the CRISPRtracrRNA prediction tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.gene_annotation.crispr_tracr import (
    CrisprTracrConfig,
    CrisprTracrInput,
    CrisprTracrOutput,
    TracrPrediction,
    run_crispr_tracr,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
)

# Checked-in positive-control FASTA files.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "crispr_tracr"
_SETUP_SH = (
    Path(__file__).resolve().parents[2]
    / "bio_programming_tools/tools/gene_annotation/crispr_tracr/standalone/setup.sh"
)

_persistent_tool = make_persistent_fixture("crispr_tracr", gpu=False)


def _read_fasta_sequence(fasta_path: Path) -> str:
    """Read a single-record FASTA file and return the sequence string."""
    lines = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                lines.append(line.strip())
    return "".join(lines)


# ── Input validation ──────────────────────────────────────────────────────


def test_input_single_sequence_normalization():
    """Single string should be normalized to list."""
    inp = CrisprTracrInput(sequences="ATCGATCG")
    assert isinstance(inp.sequences, list)
    assert len(inp.sequences) == 1
    assert inp.sequences[0] == "ATCGATCG"


def test_input_list_of_sequences():
    inp = CrisprTracrInput(sequences=["ATCG", "GCTA"])
    assert len(inp.sequences) == 2


def test_input_custom_sequence_ids():
    inp = CrisprTracrInput(sequences=["ATCG"], sequence_ids=["my_seq"])
    assert inp.sequence_ids == ["my_seq"]


def test_config_invalid_model_type():
    with pytest.raises(ValidationError, match="Input should be"):
        CrisprTracrConfig(model_type="invalid")


# ── TracrPrediction ───────────────────────────────────────────────────────


def test_has_tracr_with_prediction():
    pred = TracrPrediction(
        sequence_id="test",
        tracr_start=100,
        tracr_end=200,
        tracr_hit="hit_description",
        interaction_energy=-6.49,
    )
    assert pred.has_tracr is True


def test_has_tracr_without_prediction():
    pred = TracrPrediction(sequence_id="test")
    assert pred.has_tracr is False


# ── Output ────────────────────────────────────────────────────────────────


def test_num_with_tracr():
    p1 = TracrPrediction(sequence_id="seq1", tracr_start=100, tracr_end=200)
    p2 = TracrPrediction(sequence_id="seq2")  # No tracr
    p3 = TracrPrediction(sequence_id="seq3", tracr_start=50, tracr_end=150)

    output = CrisprTracrOutput(predictions=[p1, p2, p3])
    assert output.num_with_tracr == 2


def test_num_with_tracr_empty():
    output = CrisprTracrOutput(predictions=[])
    assert output.num_with_tracr == 0


def test_num_with_tracr_all_negative():
    p1 = TracrPrediction(sequence_id="seq1")
    p2 = TracrPrediction(sequence_id="seq2")
    output = CrisprTracrOutput(predictions=[p1, p2])
    assert output.num_with_tracr == 0


# ── Export ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_output():
    p1 = TracrPrediction(
        sequence_id="seq1",
        tracr_start=100,
        tracr_end=200,
        interaction_energy=-8.5,
    )
    p2 = TracrPrediction(sequence_id="seq2")
    return CrisprTracrOutput(predictions=[p1, p2])


def test_export_csv(sample_output, tmp_path):
    sample_output.export(
        name="tracr", export_path=str(tmp_path), file_format="csv"
    )
    csv_path = tmp_path / "tracr.csv"
    assert validate_export_output(csv_path)


def test_export_json(sample_output, tmp_path):
    sample_output.export(
        name="tracr", export_path=str(tmp_path), file_format="json"
    )
    json_path = tmp_path / "tracr.json"
    assert validate_export_output(json_path)

    data = json.loads(json_path.read_text())
    assert len(data) == 2


# ── Standalone runner config ──────────────────────────────────────────────


def test_subprocess_uses_isolated_cwd():
    """subprocess.run must receive cwd= pointing to a worker-specific directory."""
    import importlib
    run_module = importlib.import_module(
        "bio_programming_tools.tools.gene_annotation.crispr_tracr.standalone.run"
    )

    fake_script = "/fake/install/dir/CRISPRtracrRNA.py"
    fake_worker_cwd = Path("/fake/worker/cwd")
    with patch.object(run_module, "_find_crispr_tracr_script", return_value=fake_script), \
         patch.object(run_module, "_create_worker_cwd", return_value=fake_worker_cwd), \
         patch.object(run_module.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        input_data = {
            "sequences": ["ATCG" * 100],
            "sequence_ids": ["test_seq"],
            "config": {"model_type": "II"},
        }
        run_module.run_crispr_tracr(input_data)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("cwd") == str(fake_worker_cwd), (
            "subprocess.run must set cwd to an isolated worker directory "
            "so relative paths resolve correctly and parallel workers don't conflict"
        )


def test_setup_installs_crispridentify():
    content = _SETUP_SH.read_text()
    assert "CRISPRidentify" in content, "setup.sh must install CRISPRidentify"
    assert "tools/CRISPRidentify/CRISPRidentify" in content, (
        "CRISPRidentify must be cloned into tools/CRISPRidentify/CRISPRidentify/ "
        "within the CRISPRtracrRNA installation"
    )


def test_setup_installs_crisprcastidentifier():
    content = _SETUP_SH.read_text()
    assert "CRISPRcasIdentifier" in content, (
        "setup.sh must install CRISPRcasIdentifier"
    )


def test_setup_uses_python38_sklearn022():
    """setup.sh must create conda_deps with Python 3.8 + sklearn 0.22."""
    content = _SETUP_SH.read_text()
    assert "python=3.8" in content, (
        "setup.sh must install Python 3.8 in conda_deps for CRISPRidentify"
    )
    assert "scikit-learn=0.22" in content, (
        "setup.sh must install scikit-learn 0.22 in conda_deps for CRISPRidentify"
    )


def test_setup_installs_conda_tools():
    content = _SETUP_SH.read_text()
    for tool in ["intarna", "infernal", "prodigal", "hmmer", "viennarna", "vmatch", "clustalo", "blast", "fasta3"]:
        assert tool in content, f"setup.sh must install {tool} via conda"
    for dep in ["h5py", "dill", "networkx", "pyyaml", "regex", "requests"]:
        assert dep in content, (
            f"setup.sh must install {dep} in conda_deps for CRISPRidentify"
        )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="gene_annotation")
def test_run_crispr_tracr():
    """Run CRISPRtracrRNA on a synthetic sequence."""
    test_seq = "ATCGATCG" * 500
    inputs = CrisprTracrInput(
        sequences=[test_seq],
        sequence_ids=["test_seq"],
    )
    config = CrisprTracrConfig(model_type="II")
    result = run_crispr_tracr(inputs, config)

    assert isinstance(result, CrisprTracrOutput)
    assert len(result.predictions) == 1


@pytest.mark.integration
def test_run_crispr_tracr_model_type_all():
    """Run CRISPRtracrRNA with model_type='all'."""
    test_seq = "ATCGATCG" * 500
    inputs = CrisprTracrInput(sequences=[test_seq])
    config = CrisprTracrConfig(model_type="all")
    result = run_crispr_tracr(inputs, config)

    assert isinstance(result, CrisprTracrOutput)
    assert len(result.predictions) == 1


@pytest.mark.integration
def test_run_crispr_tracr_real_sequence():
    """Positive control: Listeria monocytogenes CRISPR locus."""
    test_fasta = DATA_DIR / "AAAABU010000051.fasta"
    seq = _read_fasta_sequence(test_fasta)

    inputs = CrisprTracrInput(
        sequences=[seq],
        sequence_ids=["AAAABU010000051"],
    )
    config = CrisprTracrConfig(model_type="II")
    result = run_crispr_tracr(inputs, config)

    assert isinstance(result, CrisprTracrOutput)
    assert result.success, "CRISPRtracrRNA failed on real CRISPR sequence"
    assert len(result.predictions) == 1
    pred = result.predictions[0]
    assert pred.sequence_id == "AAAABU010000051"
    assert pred.has_tracr, (
        "Expected tracrRNA detection on Listeria monocytogenes CRISPR locus"
    )
    assert pred.tracr_start is not None
    assert pred.tracr_end is not None
    assert pred.tracr_hit is not None
    assert pred.anti_repeat_similarity_coverage_multiplication is not None
    assert pred.anti_repeat_similarity_coverage_multiplication > 0.5


@pytest.mark.integration
def test_positive_control_spyogenes_sf370():
    """Positive control: S. pyogenes SF370 canonical CRISPR-Cas9 locus."""
    test_fasta = DATA_DIR / "NC_002737_849000_875000.fasta"
    seq = _read_fasta_sequence(test_fasta)

    inputs = CrisprTracrInput(
        sequences=[seq], sequence_ids=["SpCas9_SF370"]
    )
    config = CrisprTracrConfig(model_type="II")
    result = run_crispr_tracr(inputs, config)

    assert isinstance(result, CrisprTracrOutput)
    assert result.success
    assert result.num_with_tracr == 1

    pred = result.predictions[0]
    assert pred.has_tracr, (
        "Expected tracrRNA detection on canonical SpCas9 locus"
    )
    assert pred.tracr_hit is not None
    # The canonical sgRNA scaffold contains GTGGCACCGAGTCGGTGC
    assert "GTGGCACCGAGTCGGTGC" in pred.tracr_hit, (
        f"Expected canonical sgRNA scaffold in tracrRNA hit, "
        f"got: {pred.tracr_hit[:80]}..."
    )
    assert pred.anti_repeat_similarity_coverage_multiplication is not None
    assert pred.anti_repeat_similarity_coverage_multiplication > 0.5
    assert pred.intarna_anti_repeat_interaction is not None
