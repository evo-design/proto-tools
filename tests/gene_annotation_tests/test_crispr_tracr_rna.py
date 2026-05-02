"""Tests for the CRISPRtracrRNA prediction tool."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_tools.tools.gene_annotation.crispr_tracr_rna import (
    CrisprTracrRNAConfig,
    CrisprTracrRNAInput,
    CrisprTracrRNAOutput,
    CrisprTracrRNAPrediction,
    run_crispr_tracr_rna,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
    validate_output,
)

# Checked-in positive-control FASTA files.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "crispr_tracr_rna"
_SETUP_SH = (
    Path(__file__).resolve().parents[2] / "proto_tools/tools/gene_annotation/crispr_tracr_rna/standalone/setup.sh"
)

_persistent_tool = make_persistent_fixture("crispr_tracr_rna", gpu=False)


def _read_fasta_sequence(fasta_path: Path) -> str:
    """Read a single-record FASTA file and return the sequence string."""
    with open(fasta_path) as f:
        lines = [line.strip() for line in f if not line.startswith(">")]
    return "".join(lines)


# ── Input validation ──────────────────────────────────────────────────────


def test_input_single_sequence_normalization():
    """The custom validator turns a bare string into a list of one."""
    inp = CrisprTracrRNAInput(sequences="ATCGATCG")
    assert inp.sequences == ["ATCGATCG"]


# ── CrisprTracrRNAPrediction ───────────────────────────────────────────────────────


def test_has_tracr_truth_table():
    """has_tracr fires on either upstream candidate-detection field, else False."""
    populated = CrisprTracrRNAPrediction(sequence_id="test", tracr_rna_sequence="GCAUGCAUGC", anti_repeat_start=100)
    anti_repeat_only = CrisprTracrRNAPrediction(sequence_id="test", anti_repeat_start=42)
    empty = CrisprTracrRNAPrediction(sequence_id="test")
    assert populated.has_tracr is True
    assert anti_repeat_only.has_tracr is True  # model_run mode
    assert empty.has_tracr is False


# ── Output ────────────────────────────────────────────────────────────────


def test_num_with_tracr():
    """Counts only predictions where has_tracr is True."""
    output = CrisprTracrRNAOutput(
        predictions=[
            CrisprTracrRNAPrediction(sequence_id="seq1", anti_repeat_start=100),
            CrisprTracrRNAPrediction(sequence_id="seq2"),  # no tracr
            CrisprTracrRNAPrediction(sequence_id="seq3", tracr_rna_sequence="ACGUACGU"),
        ]
    )
    assert output.num_with_tracr == 2


# ── Export ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_output():
    p1 = CrisprTracrRNAPrediction(
        sequence_id="seq1",
        anti_repeat_start=100,
        anti_repeat_end=200,
        tracr_rna_sequence="ACGUACGUACGU",
        interaction_energy=-8.5,
        score=0.84,
    )
    p2 = CrisprTracrRNAPrediction(sequence_id="seq2")
    return CrisprTracrRNAOutput(predictions=[p1, p2])


def test_export_csv(sample_output, tmp_path):
    sample_output.export(name="tracr", export_path=str(tmp_path), file_format="csv")
    csv_path = tmp_path / "tracr.csv"
    assert validate_export_output(csv_path)


def test_export_json(sample_output, tmp_path):
    sample_output.export(name="tracr", export_path=str(tmp_path), file_format="json")
    json_path = tmp_path / "tracr.json"
    assert validate_export_output(json_path)

    data = json.loads(json_path.read_text())
    assert len(data) == 2


# ── Standalone runner config ──────────────────────────────────────────────


def test_subprocess_uses_isolated_cwd():
    """subprocess.run must receive cwd= pointing to a worker-specific directory."""
    import importlib

    run_module = importlib.import_module("proto_tools.tools.gene_annotation.crispr_tracr_rna.standalone.run")

    fake_script = "/fake/install/dir/CRISPRtracrRNA.py"
    fake_worker_cwd = Path("/fake/worker/cwd")
    with (
        patch.object(run_module, "_find_crispr_tracr_rna_script", return_value=fake_script),
        patch.object(run_module, "_create_worker_cwd", return_value=fake_worker_cwd),
        patch.object(run_module.subprocess, "run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # The wrapper always resolves num_workers before dispatch.
        input_data = {
            "sequences": ["ATCG" * 100],
            "sequence_ids": ["test_seq"],
            "config": {"model_type": "II", "run_type": "complete_run", "num_workers": 1},
        }
        run_module.run_crispr_tracr_rna(input_data)

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("cwd") == str(fake_worker_cwd), (
            "subprocess.run must set cwd to an isolated worker directory "
            "so relative paths resolve correctly and parallel workers don't conflict"
        )


def test_parse_tracr_results_passes_columns_through(tmp_path: Path):
    """The parser must hand every CSV column to the wrapper, normalizing only NA-style cells."""
    import importlib

    run_module = importlib.import_module("proto_tools.tools.gene_annotation.crispr_tracr_rna.standalone.run")

    # Mimic upstream's complete_run CSV with a representative slice of columns.
    csv_text = (
        "accession_number,anti_repeat_start,anti_repeat_end,interaction_energy,terminator_presence_flag,score\n"
        "seq1,42,168,-7.5,True,0.84\n"
        "seq2,,,,NA,\n"
    )
    (tmp_path / "results.csv").write_text(csv_text)

    parsed = run_module._parse_tracr_results(tmp_path, ["seq1", "seq2", "seq3"])
    assert len(parsed) == 3
    by_id = {p["sequence_id"]: p for p in parsed}

    # seq1: populated row, all columns forwarded as strings (Pydantic coerces on the wrapper side)
    assert by_id["seq1"]["anti_repeat_start"] == "42"
    assert by_id["seq1"]["interaction_energy"] == "-7.5"
    assert by_id["seq1"]["terminator_presence_flag"] == "True"
    assert by_id["seq1"]["score"] == "0.84"

    # seq2: empty / NA cells normalized to None
    assert by_id["seq2"]["anti_repeat_start"] is None
    assert by_id["seq2"]["terminator_presence_flag"] is None
    assert by_id["seq2"]["score"] is None

    # seq3: missing from CSV — placeholder row with only sequence_id
    assert by_id["seq3"] == {"sequence_id": "seq3"}


def test_build_upstream_flags_omits_defaults():
    """Flags whose value matches upstream's default must not appear on the CLI."""
    import importlib

    run_module = importlib.import_module("proto_tools.tools.gene_annotation.crispr_tracr_rna.standalone.run")

    config = {
        "anti_repeat_similarity_threshold": 0.7,  # upstream default
        "weight_interaction_score": 0.6,  # upstream default
        "perform_type_v_anti_repeat_analysis": False,
    }
    flags = run_module._build_upstream_flags(config)
    assert flags == [], f"defaults must be omitted, got {flags}"


def test_build_upstream_flags_emits_overrides():
    """Non-default values must appear with their --flag <value> form."""
    import importlib

    run_module = importlib.import_module("proto_tools.tools.gene_annotation.crispr_tracr_rna.standalone.run")

    config = {
        "anti_repeat_similarity_threshold": 0.85,  # override
        "weight_interaction_score": 1.0,  # override
        "perform_type_v_anti_repeat_analysis": True,  # bool override
    }
    flags = run_module._build_upstream_flags(config)
    assert "--anti_repeat_similarity_threshold" in flags
    assert "0.85" in flags
    assert "--weight_interaction_score" in flags
    assert "1.0" in flags
    # Bool flag: only appears when True, with the literal "True" value
    assert "--perform_type_v_anti_repeat_analysis" in flags
    assert "True" in flags


def test_setup_installs_crispridentify():
    content = _SETUP_SH.read_text()
    assert "CRISPRidentify" in content, "setup.sh must install CRISPRidentify"
    assert "tools/CRISPRidentify/CRISPRidentify" in content, (
        "CRISPRidentify must be cloned into tools/CRISPRidentify/CRISPRidentify/ within the CRISPRtracrRNA installation"
    )


def test_setup_installs_crisprcastidentifier():
    content = _SETUP_SH.read_text()
    assert "CRISPRcasIdentifier" in content, "setup.sh must install CRISPRcasIdentifier"


def test_setup_downloads_crisprcastidentifier_models():
    """setup.sh must download the CRISPRcasIdentifier HMM/ML models from Google Drive.

    Without these, the cas-effector-detection leg of complete_run is a silent no-op.
    File IDs are from the upstream README.
    """
    content = _SETUP_SH.read_text()
    assert "proto_download_gdrive" in content, "setup.sh must use the proto_download_gdrive helper"
    assert "1YbTxkn9KuJP2D7U1-6kL1Yimu_4RqSl1" in content, "setup.sh must download HMM_sets archive"
    assert "1Nc5o6QVB6QxMxpQjmLQcbwQwkRLk-thM" in content, "setup.sh must download trained_models archive"
    # Filenames must match upstream's canonical names; CRISPRcasIdentifier.py
    # looks them up as BASE_DIR/{HMM_sets,trained_models}.tar.gz at runtime.
    assert "HMM_sets.tar.gz" in content, "setup.sh must save first archive as canonical HMM_sets.tar.gz"
    assert "trained_models.tar.gz" in content, "setup.sh must save second archive as canonical trained_models.tar.gz"
    assert "tar -xzf HMM_sets.tar.gz" in content, "setup.sh must pre-extract HMM_sets.tar.gz"
    assert "tar -xzf trained_models.tar.gz" in content, "setup.sh must pre-extract trained_models.tar.gz"


def test_setup_uses_python38_sklearn022():
    """setup.sh must create conda_deps with Python 3.8 + sklearn 0.22."""
    content = _SETUP_SH.read_text()
    assert "python=3.8" in content, "setup.sh must install Python 3.8 in conda_deps for CRISPRidentify"
    assert "scikit-learn=0.22" in content, "setup.sh must install scikit-learn 0.22 in conda_deps for CRISPRidentify"


def test_setup_installs_conda_tools():
    content = _SETUP_SH.read_text()
    for tool in ["intarna", "infernal", "prodigal", "hmmer", "viennarna", "vmatch", "clustalo", "blast", "fasta3"]:
        assert tool in content, f"setup.sh must install {tool} via conda"
    for dep in ["h5py", "dill", "networkx", "pyyaml", "regex", "requests"]:
        assert dep in content, f"setup.sh must install {dep} in conda_deps for CRISPRidentify"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_crispr_tracr_rna():
    """Run CRISPRtracrRNA on a synthetic sequence."""
    test_seq = "ATCGATCG" * 500
    inputs = CrisprTracrRNAInput(
        sequences=[test_seq],
        sequence_ids=["test_seq"],
    )
    config = CrisprTracrRNAConfig(model_type="II")
    result = run_crispr_tracr_rna(inputs, config)

    assert isinstance(result, CrisprTracrRNAOutput)
    assert len(result.predictions) == 1


@pytest.mark.integration
def test_run_crispr_tracr_rna_model_type_all():
    """Run CRISPRtracrRNA with model_type='all'."""
    test_seq = "ATCGATCG" * 500
    inputs = CrisprTracrRNAInput(sequences=[test_seq])
    config = CrisprTracrRNAConfig(model_type="all")
    result = run_crispr_tracr_rna(inputs, config)

    assert isinstance(result, CrisprTracrRNAOutput)
    assert len(result.predictions) == 1


@pytest.mark.integration
def test_run_crispr_tracr_rna_real_sequence():
    """Positive control: Listeria monocytogenes CRISPR locus."""
    test_fasta = DATA_DIR / "AAAABU010000051.fasta"
    seq = _read_fasta_sequence(test_fasta)

    inputs = CrisprTracrRNAInput(
        sequences=[seq],
        sequence_ids=["AAAABU010000051"],
    )
    config = CrisprTracrRNAConfig(model_type="II")
    result = run_crispr_tracr_rna(inputs, config)

    assert isinstance(result, CrisprTracrRNAOutput)
    assert result.success, "CRISPRtracrRNA failed on real CRISPR sequence"
    assert len(result.predictions) == 1
    pred = result.predictions[0]
    assert pred.sequence_id == "AAAABU010000051"
    assert pred.has_tracr, "Expected tracrRNA detection on Listeria monocytogenes CRISPR locus"
    assert pred.anti_repeat_start is not None
    assert pred.anti_repeat_end is not None
    assert pred.tracr_rna_sequence is not None
    assert pred.anti_repeat_similarity_coverage_multiplication is not None
    assert pred.anti_repeat_similarity_coverage_multiplication > 0.5


@pytest.mark.integration
def test_positive_control_spyogenes_sf370():
    """Positive control: S. pyogenes SF370 canonical CRISPR-Cas9 locus."""
    test_fasta = DATA_DIR / "NC_002737_849000_875000.fasta"
    seq = _read_fasta_sequence(test_fasta)

    inputs = CrisprTracrRNAInput(sequences=[seq], sequence_ids=["SpCas9_SF370"])
    config = CrisprTracrRNAConfig(model_type="II")
    result = run_crispr_tracr_rna(inputs, config)

    assert isinstance(result, CrisprTracrRNAOutput)
    assert result.success
    assert result.num_with_tracr == 1

    pred = result.predictions[0]
    assert pred.has_tracr, "Expected tracrRNA detection on canonical SpCas9 locus"
    assert pred.tracr_rna_sequence is not None
    # The canonical sgRNA scaffold contains GTGGCACCGAGTCGGTGC
    assert "GTGGCACCGAGTCGGTGC" in pred.tracr_rna_sequence, (
        f"Expected canonical sgRNA scaffold in tracrRNA sequence, got: {pred.tracr_rna_sequence[:80]}..."
    )
    assert pred.anti_repeat_similarity_coverage_multiplication is not None
    assert pred.anti_repeat_similarity_coverage_multiplication > 0.5
    assert pred.intarna_anti_repeat_interaction is not None


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("crispr-tracr-rna")
@pytest.mark.slow
def test_crispr_tracr_rna_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark crispr-tracr-rna: full S. pyogenes SF370 CRISPR-Cas9 locus (~26 kbp), model_type='II' (cold + warm)."""
    test_fasta = DATA_DIR / "NC_002737_849000_875000.fasta"
    seq = _read_fasta_sequence(test_fasta)
    inputs = CrisprTracrRNAInput(sequences=[seq], sequence_ids=["SpCas9_SF370"])
    config = CrisprTracrRNAConfig(model_type="II")

    result = benchmark_twice(request, "crispr_tracr_rna", lambda: run_crispr_tracr_rna(inputs, config))
    validate_output(result)

    assert result.tool_id == "crispr-tracr-rna"
    assert len(result.predictions) == 1
