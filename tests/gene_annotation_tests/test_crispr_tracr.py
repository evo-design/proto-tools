"""
Tests for the CRISPRtracrRNA prediction tool.

Unit tests for data models and input normalization, plus integration
tests (skip_ci) for actual CRISPRtracrRNA execution.

Setup/dependency tests verify that the standalone environment is correctly
configured (CWD, CRISPRidentify, sklearn compat, conda tools).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bio_programming_tools.tools.gene_annotation.crispr_tracr import (
    CrisprTracrConfig,
    CrisprTracrInput,
    CrisprTracrOutput,
    TracrPrediction,
    run_crispr_tracr,
)
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
)

# Checked-in positive-control FASTA files.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "crispr_tracr"


# ============================================================================
# Data Model Tests
# ============================================================================
class TestCrisprTracrInput:
    """Tests for CrisprTracrInput validation and normalization."""

    def test_single_sequence_normalization(self):
        """Single string should be normalized to list."""
        inp = CrisprTracrInput(sequences="ATCGATCG")
        assert isinstance(inp.sequences, list)
        assert len(inp.sequences) == 1
        assert inp.sequences[0] == "ATCGATCG"

    def test_list_of_sequences(self):
        """List of sequences should be preserved."""
        inp = CrisprTracrInput(sequences=["ATCG", "GCTA"])
        assert len(inp.sequences) == 2

    def test_optional_sequence_ids(self):
        """Sequence IDs should default to None."""
        inp = CrisprTracrInput(sequences=["ATCG"])
        assert inp.sequence_ids is None

    def test_custom_sequence_ids(self):
        """Custom sequence IDs should be preserved."""
        inp = CrisprTracrInput(sequences=["ATCG"], sequence_ids=["my_seq"])
        assert inp.sequence_ids == ["my_seq"]


class TestCrisprTracrConfig:
    """Tests for CrisprTracrConfig defaults and validation."""

    def test_defaults(self):
        """Default config should use model_type 'II'."""
        config = CrisprTracrConfig()
        assert config.model_type == "II"

    def test_model_type_all(self):
        """model_type 'all' should be accepted."""
        config = CrisprTracrConfig(model_type="all")
        assert config.model_type == "all"

    def test_invalid_model_type(self):
        """Invalid model_type should raise an error."""
        with pytest.raises(Exception):
            CrisprTracrConfig(model_type="invalid")

    def test_extra_ignored(self):
        """Extra fields should be ignored."""
        config = CrisprTracrConfig(extra_field="ignored")
        assert config.model_type == "II"


# ============================================================================
# TracrPrediction Tests
# ============================================================================
class TestTracrPrediction:
    """Tests for TracrPrediction model."""

    def test_has_tracr_with_prediction(self):
        """has_tracr should be True when tracr_start is set."""
        pred = TracrPrediction(
            sequence_id="test",
            tracr_start=100,
            tracr_end=200,
            tracr_hit="hit_description",
            interaction_energy=-6.49,
        )
        assert pred.has_tracr is True

    def test_has_tracr_without_prediction(self):
        """has_tracr should be False when tracr_start is None."""
        pred = TracrPrediction(sequence_id="test")
        assert pred.has_tracr is False

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        pred = TracrPrediction(sequence_id="test")
        assert pred.tracr_start is None
        assert pred.tracr_end is None
        assert pred.tracr_hit is None
        assert pred.interaction_energy is None
        assert pred.anti_repeat_similarity_coverage_multiplication is None
        assert pred.intarna_anti_repeat_interaction is None


# ============================================================================
# CrisprTracrOutput Tests
# ============================================================================
class TestCrisprTracrOutput:
    """Tests for CrisprTracrOutput properties."""

    def test_num_with_tracr(self):
        """num_with_tracr should count predictions with tracr_start set."""
        p1 = TracrPrediction(sequence_id="seq1", tracr_start=100, tracr_end=200)
        p2 = TracrPrediction(sequence_id="seq2")  # No tracr
        p3 = TracrPrediction(sequence_id="seq3", tracr_start=50, tracr_end=150)

        output = CrisprTracrOutput(predictions=[p1, p2, p3])
        assert output.num_with_tracr == 2

    def test_empty_output(self):
        """Empty output should have 0 with tracr."""
        output = CrisprTracrOutput(predictions=[])
        assert output.num_with_tracr == 0

    def test_all_without_tracr(self):
        """All predictions without tracr should give 0."""
        p1 = TracrPrediction(sequence_id="seq1")
        p2 = TracrPrediction(sequence_id="seq2")
        output = CrisprTracrOutput(predictions=[p1, p2])
        assert output.num_with_tracr == 0


# ============================================================================
# Export Tests
# ============================================================================
class TestCrisprTracrExport:
    """Tests for CrisprTracrOutput export functionality."""

    @pytest.fixture
    def sample_output(self):
        p1 = TracrPrediction(
            sequence_id="seq1",
            tracr_start=100,
            tracr_end=200,
            interaction_energy=-8.5,
        )
        p2 = TracrPrediction(sequence_id="seq2")
        return CrisprTracrOutput(predictions=[p1, p2])

    def test_export_csv(self, sample_output, tmp_path):
        """Export to CSV format."""
        sample_output.export(
            name="tracr", export_path=str(tmp_path), file_format="csv"
        )
        csv_path = tmp_path / "tracr.csv"
        assert validate_export_output(csv_path)

    def test_export_json(self, sample_output, tmp_path):
        """Export to JSON format."""
        sample_output.export(
            name="tracr", export_path=str(tmp_path), file_format="json"
        )
        json_path = tmp_path / "tracr.json"
        assert validate_export_output(json_path)

        data = json.loads(json_path.read_text())
        assert len(data) == 2

    def test_output_format_options(self, sample_output):
        """Check supported output formats."""
        assert "csv" in sample_output.output_format_options
        assert "json" in sample_output.output_format_options
        assert sample_output.output_format_default == "csv"


# ============================================================================
# Standalone Runner Tests (would have caught CWD + dependency bugs)
# ============================================================================
class TestStandaloneRunnerConfig:
    """Tests for standalone run.py configuration.

    These tests verify that run.py sets subprocess CWD correctly and that
    setup.sh installs all required dependencies.  The CWD bug (GitHub issue)
    caused CRISPRtracrRNA to look for tools/ relative to the repo root
    instead of its own installation directory.
    """

    def test_subprocess_uses_isolated_cwd(self):
        """subprocess.run must receive cwd= pointing to a worker-specific directory.

        Each worker gets its own CWD with symlinks to the CRISPRtracrRNA install
        dir to avoid file contention from intermediate files. The CWD must NOT
        be the repo root (which caused the original CRISPRidentify FileNotFoundError).
        """
        import importlib
        run_module = importlib.import_module(
            "bio_programming_tools.tools.gene_annotation.crispr_tracr.standalone.run"
        )

        # Mock _find_crispr_tracr_script to return a fake path
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

            # Verify subprocess.run was called with cwd set to the worker dir
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("cwd") == str(fake_worker_cwd), (
                "subprocess.run must set cwd to an isolated worker directory "
                "so relative paths resolve correctly and parallel workers don't conflict"
            )

    def test_setup_installs_crispridentify(self):
        """setup.sh must clone CRISPRidentify into the correct relative path."""
        setup_sh = Path(__file__).resolve().parents[2] / (
            "bio_programming_tools/tools/gene_annotation/"
            "crispr_tracr/standalone/setup.sh"
        )
        content = setup_sh.read_text()
        assert "CRISPRidentify" in content, "setup.sh must install CRISPRidentify"
        assert "tools/CRISPRidentify/CRISPRidentify" in content, (
            "CRISPRidentify must be cloned into tools/CRISPRidentify/CRISPRidentify/ "
            "within the CRISPRtracrRNA installation"
        )

    def test_setup_installs_crisprcastidentifier(self):
        """setup.sh must clone CRISPRcasIdentifier into the correct relative path."""
        setup_sh = Path(__file__).resolve().parents[2] / (
            "bio_programming_tools/tools/gene_annotation/"
            "crispr_tracr/standalone/setup.sh"
        )
        content = setup_sh.read_text()
        assert "CRISPRcasIdentifier" in content, (
            "setup.sh must install CRISPRcasIdentifier"
        )

    def test_setup_uses_python38_sklearn022(self):
        """setup.sh must create conda_deps with Python 3.8 + sklearn 0.22.

        CRISPRidentify's pickled models require sklearn 0.22 which is
        incompatible with Python 3.12 / sklearn >= 1.3.  The conda_deps
        environment provides Python 3.8 so CRISPRidentify subprocesses
        (called via bare 'python') get the right runtime.
        """
        setup_sh = Path(__file__).resolve().parents[2] / (
            "bio_programming_tools/tools/gene_annotation/"
            "crispr_tracr/standalone/setup.sh"
        )
        content = setup_sh.read_text()
        assert "python=3.8" in content, (
            "setup.sh must install Python 3.8 in conda_deps for CRISPRidentify"
        )
        assert "scikit-learn=0.22" in content, (
            "setup.sh must install scikit-learn 0.22 in conda_deps for CRISPRidentify"
        )

    def test_setup_installs_conda_tools(self):
        """setup.sh must install bioinformatics tools and CRISPRidentify deps in conda_deps."""
        setup_sh = Path(__file__).resolve().parents[2] / (
            "bio_programming_tools/tools/gene_annotation/"
            "crispr_tracr/standalone/setup.sh"
        )
        content = setup_sh.read_text()
        for tool in ["intarna", "infernal", "prodigal", "hmmer", "viennarna", "vmatch", "clustalo", "blast", "fasta3"]:
            assert tool in content, f"setup.sh must install {tool} via conda"
        for dep in ["h5py", "dill", "networkx", "pyyaml", "regex", "requests"]:
            assert dep in content, (
                f"setup.sh must install {dep} in conda_deps for CRISPRidentify"
            )


# ============================================================================
# Integration Tests (require CRISPRtracrRNA)
# ============================================================================
class TestCrisprTracrIntegration:
    """Integration tests that require the CRISPRtracrRNA tool."""

    @pytest.mark.skip_ci
    def test_run_crispr_tracr(self):
        """Run CRISPRtracrRNA on a test sequence."""
        # Use a synthetic sequence
        test_seq = "ATCGATCG" * 500
        inputs = CrisprTracrInput(
            sequences=[test_seq],
            sequence_ids=["test_seq"],
        )
        config = CrisprTracrConfig(model_type="II")
        result = run_crispr_tracr(inputs, config)

        assert isinstance(result, CrisprTracrOutput)
        assert len(result.predictions) == 1

    @pytest.mark.skip_ci
    def test_run_crispr_tracr_model_type_all(self):
        """Run CRISPRtracrRNA with model_type='all'."""
        test_seq = "ATCGATCG" * 500
        inputs = CrisprTracrInput(sequences=[test_seq])
        config = CrisprTracrConfig(model_type="all")
        result = run_crispr_tracr(inputs, config)

        assert isinstance(result, CrisprTracrOutput)
        assert len(result.predictions) == 1

    @staticmethod
    def _read_fasta_sequence(fasta_path: Path) -> str:
        """Read a single-record FASTA file and return the sequence string."""
        lines = []
        with open(fasta_path) as f:
            for line in f:
                if not line.startswith(">"):
                    lines.append(line.strip())
        return "".join(lines)

    @pytest.mark.skip_ci
    def test_run_crispr_tracr_real_sequence(self):
        """Positive control: Listeria monocytogenes CRISPR locus.

        AAAABU010000051 is a ~12 kb contig with a known Type II CRISPR
        array.  The tool must detect a tracrRNA with high anti-repeat
        similarity.
        """
        test_fasta = DATA_DIR / "AAAABU010000051.fasta"
        seq = self._read_fasta_sequence(test_fasta)

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

    @pytest.mark.skip_ci
    def test_positive_control_spyogenes_sf370(self):
        """Positive control: S. pyogenes SF370 canonical CRISPR-Cas9 locus.

        NC_002737:849000-875000 contains the Type II-A CRISPR-Cas9 locus
        whose tracrRNA was first characterized by Deltcheva et al. 2011.
        The tool must detect the canonical sgRNA scaffold sequence.
        """
        test_fasta = DATA_DIR / "NC_002737_849000_875000.fasta"
        seq = self._read_fasta_sequence(test_fasta)

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
