"""
Tests for the Gap Gini alignment quality tool.

Tests internal utility functions (_gini, _gap_runs, _gap_gini_single)
and the end-to-end run_gap_gini tool.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bio_programming_tools.tools.sequence_alignment.gap_gini import (
    GapGiniConfig,
    GapGiniInput,
    GapGiniOutput,
    run_gap_gini,
)
from bio_programming_tools.tools.sequence_alignment.gap_gini.gap_gini import (
    _gap_gini_single,
    _gap_runs,
    _gini,
)
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
)


# ============================================================================
# Tests for _gini()
# ============================================================================
class TestGini:
    """Unit tests for the Gini coefficient computation."""

    def test_all_equal(self):
        """All-equal array should give Gini = 0."""
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        assert _gini(arr) == pytest.approx(0.0)

    def test_all_equal_ones(self):
        """Array of ones should give Gini = 0."""
        arr = np.ones(10)
        assert _gini(arr) == pytest.approx(0.0)

    def test_maximally_unequal(self):
        """One large value among zeros should give high Gini."""
        arr = np.array([0.0, 0.0, 0.0, 100.0])
        gini = _gini(arr)
        assert gini > 0.5

    def test_empty_array(self):
        """Empty array should return 0."""
        assert _gini(np.array([])) == 0.0

    def test_all_zeros(self):
        """All-zero array should return 0 (mean is 0)."""
        assert _gini(np.zeros(5)) == 0.0

    def test_single_element(self):
        """Single element array should return 0."""
        assert _gini(np.array([42.0])) == 0.0

    def test_known_value(self):
        """Known Gini coefficient for [1, 2, 3, 4, 5]."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gini = _gini(arr)
        # Gini for uniform distribution [1..5] is 4/15 ≈ 0.2667
        assert 0.2 < gini < 0.35

    def test_gini_range(self):
        """Gini coefficient should always be in [0, 1]."""
        for _ in range(20):
            arr = np.random.exponential(scale=1.0, size=50)
            g = _gini(arr)
            assert 0.0 <= g <= 1.0


# ============================================================================
# Tests for _gap_runs()
# ============================================================================
class TestGapRuns:
    """Unit tests for gap run-length extraction."""

    def test_no_gaps(self):
        """Sequence without gaps should give all run lengths of 1."""
        runs = _gap_runs("ACGT")
        assert all(r == 1 for r in runs)

    def test_single_gap(self):
        """Sequence with single gap character."""
        runs = _gap_runs("AC-GT")
        assert 1 in runs  # Non-gap runs are 1

    def test_consecutive_gaps(self):
        """Consecutive gaps should be counted as one long run."""
        runs = _gap_runs("AC---GT")
        assert 3 in runs  # Three consecutive gaps

    def test_empty_sequence(self):
        """Empty sequence returns empty list."""
        assert _gap_runs("") == []

    def test_all_gaps(self):
        """All-gap sequence should give one long run."""
        runs = _gap_runs("-----")
        assert max(runs) == 5

    def test_alternating_gaps(self):
        """Alternating gaps and residues."""
        runs = _gap_runs("A-C-G-T")
        # Each gap and non-gap is length 1
        assert all(r == 1 for r in runs)

    def test_terminal_gaps(self):
        """Gaps at the start and end of sequence."""
        runs = _gap_runs("--ACG--")
        assert 2 in runs  # Terminal gap run of length 2


# ============================================================================
# Tests for _gap_gini_single()
# ============================================================================
class TestGapGiniSingle:
    """Unit tests for single pairwise alignment Gini computation."""

    def test_perfect_alignment_no_gaps(self):
        """Perfectly aligned sequences (no gaps) should have low Gini."""
        alignment = ">seq1\nACGTACGT\n>seq2\nACGTACGT\n"
        score = _gap_gini_single(alignment)
        assert score == pytest.approx(0.0)

    def test_truncation_artifact(self):
        """Concentrated gaps (truncation) should have high Gini."""
        # One sequence is much shorter, padded with gaps at the end
        alignment = ">seq1\nACGTACGTACGT--------\n>seq2\nACGTACGTACGTACGTACGT\n"
        score = _gap_gini_single(alignment)
        assert score > 0.3  # High Gini = concentrated gaps

    def test_evenly_distributed_gaps(self):
        """Evenly distributed gaps should have low Gini."""
        alignment = ">seq1\nA-C-G-T-A-C-G-T-\n>seq2\nAACCGGTTAACCGGTT\n"
        score = _gap_gini_single(alignment)
        assert score < 0.15  # Low Gini = even gaps

    def test_wrong_number_of_sequences(self):
        """Should raise ValueError with != 2 sequences."""
        alignment = ">seq1\nACGT\n"
        with pytest.raises(ValueError, match="Expected 2 sequences"):
            _gap_gini_single(alignment)

    def test_three_sequences_error(self):
        """Three sequences should raise ValueError."""
        alignment = ">seq1\nACGT\n>seq2\nACGT\n>seq3\nACGT\n"
        with pytest.raises(ValueError, match="Expected 2 sequences"):
            _gap_gini_single(alignment)


# ============================================================================
# Tests for run_gap_gini() end-to-end
# ============================================================================
class TestRunGapGini:
    """End-to-end tests for the gap Gini tool."""

    def test_single_alignment(self):
        """Run on a single alignment."""
        alignment = ">seq1\nACGT--ACGT\n>seq2\nACGTACACGT\n"
        inputs = GapGiniInput(alignments=[alignment])
        config = GapGiniConfig()
        result = run_gap_gini(inputs, config)

        assert isinstance(result, GapGiniOutput)
        assert len(result.gini_scores) == 1
        assert 0.0 <= result.gini_scores[0] <= 1.0

    def test_multiple_alignments(self):
        """Run on multiple alignments."""
        al1 = ">s1\nACGTACGT\n>s2\nACGTACGT\n"
        al2 = ">s3\nACGT----ACGT\n>s4\nACGTACGTACGT\n"
        inputs = GapGiniInput(alignments=[al1, al2])
        config = GapGiniConfig()
        result = run_gap_gini(inputs, config)

        assert len(result.gini_scores) == 2
        # First alignment (no gaps) should have lower score than second
        assert result.gini_scores[0] <= result.gini_scores[1]

    def test_input_normalization_single_string(self):
        """Single string input should be normalized to list."""
        alignment = ">seq1\nACGTACGT\n>seq2\nACGTACGT\n"
        inputs = GapGiniInput(alignments=alignment)
        assert isinstance(inputs.alignments, list)
        assert len(inputs.alignments) == 1

    def test_config_defaults(self):
        """GapGiniConfig should accept no arguments."""
        config = GapGiniConfig()
        assert config is not None

    def test_metadata(self):
        """Output should include metadata."""
        alignment = ">seq1\nACGT\n>seq2\nACGT\n"
        result = run_gap_gini(
            GapGiniInput(alignments=[alignment]),
            GapGiniConfig(),
        )
        assert result.metadata["num_alignments"] == 1


# ============================================================================
# Tests for output export
# ============================================================================
class TestGapGiniExport:
    """Tests for GapGiniOutput export functionality."""

    @pytest.fixture
    def sample_output(self):
        return GapGiniOutput(
            metadata={"num_alignments": 3},
            gini_scores=[0.05, 0.42, 0.78],
        )

    def test_export_csv(self, sample_output, tmp_path):
        """Export to CSV format."""
        sample_output.export(
            name="gap_gini", export_path=str(tmp_path), file_format="csv"
        )
        csv_path = tmp_path / "gap_gini.csv"
        assert validate_export_output(csv_path)

    def test_export_json(self, sample_output, tmp_path):
        """Export to JSON format."""
        sample_output.export(
            name="gap_gini", export_path=str(tmp_path), file_format="json"
        )
        json_path = tmp_path / "gap_gini.json"
        assert validate_export_output(json_path)

        data = json.loads(json_path.read_text())
        assert len(data) == 3

    def test_output_format_options(self, sample_output):
        """Check supported output formats."""
        assert "csv" in sample_output.output_format_options
        assert "json" in sample_output.output_format_options
        assert sample_output.output_format_default == "csv"
