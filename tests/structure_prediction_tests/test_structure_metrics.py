"""
Tests for the Structure Metrics tool.

Unit tests for data models and input normalization, plus integration
tests (skip_ci) for actual biotite-based structure analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bio_programming_tools.tools.structure_prediction.structure_metrics import (
    StructureMetrics,
    StructureMetricsConfig,
    StructureMetricsInput,
    StructureMetricsOutput,
    run_structure_metrics,
)
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
)


# ============================================================================
# Data Model Tests
# ============================================================================
class TestStructureMetricsInput:
    """Tests for StructureMetricsInput validation and normalization."""

    def test_single_path_normalization(self):
        """Single string path should be normalized to list."""
        inp = StructureMetricsInput(pdb_paths="/path/to/structure.pdb")
        assert isinstance(inp.pdb_paths, list)
        assert len(inp.pdb_paths) == 1
        assert inp.pdb_paths[0] == "/path/to/structure.pdb"

    def test_list_of_paths(self):
        """List of paths should be preserved."""
        inp = StructureMetricsInput(pdb_paths=["/path/a.pdb", "/path/b.pdb"])
        assert len(inp.pdb_paths) == 2

    def test_path_objects_converted_to_strings(self):
        """Path objects should be converted to strings."""
        inp = StructureMetricsInput(pdb_paths=[Path("/path/to/structure.pdb")])
        assert isinstance(inp.pdb_paths[0], str)


class TestStructureMetricsConfig:
    """Tests for StructureMetricsConfig."""

    def test_defaults(self):
        """Config should work with no arguments."""
        config = StructureMetricsConfig()
        assert config is not None

    def test_extra_ignored(self):
        """Extra fields should be ignored."""
        config = StructureMetricsConfig(extra_field="should_be_ignored")
        assert config is not None


class TestStructureMetricsModel:
    """Tests for the StructureMetrics Pydantic model."""

    def test_construction(self):
        """StructureMetrics should be constructable with required fields."""
        m = StructureMetrics(
            pdb_path="/path/test.pdb",
            longest_alpha_helix=25,
            gyration_radius=30.5,
        )
        assert m.pdb_path == "/path/test.pdb"
        assert m.longest_alpha_helix == 25
        assert m.gyration_radius == 30.5

    def test_serialization(self):
        """Model should be serializable."""
        m = StructureMetrics(
            pdb_path="/path/test.pdb",
            longest_alpha_helix=10,
            gyration_radius=20.0,
        )
        d = m.model_dump()
        assert d["pdb_path"] == "/path/test.pdb"
        assert d["longest_alpha_helix"] == 10
        assert d["gyration_radius"] == 20.0


# ============================================================================
# Export Tests
# ============================================================================
class TestStructureMetricsExport:
    """Tests for StructureMetricsOutput export functionality."""

    @pytest.fixture
    def sample_output(self):
        return StructureMetricsOutput(
            metadata={"num_structures": 2},
            metrics=[
                StructureMetrics(
                    pdb_path="/path/a.pdb",
                    longest_alpha_helix=15,
                    gyration_radius=28.3,
                ),
                StructureMetrics(
                    pdb_path="/path/b.pdb",
                    longest_alpha_helix=45,
                    gyration_radius=52.1,
                ),
            ],
        )

    def test_export_csv(self, sample_output, tmp_path):
        """Export to CSV format."""
        sample_output.export(
            name="metrics", export_path=str(tmp_path), file_format="csv"
        )
        csv_path = tmp_path / "metrics.csv"
        assert validate_export_output(csv_path)

    def test_export_json(self, sample_output, tmp_path):
        """Export to JSON format."""
        sample_output.export(
            name="metrics", export_path=str(tmp_path), file_format="json"
        )
        json_path = tmp_path / "metrics.json"
        assert validate_export_output(json_path)

        data = json.loads(json_path.read_text())
        assert len(data) == 2

    def test_output_format_options(self, sample_output):
        """Check supported output formats."""
        assert "csv" in sample_output.output_format_options
        assert "json" in sample_output.output_format_options
        assert sample_output.output_format_default == "csv"


# ============================================================================
# Integration Tests (require biotite)
# ============================================================================
class TestStructureMetricsIntegration:
    """Integration tests that require biotite and real PDB files."""

    @pytest.mark.skip_ci
    def test_run_structure_metrics_on_pdb(self, tmp_path):
        """Run structure metrics on a minimal PDB file."""
        # Create a minimal PDB file with a few atoms
        pdb_content = """\
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   1.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   2.000   1.000  1.00  0.00           O
ATOM      5  N   ALA A   2       4.000   1.000   1.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       5.000   1.000   1.000  1.00  0.00           C
ATOM      7  C   ALA A   2       6.000   1.000   1.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.000   2.000   1.000  1.00  0.00           O
END
"""
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content)

        inputs = StructureMetricsInput(pdb_paths=[str(pdb_path)])
        config = StructureMetricsConfig()
        result = run_structure_metrics(inputs, config)

        assert isinstance(result, StructureMetricsOutput)
        assert len(result.metrics) == 1
        assert result.metrics[0].gyration_radius >= 0.0
        assert result.metrics[0].longest_alpha_helix >= 0
