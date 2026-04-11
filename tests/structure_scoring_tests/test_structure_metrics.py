"""Tests for structure metrics."""

import json
from pathlib import Path

import pytest

from proto_tools.tools.structure_scoring.structure_metrics import (
    StructureMetrics,
    StructureMetricsConfig,
    StructureMetricsInput,
    StructureMetricsOutput,
    run_structure_metrics,
)
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
)

# ── Input validation ─────────────────────────────────────────────────────────


def test_input_single_path_normalized_to_list():
    inp = StructureMetricsInput(pdb_paths="/path/to/structure.pdb")
    assert isinstance(inp.pdb_paths, list)
    assert len(inp.pdb_paths) == 1
    assert inp.pdb_paths[0] == "/path/to/structure.pdb"


def test_input_list_of_paths_preserved():
    inp = StructureMetricsInput(pdb_paths=["/path/a.pdb", "/path/b.pdb"])
    assert len(inp.pdb_paths) == 2


def test_input_path_objects_converted_to_strings():
    inp = StructureMetricsInput(pdb_paths=[Path("/path/to/structure.pdb")])
    assert isinstance(inp.pdb_paths[0], str)


# ── Config ───────────────────────────────────────────────────────────────────


def test_config_extra_fields_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StructureMetricsConfig(extra_field="should_be_rejected")


# ── StructureMetrics data model ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "longest_alpha_helix,gyration_radius",
    [
        (0, 0.0),
        (25, 30.5),
        (100, 99.9),
    ],
)
def test_structure_metrics_model_dump(longest_alpha_helix, gyration_radius):
    m = StructureMetrics(
        pdb_path="/path/test.pdb",
        longest_alpha_helix=longest_alpha_helix,
        gyration_radius=gyration_radius,
    )
    d = m.model_dump()
    assert d["pdb_path"] == "/path/test.pdb"
    assert d["longest_alpha_helix"] == longest_alpha_helix
    assert d["gyration_radius"] == gyration_radius


# ── Export ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sample_output():
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


def test_export_csv(sample_output, tmp_path):
    sample_output.export(name="metrics", export_path=str(tmp_path), file_format="csv")
    assert validate_export_output(tmp_path / "metrics.csv")


def test_export_json(sample_output, tmp_path):
    sample_output.export(name="metrics", export_path=str(tmp_path), file_format="json")
    json_path = tmp_path / "metrics.json"
    assert validate_export_output(json_path)
    data = json.loads(json_path.read_text())
    assert len(data) == 2


def test_output_format_options(sample_output):
    assert "csv" in sample_output.output_format_options
    assert "json" in sample_output.output_format_options
    assert sample_output.output_format_default == "csv"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_structure_metrics_on_pdb(tmp_path):
    """Run structure metrics on a minimal PDB file."""
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

    result = run_structure_metrics(StructureMetricsInput(pdb_paths=[str(pdb_path)]))

    assert isinstance(result, StructureMetricsOutput)
    assert len(result.metrics) == 1
    assert result.metrics[0].gyration_radius >= 0.0
    assert result.metrics[0].longest_alpha_helix >= 0
