"""Tests for structure metrics."""

import json
from pathlib import Path

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.structure_metrics import (
    StructureMetricsConfig,
    StructureMetricsInput,
    StructureMetricsOutput,
    StructureQualityMetrics,
    run_structure_metrics,
)
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
    validate_output,
)

_FIXTURE_PDB = (
    Path(__file__).parent.parent.parent
    / "proto_tools"
    / "tools"
    / "structure_scoring"
    / "structure_metrics"
    / "example_input_fixture.pdb"
)


# ── Input validation ─────────────────────────────────────────────────────────


def test_input_single_path_wrapped_in_list():
    inp = StructureMetricsInput(structures=str(_FIXTURE_PDB))
    assert isinstance(inp.structures, list)
    assert len(inp.structures) == 1
    assert isinstance(inp.structures[0], Structure)


def test_input_list_of_paths_preserved():
    inp = StructureMetricsInput(structures=[str(_FIXTURE_PDB), str(_FIXTURE_PDB)])
    assert len(inp.structures) == 2
    assert all(isinstance(s, Structure) for s in inp.structures)


def test_input_path_object_coerced_to_structure():
    inp = StructureMetricsInput(structures=[_FIXTURE_PDB])
    assert isinstance(inp.structures[0], Structure)


def test_input_structure_object_passes_through():
    s = Structure.from_file(_FIXTURE_PDB)
    inp = StructureMetricsInput(structures=[s])
    assert inp.structures[0].structure == s.structure


# ── Config ───────────────────────────────────────────────────────────────────


def test_config_extra_fields_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StructureMetricsConfig(extra_field="should_be_rejected")


# ── StructureQualityMetrics data model ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "longest_alpha_helix,gyration_radius",
    [
        (0, 0.0),
        (25, 30.5),
        (100, 99.9),
    ],
)
def test_structure_metrics_model_dump(longest_alpha_helix, gyration_radius):
    m = StructureQualityMetrics(
        longest_alpha_helix=longest_alpha_helix,
        gyration_radius=gyration_radius,
    )
    d = m.model_dump()
    assert d["longest_alpha_helix"] == longest_alpha_helix
    assert d["gyration_radius"] == gyration_radius


# ── Export ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sample_output():
    return StructureMetricsOutput(
        metadata={"num_structures": 2},
        metrics=[
            StructureQualityMetrics(
                longest_alpha_helix=15,
                gyration_radius=28.3,
            ),
            StructureQualityMetrics(
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

    result = run_structure_metrics(StructureMetricsInput(structures=[str(pdb_path)]))
    assert_metrics_in_spec(result)

    assert isinstance(result, StructureMetricsOutput)
    assert len(result.metrics) == 1


# ── Benchmark ─────────────────────────────────────────────────────────────────

_BENCH_RENIN_PDB = str(Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb")


@pytest.mark.benchmark("structure-metrics")
@pytest.mark.slow
def test_structure_metrics_benchmark(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    """Benchmark structure-metrics: 50 distinct renin_af3 copies (~340 aa each), all metrics enabled (cold + warm).

    structure-metrics is an in-process Biotite/gemmi compute tool with no
    persistent worker, so we time both passes directly. Distinct file paths
    break the @tool iterable-input dedup.
    """
    import shutil
    import time

    structure_paths = []
    for i in range(50):
        p = tmp_path / f"renin_{i:03d}.pdb"
        shutil.copyfile(_BENCH_RENIN_PDB, p)
        structure_paths.append(str(p))
    inputs = StructureMetricsInput(structures=structure_paths)
    runner = lambda: run_structure_metrics(inputs)  # noqa: E731

    t0 = time.perf_counter()
    _ = runner()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = runner()
    warm = time.perf_counter() - t0
    request.node.user_properties.append(("cold_seconds", cold))
    request.node.user_properties.append(("warm_seconds", warm))

    validate_output(result)
    assert result.tool_id == "structure-metrics"
    assert len(result.metrics) == 50
    for m in result.metrics:
        assert m.gyration_radius > 0
        assert m.longest_alpha_helix >= 0
