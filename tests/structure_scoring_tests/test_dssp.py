"""Tests for DSSP secondary-structure assignment models."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import Structure
from proto_tools.tools import ToolRegistry
from proto_tools.tools.structure_scoring.dssp import (
    DSSPSecondaryStructureInput,
    DSSPSecondaryStructureMetrics,
    DSSPSecondaryStructureOutput,
    DSSPStructureInput,
    run_dssp_secondary_structure,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "proto_tools/tools/structure_scoring/structure_metrics/example_input_fixture.pdb"
)


def test_dssp_tool_is_registered() -> None:
    """ToolRegistry exposes DSSP with a valid example input."""
    spec = ToolRegistry.get("dssp-secondary-structure")

    assert spec.key == "dssp-secondary-structure"
    assert spec.category == "structure_scoring"
    assert spec.input_model is DSSPSecondaryStructureInput
    assert spec.example_input is not None
    example = spec.example_input()
    assert isinstance(example, DSSPSecondaryStructureInput)
    assert ToolRegistry.get_links("dssp-secondary-structure") == {"github": "https://github.com/PDB-REDO/dssp"}


def test_dssp_input_accepts_single_path_and_defaults_chain() -> None:
    """Path inputs are wrapped, coerced to Structure, and default to chain A."""
    inp = DSSPSecondaryStructureInput(inputs=FIXTURE)  # type: ignore[arg-type]

    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0].structure, Structure)
    assert inp.inputs[0].chain_id == "A"


def test_dssp_structure_input_rejects_missing_chain() -> None:
    """DSSPStructureInput validates that the requested chain exists."""
    with pytest.raises(ValidationError, match="Chain 'Z' not found"):
        DSSPStructureInput(structure=FIXTURE, chain_id="Z")  # type: ignore[arg-type]


def test_export_dssp_output_csv_and_json(tmp_path: Path) -> None:
    """DSSP output exports metric values plus chain_id in tabular formats."""
    output = DSSPSecondaryStructureOutput(
        results=[DSSPSecondaryStructureMetrics(chain_id="A", helix_pct=50.0, sheet_pct=25.0, loop_pct=25.0)]
    )

    output.export("dssp", tmp_path, file_format="csv")
    output.export("dssp", tmp_path, file_format="json")

    with (tmp_path / "dssp.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    json_rows = json.loads((tmp_path / "dssp.json").read_text())

    assert csv_rows == [{"helix_pct": "50.0", "sheet_pct": "25.0", "loop_pct": "25.0", "chain_id": "A"}]
    assert json_rows == [{"helix_pct": 50.0, "sheet_pct": 25.0, "loop_pct": 25.0, "chain_id": "A"}]


# ── Benchmark ─────────────────────────────────────────────────────────────────

_BENCH_RENIN_PDB = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.mark.benchmark("dssp-secondary-structure")
@pytest.mark.slow
def test_dssp_secondary_structure_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark dssp-secondary-structure: 50 distinct renin_af3 copies (~340 aa each) (cold + warm)."""
    # Distinct metrics break the @tool iterable-input dedup that would otherwise
    # collapse N identical structures to a single compute, defeating the benchmark.
    structures = [
        {"structure": Structure(structure=str(_BENCH_RENIN_PDB), metrics={"_bench_id": i})} for i in range(50)
    ]
    inputs = DSSPSecondaryStructureInput(inputs=structures)

    result = benchmark_twice(request, "dssp", lambda: run_dssp_secondary_structure(inputs))
    validate_output(result)

    assert result.tool_id == "dssp-secondary-structure"
    assert len(result.results) == 50
    for r in result.results:
        assert 0.0 <= r.helix_pct <= 100.0
        assert 0.0 <= r.sheet_pct <= 100.0
        assert 0.0 <= r.loop_pct <= 100.0
