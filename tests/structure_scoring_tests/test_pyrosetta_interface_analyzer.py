"""Tests for PyRosetta interface analyzer scoring tool."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_interface_analyzer import (
    InterfaceStructureInput,
    PyRosettaInterfaceAnalyzerConfig,
    PyRosettaInterfaceAnalyzerInput,
    run_pyrosetta_interface_analyzer,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import PyRosettaRelaxConfig
from tests.conftest import benchmark_twice
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB = str(Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb")

# ── Validation ────────────────────────────────────────────────────────────────


def test_interface_analyzer_input_normalizes_single_structure():
    """Bare Structure / path / dict gets wrapped into a list by the custom validator."""
    structure = Structure(structure=TEST_PDB)
    inp = PyRosettaInterfaceAnalyzerInput(inputs=structure)
    assert len(inp.inputs) == 1
    assert isinstance(inp.inputs[0], InterfaceStructureInput)
    assert inp.inputs[0].target_chain == "A"
    assert inp.inputs[0].binder_chain == "B"


def test_interface_analyzer_rejects_missing_chain():
    """target_chain/binder_chain absent from the structure raises at input construction."""
    with pytest.raises(ValidationError, match="not found in structure"):
        PyRosettaInterfaceAnalyzerInput(inputs=[{"structure": TEST_PDB, "target_chain": "Z", "binder_chain": "B"}])


def test_interface_analyzer_rejects_equal_chains():
    """target_chain == binder_chain is rejected — an interface requires two distinct chains."""
    with pytest.raises(ValidationError, match="must differ"):
        PyRosettaInterfaceAnalyzerInput(inputs=[{"structure": TEST_PDB, "target_chain": "A", "binder_chain": "A"}])


def test_interface_analyzer_input_accepts_multiple_structures():
    """A list of multiple structures constructs correctly and preserves per-input chain labels."""
    inp = PyRosettaInterfaceAnalyzerInput(
        inputs=[
            TEST_PDB,
            {"structure": TEST_PDB, "target_chain": "B", "binder_chain": "A"},
        ],
    )
    assert len(inp.inputs) == 2
    assert inp.inputs[0].target_chain == "A"
    assert inp.inputs[0].binder_chain == "B"
    assert inp.inputs[1].target_chain == "B"
    assert inp.inputs[1].binder_chain == "A"


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_run_pyrosetta_interface_analyzer_on_pdb():
    """Default path: analyze the interface between chains A and B on pdl1."""
    result = run_pyrosetta_interface_analyzer(PyRosettaInterfaceAnalyzerInput(inputs=[TEST_PDB]))
    assert_metrics_in_spec(result)

    assert result.success

    m = result.results[0]
    assert isinstance(m.interface_sc, float) and 0.0 <= m.interface_sc <= 1.0
    assert isinstance(m.interface_hbonds, int) and m.interface_hbonds >= 0
    assert isinstance(m.interface_dG, float)
    assert isinstance(m.interface_dSASA, float) and m.interface_dSASA > 0.0
    assert isinstance(m.interface_packstat, float) and 0.0 <= m.interface_packstat <= 1.0
    assert isinstance(m.interface_hydrophobicity, float) and 0.0 <= m.interface_hydrophobicity <= 100.0
    assert isinstance(m.surface_hydrophobicity, float) and 0.0 <= m.surface_hydrophobicity <= 1.0
    if "delta_unsat_hbonds" in m:
        assert isinstance(m.delta_unsat_hbonds, int) and m.delta_unsat_hbonds >= 0


@pytest.mark.integration
def test_interface_analyzer_with_pre_relax_preprocess():
    """Setting pre_relax_structures=True should shift interface_dG meaningfully.

    FastRelax resolves steric clashes in raw predicted complexes, so the
    relaxed interface_dG should differ from the raw value by more than a
    numerical-noise floor (0.1 REU).
    """
    raw = run_pyrosetta_interface_analyzer(PyRosettaInterfaceAnalyzerInput(inputs=[TEST_PDB]))
    relaxed = run_pyrosetta_interface_analyzer(
        PyRosettaInterfaceAnalyzerInput(inputs=[TEST_PDB]),
        PyRosettaInterfaceAnalyzerConfig(
            pre_relax_structures=True,
            relax_config=PyRosettaRelaxConfig(relax_cycles=1, seed=42),
        ),
    )
    assert raw.success and relaxed.success
    assert abs(raw.results[0].interface_dG - relaxed.results[0].interface_dG) > 0.1, (
        f"interface_dG unchanged after relax: raw={raw.results[0].interface_dG}, "
        f"relaxed={relaxed.results[0].interface_dG}"
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("pyrosetta-interface-analyzer")
@pytest.mark.slow
def test_pyrosetta_interface_analyzer_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pyrosetta-interface-analyzer: 3 distinct pdl1 copies (multi-chain interface) (cold + warm)."""
    structures = [{"structure": Structure(structure=TEST_PDB, metrics={"_bench_id": i})} for i in range(3)]
    inputs = PyRosettaInterfaceAnalyzerInput(inputs=structures)

    result = benchmark_twice(request, "pyrosetta", lambda: run_pyrosetta_interface_analyzer(inputs))
    validate_output(result)

    assert result.tool_id == "pyrosetta-interface-analyzer"
    assert len(result.results) == 3
    for r in result.results:
        assert isinstance(r.interface_dG, float)
