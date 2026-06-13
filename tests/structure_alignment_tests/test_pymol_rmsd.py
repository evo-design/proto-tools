"""Tests for the PyMOL RMSD alignment tool."""

from pathlib import Path
from unittest.mock import patch

import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_alignment.pymol_rmsd import (
    PyMOLRMSDConfig,
    PyMOLRMSDInput,
    run_pymol_rmsd_alignment,
)
from proto_tools.utils import ToolInstance
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_DUMMY_DATA = Path(__file__).parent.parent / "dummy_data"
_PDB_PATH = _DUMMY_DATA / "test_structure_similarity.pdb"
_RENIN_PDB_PATH = _DUMMY_DATA / "renin_af3.pdb"
_RENIN_CIF_PATH = _DUMMY_DATA / "renin.cif"


def test_pymol_rmsd_config_method_is_literal_enum():
    schema = PyMOLRMSDConfig.model_json_schema()
    assert schema["properties"]["method"]["enum"] == ["cealign", "align"]


def test_pymol_rmsd_dispatches_cealign_payload():
    structure = Structure.from_file(_PDB_PATH)
    inputs = PyMOLRMSDInput(target_structure=structure, mobile_structure=structure)

    with patch("proto_tools.tools.structure_alignment.pymol_rmsd.pymol_rmsd.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = {
            "method": "cealign",
            "rmsd": 0.0,
            "aligned_length": 42,
        }
        result = run_pymol_rmsd_alignment(inputs, PyMOLRMSDConfig())

    validate_output(result)
    assert result.tool_id == "pymol-rmsd-alignment"
    assert result.method == "cealign"
    assert result.rmsd == 0.0
    assert result.aligned_length == 42
    assert "alignment_score" not in dict(result.metrics.items())
    mock_dispatch.assert_called_once()
    toolkit, payload = mock_dispatch.call_args.args[:2]
    assert toolkit == "pymol_rmsd"
    assert payload["method"] == "cealign"
    assert payload["target_pdb_text"] == structure.structure_pdb
    assert payload["mobile_pdb_text"] == structure.structure_pdb
    assert payload["target_selection"] == "target"
    assert payload["mobile_selection"] == "mobile"
    assert payload["device"] == "cpu"


def test_pymol_rmsd_dispatches_align_payload_with_selection_overrides():
    structure = Structure.from_file(_PDB_PATH)
    inputs = PyMOLRMSDInput(target_structure=structure, mobile_structure=structure)
    config = PyMOLRMSDConfig(
        method="align",
        target_selection="target and name CA",
        mobile_selection="mobile and name CA",
    )

    with patch("proto_tools.tools.structure_alignment.pymol_rmsd.pymol_rmsd.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = {
            "method": "align",
            "rmsd": 1.2,
            "aligned_atoms": 10,
            "alignment_cycles": 5,
            "pre_refinement_rmsd": 2.3,
            "pre_refinement_aligned_atoms": 12,
            "alignment_score": 11.0,
            "aligned_residues": 10,
        }
        result = run_pymol_rmsd_alignment(inputs, config)

    validate_output(result)
    assert result.method == "align"
    assert result.rmsd == 1.2
    assert result.aligned_atoms == 10
    assert result.alignment_cycles == 5
    assert result.pre_refinement_rmsd == 2.3
    assert result.pre_refinement_aligned_atoms == 12
    assert result.alignment_score == 11.0
    assert result.aligned_residues == 10
    toolkit, payload = mock_dispatch.call_args.args[:2]
    assert toolkit == "pymol_rmsd"
    assert payload["method"] == "align"
    assert payload["target_selection"] == "target and name CA"
    assert payload["mobile_selection"] == "mobile and name CA"


@pytest.mark.integration
def test_pymol_rmsd_e2e_self_alignment():
    """Run PyMOL through ToolInstance instead of mocking dispatch."""
    structure = Structure.from_file(_PDB_PATH)
    inputs = PyMOLRMSDInput(target_structure=structure, mobile_structure=structure)

    with ToolInstance.scope(), ToolInstance.persist_tool("pymol_rmsd"):
        cealign = run_pymol_rmsd_alignment(inputs, PyMOLRMSDConfig(method="cealign"))
        align = run_pymol_rmsd_alignment(inputs, PyMOLRMSDConfig(method="align"))

    validate_output(cealign)
    validate_output(align)
    assert cealign.rmsd == pytest.approx(0.0, abs=1e-4)
    assert cealign.aligned_length > 0
    assert "alignment_score" not in dict(cealign.metrics.items())
    assert align.rmsd == pytest.approx(0.0, abs=1e-4)
    assert align.aligned_atoms > 0
    assert align.aligned_residues > 0


@pytest.mark.benchmark("pymol-rmsd-alignment")
@pytest.mark.slow
def test_pymol_rmsd_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark pymol-rmsd-alignment: 20 sequential cealign of two renin models (~340 aa, RMSD ~0.6 A) (cold + warm)."""
    target = Structure.from_file(_RENIN_PDB_PATH)
    mobile = Structure.from_file(_RENIN_CIF_PATH)
    inputs = PyMOLRMSDInput(target_structure=target, mobile_structure=mobile)
    config = PyMOLRMSDConfig(method="cealign")

    def run_batch():
        last = None
        for _ in range(20):
            last = run_pymol_rmsd_alignment(inputs, config)
        return last

    result = benchmark_twice(request, "pymol_rmsd", run_batch)
    validate_output(result)

    assert result.tool_id == "pymol-rmsd-alignment"
    assert result.method == "cealign"
    assert 0.0 < result.rmsd < 2.0
    assert result.aligned_length > 250
