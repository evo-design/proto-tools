"""
test_alphafold2.py

AlphaFold2-specific tests for code paths not covered by the shared
test_structure_prediction.py parametrized suite.
"""

from __future__ import annotations

import pytest

from bio_programming_tools.entities.structures import is_valid_structure
from bio_programming_tools.tools.structure_prediction import (
    AlphaFold2Config,
    AlphaFold2Input,
    StructurePredictionComplex,
    run_alphafold2,
)
from bio_programming_tools.utils.tool_instance import ToolInstance


@pytest.fixture(scope="module", autouse=True)
def _persistent_worker(request):
    if request.config.getoption("--cpu"):
        yield
        return
    with ToolInstance.scope():
        yield


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_homooligomer():
    """Verify the homo-oligomer code path (identical chains → copies)."""
    seq = "MARFLGLYTWHK"
    complexes = [StructurePredictionComplex(chains=[seq, seq])]

    inputs = AlphaFold2Input(complexes=complexes)
    config = AlphaFold2Config(use_msa=False, verbose=True)
    output = run_alphafold2(inputs, config)

    assert output.success
    assert len(output.structures) == 1

    structure = output.structures[0]
    assert is_valid_structure(structure.structure_cif)
    assert 0 <= structure.metrics["avg_plddt"] <= 1.0
    assert 0 <= structure.metrics["ptm"] <= 1.0
    assert structure.metrics["iptm"] is not None
