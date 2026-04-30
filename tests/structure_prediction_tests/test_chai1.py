"""tests/structure_prediction_tests/test_chai1.py.

Benchmark tests for Chai1 structure prediction.

Cross-tool integration coverage lives in ``test_structure_prediction.py``;
this file holds the cold/warm benchmark and any Chai1-specific tests.
"""

import pytest

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Chai1Config,
    Chai1Input,
    run_chai1,
)
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec


@pytest.mark.benchmark("chai1-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_chai1_benchmark(request):
    """Benchmark chai1-prediction on the MfnG protein + L-tyrosine ligand (cold + warm).

    Single ~390-residue protein-ligand complex — a representative target for the
    Chai1 multi-modal predictor without MSA.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = Chai1Input(complexes=[complex_])
    config = Chai1Config(use_msa=False, verbose=True)

    result = benchmark_twice(request, "chai1", lambda: run_chai1(inputs=inputs, config=config))

    assert result.success, "Chai1 benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
