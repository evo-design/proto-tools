"""tests/structure_prediction_tests/test_esmfold.py.

Benchmark tests for ESMFold structure prediction.

Cross-tool integration coverage lives in ``test_structure_prediction.py``;
this file holds the cold/warm benchmark and any ESMFold-specific tests.
"""

import pytest

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    ESMFoldConfig,
    ESMFoldInput,
    StructurePredictionComplex,
    run_esmfold,
)
from tests.conftest import benchmark_twice, random_protein_sequences


@pytest.mark.benchmark("esmfold-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold_benchmark(request):
    """Benchmark esmfold-prediction on 10 random 300-residue proteins (cold + warm).

    Mirrors a realistic batched ESMFold screen of ~10 designs at typical
    single-domain length. ``max_batch_residues=4096`` lets the entire 3000-residue
    workload pack into a single GPU forward pass on a modern accelerator —
    exercising the high-throughput batched path rather than the default ``1200``
    which would split into three sub-batches.
    """
    sequences = random_protein_sequences(n=10, length=300, seed=0)
    complexes = [StructurePredictionComplex(chains=[seq]) for seq in sequences]
    inputs = ESMFoldInput(complexes=complexes)
    config = ESMFoldConfig(max_batch_residues=4096)

    result = benchmark_twice(request, "esmfold", lambda: run_esmfold(inputs=inputs, config=config))

    assert result.success, "ESMFold benchmark run failed"
    assert len(result.structures) == 10
    for structure in result.structures:
        assert is_valid_structure(structure.structure_cif)
