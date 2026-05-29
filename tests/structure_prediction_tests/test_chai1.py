"""tests/structure_prediction_tests/test_chai1.py.

Benchmark and unit tests for Chai1 structure prediction.

Cross-tool integration coverage lives in ``test_structure_prediction.py``;
this file holds the cold/warm benchmark, Chai-1's glycan-aware token counter,
and its token-budget validator. The shared structure-token counter is tested in
``test_shared_data_models.py``.
"""

import pytest

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Chai1Config,
    Chai1Input,
    run_chai1,
)
from proto_tools.tools.structure_prediction.chai1.helpers import count_chai1_tokens
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
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


# ============================================================================
# Token-budget validation (CPU-only, no model dispatch)
# ============================================================================
def test_count_chai1_tokens_glycan_sums_sugar_heavy_atoms():
    """Glycan sugars add their heavy-atom counts; other chains delegate to the shared counter."""
    chains = [
        Chain(sequence="MKTL", entity_type="protein"),
        Chain(sequence="MAN(6-1 FUC)(4-1 MAN)", entity_type="glycan"),
    ]
    assert count_chai1_tokens(chains) == 4 + 12 + 11 + 12  # protein 4 + MAN 12 + FUC 11 + MAN 12


def test_chai1_input_within_token_budget_passes():
    """A small protein + small ligand stays well under 2048 tokens."""
    Chai1Input(complexes=[Complex(chains=["MKTL", "CCO"])])
