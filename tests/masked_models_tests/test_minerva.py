"""Tests for Minerva mixed protein/DNA masked language models."""

import numpy as np
import pytest
from pydantic import ValidationError
from standalone_helpers.mixed_sequence import tokenize_mixed_sequence

from proto_tools.tools import MinervaInteractionsConfig
from tests.conftest import make_persistent_fixture
from tests.masked_models_tests._mixed_checks import (
    LOCUS,
    benchmark_operation,
    call_model,
    check_embeddings_score_gradient,
    check_gradient_finite_difference,
    check_sampling,
    check_upstream_pll,
    upstream_reference,
)

_persistent_tool = make_persistent_fixture("minerva")
_CHECKPOINT = "gbrixi/minerva-mlm"


def test_minerva_interactions_rejects_duplicate_heads():
    with pytest.raises(ValidationError, match="duplicates"):
        MinervaInteractionsConfig(heads=["protein", "protein"])


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_embeddings_score_and_gradient_persistent_worker():
    check_embeddings_score_gradient("minerva", _CHECKPOINT)


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("sampling_method", ["single_pass", "iterative_refinement"])
def test_minerva_sampling_preserves_modalities_and_advances_rng(sampling_method):
    check_sampling("minerva", _CHECKPOINT, sampling_method)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_gradient_matches_finite_difference():
    check_gradient_finite_difference("minerva", _CHECKPOINT)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_score_matches_native_upstream():
    check_upstream_pll("minerva", _CHECKPOINT)


@pytest.mark.benchmark("minerva-embedding")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_embeddings_benchmark(request):
    benchmark_operation(request, "minerva", _CHECKPOINT, "embeddings")


@pytest.mark.benchmark("minerva-score")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_score_benchmark(request):
    benchmark_operation(request, "minerva", _CHECKPOINT, "score")


@pytest.mark.benchmark("minerva-sample")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_sample_benchmark(request):
    benchmark_operation(request, "minerva", _CHECKPOINT, "sample")


@pytest.mark.benchmark("minerva-gradient")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_gradient_benchmark(request):
    benchmark_operation(request, "minerva", _CHECKPOINT, "gradient")


@pytest.mark.benchmark("minerva-interactions")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_minerva_interactions_benchmark(request):
    benchmark_operation(request, "minerva", _CHECKPOINT, "interactions")


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("interaction_layers", [2, 6])
def test_minerva_interactions_matches_native_upstream(interaction_layers):
    reference = upstream_reference("minerva", _CHECKPOINT, LOCUS, interaction_layers)
    output = call_model(
        "minerva", "interactions", _CHECKPOINT, {"sequences": LOCUS}, interaction_layers=interaction_layers
    )
    maps = output.results[0].maps
    assert set(maps) == {"protein", "base_pairing", "repeat"}
    tokens = tokenize_mixed_sequence(LOCUS)
    for head, contact_map in maps.items():
        values = np.asarray(contact_map.values)
        assert contact_map.axis_labels == tokens
        assert values.shape == (len(tokens), len(tokens))
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()
        np.testing.assert_allclose(values, values.T, atol=1e-6)
        np.testing.assert_allclose(values, reference["maps"][head], rtol=1e-4, atol=1e-5)
    subset = call_model(
        "minerva",
        "interactions",
        _CHECKPOINT,
        {"sequences": LOCUS},
        heads=["repeat"],
        interaction_layers=interaction_layers,
    )
    assert set(subset.results[0].maps) == {"repeat"}
    np.testing.assert_allclose(subset.results[0].maps["repeat"].values, maps["repeat"].values)
