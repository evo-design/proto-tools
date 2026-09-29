"""Tests for GLM2 mixed protein/DNA masked language models."""

import pytest

from tests.conftest import make_persistent_fixture
from tests.masked_models_tests._mixed_checks import (
    benchmark_operation,
    check_embeddings_score_gradient,
    check_gradient_finite_difference,
    check_sampling,
    check_upstream_pll,
)

_persistent_tool = make_persistent_fixture("glm2")
_CHECKPOINT = "tattabio/gLM2_150M"


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_embeddings_score_and_gradient_persistent_worker():
    check_embeddings_score_gradient("glm2", _CHECKPOINT)


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("sampling_method", ["single_pass", "iterative_refinement"])
def test_glm2_sampling_preserves_modalities_and_advances_rng(sampling_method):
    check_sampling("glm2", _CHECKPOINT, sampling_method)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_gradient_matches_finite_difference():
    check_gradient_finite_difference("glm2", _CHECKPOINT)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_score_matches_native_upstream():
    check_upstream_pll("glm2", _CHECKPOINT)


@pytest.mark.benchmark("glm2-embedding")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_embeddings_benchmark(request):
    benchmark_operation(request, "glm2", _CHECKPOINT, "embeddings")


@pytest.mark.benchmark("glm2-score")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_score_benchmark(request):
    benchmark_operation(request, "glm2", _CHECKPOINT, "score")


@pytest.mark.benchmark("glm2-sample")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_sample_benchmark(request):
    benchmark_operation(request, "glm2", _CHECKPOINT, "sample")


@pytest.mark.benchmark("glm2-gradient")
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_glm2_gradient_benchmark(request):
    benchmark_operation(request, "glm2", _CHECKPOINT, "gradient")
