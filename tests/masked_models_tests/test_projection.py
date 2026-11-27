"""Tests for the UMAP projection helper used by embedding tools."""

import numpy as np
import pytest

from proto_tools.tools.masked_models.projection import attach_projections, compute_projections
from proto_tools.tools.masked_models.shared_data_models import Projection2D, SequenceEmbedding


def _random_embeddings(n: int, dim: int, seed: int = 0) -> list[list[float]]:
    return np.random.default_rng(seed).normal(size=(n, dim)).tolist()


def _make_results(n: int, dim: int = 16, *, stale: Projection2D | None = None) -> list[SequenceEmbedding]:
    results = [SequenceEmbedding(mean_embedding=e, attention_mask=[1] * len(e)) for e in _random_embeddings(n, dim)]
    if stale is not None:
        for r in results:
            r.projection = stale
    return results


@pytest.mark.parametrize("n", [1, 3])
def test_compute_projections_returns_none_below_threshold(n):
    assert compute_projections(_random_embeddings(n, 8)) is None


# n=4 is the threshold; n=5 exercises the n_neighbors auto-clamp (default 15 > 4).
@pytest.mark.parametrize("n", [4, 5, 25])
def test_compute_projections_returns_one_projection_per_embedding(n):
    result = compute_projections(_random_embeddings(n, 16))
    assert result is not None
    assert len(result) == n
    assert all(isinstance(p, Projection2D) for p in result)


def test_compute_projections_deterministic_with_seed():
    embs = _random_embeddings(12, 32)
    assert compute_projections(embs) == compute_projections(embs)


@pytest.mark.parametrize("stale", [None, Projection2D(x=999.0, y=-999.0)])
def test_attach_projections_populates_above_threshold(stale):
    results = _make_results(6, stale=stale)
    attach_projections(results)
    assert all(r.projection is not None and r.projection != stale for r in results)


@pytest.mark.parametrize("stale", [None, Projection2D(x=1.0, y=2.0)])
def test_attach_projections_clears_below_threshold(stale):
    # Cache returns object references — stale projections from a prior multi-point
    # fit must be cleared, not preserved, when the current batch is sub-threshold.
    results = _make_results(2, stale=stale)
    attach_projections(results)
    assert all(r.projection is None for r in results)


def test_attach_projections_consistent_across_calls_on_same_batch():
    a, b = _make_results(10), _make_results(10)
    attach_projections(a)
    attach_projections(b)
    assert [(r.projection.x, r.projection.y) for r in a] == [(r.projection.x, r.projection.y) for r in b]
