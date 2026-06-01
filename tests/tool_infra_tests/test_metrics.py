"""Unit tests for the ``Metrics`` container in ``proto_tools/utils/tool_io.py``.

Covers dual attribute/mapping access, ``None``-stripping at construction,
``primary_value``, ``update``, and ``validate_against_spec`` (presence, type
enforcement, and element-wise bounds).
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers.scoring import (
    log_likelihood_metrics,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec


class _SampleMetrics(Metrics):
    """Subclass with mixed scalar and per-position metric specs."""

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "perplexity": {
            "availability": "always",
            "type": "float",
            "min": 1.0,
            "max": None,
            "better_values_are": "lower",
        },
        "log_likelihood": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": 0.0,
            "better_values_are": "higher",
        },
        "optional_score": {
            "availability": "depends on input",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "per_position": {
            "availability": "always",
            "type": "list[float|None]",
            "min": -10.0,
            "max": 0.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = "perplexity"


class _IntMetrics(Metrics):
    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "count": {
            "availability": "always",
            "type": "int",
            "min": 0,
            "max": None,
            "better_values_are": "context-dependent",
        },
    }


class _BoolMetrics(Metrics):
    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "flag": {
            "availability": "always",
            "type": "bool",
            "min": None,
            "max": None,
            "better_values_are": "context-dependent",
        },
    }


class _ListMetrics(Metrics):
    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "scores": {
            "availability": "always",
            "type": "list[float]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }


class _MatrixMetrics(Metrics):
    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "matrix": {
            "availability": "always",
            "type": "list[list[float]]",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "context-dependent",
        },
    }


# ── Dual access ──────────────────────────────────────────────────────────────


def test_attribute_and_mapping_access_agree():
    m = _SampleMetrics(perplexity=2.5, log_likelihood=-3.0, per_position=[-1.0, -2.0])
    assert m.perplexity == 2.5
    assert m["perplexity"] == 2.5
    assert "perplexity" in m
    assert set(m.keys()) == {"perplexity", "log_likelihood", "per_position"}
    assert dict(m.items())["log_likelihood"] == -3.0


def test_setitem_round_trips():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0])
    m["new_metric"] = 42.0
    assert m["new_metric"] == 42.0
    assert m.new_metric == 42.0


def test_get_returns_default_for_missing():
    m = _SampleMetrics(perplexity=1.5, log_likelihood=-1.0, per_position=[-1.0])
    assert m.get("optional_score") is None
    assert m.get("optional_score", 0.5) == 0.5


def test_none_values_stripped_at_construction():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0], optional_score=None)
    assert "optional_score" not in m


def test_iter_and_len_walk_extras_only():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0])
    assert len(m) == 3
    assert sorted(iter(m)) == ["log_likelihood", "per_position", "perplexity"]


def test_update_merges_extras():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0])
    other = _SampleMetrics(perplexity=3.0, log_likelihood=-2.0, per_position=[-2.0], optional_score=0.5)
    m.update(other)
    assert m["perplexity"] == 3.0
    assert m["optional_score"] == 0.5


def test_update_accepts_plain_mapping():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0])
    m.update({"perplexity": 4.0, "optional_score": 0.25})
    assert m["perplexity"] == 4.0
    assert m["optional_score"] == 0.25


def test_primary_value_resolves_to_named_metric():
    m = _SampleMetrics(perplexity=2.5, log_likelihood=-1.0, per_position=[-1.0])
    assert m.primary_value == 2.5


def test_primary_value_none_when_metric_missing():
    class _NoPrimary(Metrics):
        metric_spec: ClassVar[dict[str, MetricSpec]] = {}
        primary_metric: str | None = "missing_key"

    assert _NoPrimary().primary_value is None


def test_getitem_raises_keyerror_for_missing():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0])
    with pytest.raises(KeyError):
        _ = m["does_not_exist"]


# ── validate_against_spec: presence ──────────────────────────────────────────


def test_validate_passes_when_all_always_metrics_present():
    m = _SampleMetrics(perplexity=2.5, log_likelihood=-3.0, per_position=[-1.0, -2.0])
    m.validate_against_spec()


def test_validate_raises_when_always_metric_missing():
    m = _SampleMetrics(perplexity=2.5, per_position=[-1.0])
    with pytest.raises(AssertionError, match=r"log_likelihood.*always-available"):
        m.validate_against_spec()


# ── validate_against_spec: bounds ────────────────────────────────────────────


def test_validate_raises_below_min():
    m = _SampleMetrics(perplexity=0.5, log_likelihood=-1.0, per_position=[-1.0])
    with pytest.raises(AssertionError, match=r"perplexity.*below declared min 1\.0"):
        m.validate_against_spec()


def test_validate_raises_above_max():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=5.0, per_position=[-1.0])
    with pytest.raises(AssertionError, match=r"log_likelihood.*above declared max 0\.0"):
        m.validate_against_spec()


def test_validate_skips_undeclared_metrics():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0], undeclared=999.0)
    m.validate_against_spec()


def test_validate_list_metric_element_wise():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0, -100.0])
    with pytest.raises(AssertionError, match=r"per_position.*index 1.*below declared min"):
        m.validate_against_spec()


def test_validate_list_metric_tolerates_none_gaps():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[None, -1.0, None, -2.0])
    m.validate_against_spec()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validate_rejects_non_finite_scalar(bad: float):
    m = _SampleMetrics(perplexity=bad, log_likelihood=-1.0, per_position=[-1.0])
    with pytest.raises(AssertionError, match=r"perplexity.*not finite"):
        m.validate_against_spec()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validate_rejects_non_finite_list_element(bad: float):
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0, bad])
    with pytest.raises(AssertionError, match=r"per_position.*index 1.*not finite"):
        m.validate_against_spec()


def test_validate_nested_list_bounds():
    _MatrixMetrics(matrix=[[0.1, 0.5], [0.2, 0.9]]).validate_against_spec()
    with pytest.raises(AssertionError, match="matrix"):
        _MatrixMetrics(matrix=[[0.1, 0.5], [0.2, 1.5]]).validate_against_spec()


# ── validate_against_spec: types ────────────────────────────────────────────


def test_validate_type_scalar():
    """Type enforcement for float, int, and bool — including the bool-subclasses-int trap."""
    # int accepted for "float" spec
    _SampleMetrics(perplexity=2.5, log_likelihood=-3, per_position=[-1.0]).validate_against_spec()
    # wrong types rejected for each scalar spec
    with pytest.raises(AssertionError, match=r"'float'"):
        _SampleMetrics(perplexity="bad", log_likelihood=-1.0, per_position=[-1.0]).validate_against_spec()
    with pytest.raises(AssertionError, match=r"bool.*'float'"):
        _SampleMetrics(perplexity=True, log_likelihood=-1.0, per_position=[-1.0]).validate_against_spec()
    with pytest.raises(AssertionError, match=r"float.*'int'"):
        _IntMetrics(count=3.14).validate_against_spec()
    with pytest.raises(AssertionError, match=r"bool.*'int'"):
        _IntMetrics(count=True).validate_against_spec()
    _BoolMetrics(flag=True).validate_against_spec()
    with pytest.raises(AssertionError, match=r"int.*'bool'"):
        _BoolMetrics(flag=1).validate_against_spec()


def test_validate_type_list():
    """Type enforcement for list[float], list[float|None], and list[list[float]]."""
    with pytest.raises(AssertionError, match=r"None.*'list\[float\]'"):
        _ListMetrics(scores=[0.5, None, 0.9]).validate_against_spec()
    with pytest.raises(AssertionError, match=r"'list\[float\]'"):
        _ListMetrics(scores=0.5).validate_against_spec()
    # list[float|None] accepts None but rejects non-numeric types
    with pytest.raises(AssertionError, match=r"str.*'list\[float\|None\]'"):
        _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0, "bad"]).validate_against_spec()
    with pytest.raises(AssertionError, match=r"matrix\[1\].*str"):
        _MatrixMetrics(matrix=[[0.1], ["bad"]]).validate_against_spec()
    with pytest.raises(AssertionError, match=r"expects inner list"):
        _MatrixMetrics(matrix=[0.5, 0.6]).validate_against_spec()


def test_validate_type_legacy_fallback():
    """Specs without a ``type`` field fall back to bounds-only checking."""

    class _LegacyMetrics(Metrics):
        metric_spec: ClassVar[dict[str, MetricSpec]] = {
            "score": {"availability": "always", "min": 0.0, "max": 1.0},
        }

    _LegacyMetrics(score=0.5).validate_against_spec()
    with pytest.raises(AssertionError, match=r"score.*above declared max"):
        _LegacyMetrics(score=1.5).validate_against_spec()


# ============================================================================
# log_likelihood_metrics helper
# ============================================================================
def test_log_likelihood_metrics_returns_canonical_triple():
    """Helper derives the full ``{ll, avg_ll, perplexity}`` triple from ``(avg, seq_len)``."""
    m = log_likelihood_metrics(avg_log_likelihood=-2.0, seq_len=100)
    assert m == {
        "log_likelihood": pytest.approx(-200.0),
        "avg_log_likelihood": pytest.approx(-2.0),
        "perplexity": pytest.approx(math.exp(2.0)),
    }


def test_log_likelihood_metrics_satisfies_scoring_spec():
    """The triple validates against the three shared scoring-metrics specs."""
    from proto_tools.tools.causal_models.shared_data_models import CausalModelScoringMetrics
    from proto_tools.tools.inverse_folding.shared_data_models import InverseFoldingScoringMetrics
    from proto_tools.tools.masked_models.shared_data_models import MaskedModelScoringMetrics

    triple = log_likelihood_metrics(avg_log_likelihood=-1.5, seq_len=50)
    for cls in (CausalModelScoringMetrics, MaskedModelScoringMetrics, InverseFoldingScoringMetrics):
        cls(**triple).validate_against_spec()


def test_log_likelihood_metrics_seq_len_one_collapses_sum_and_mean():
    """At ``seq_len=1`` the sum and mean coincide."""
    m = log_likelihood_metrics(avg_log_likelihood=-0.7, seq_len=1)
    assert m["log_likelihood"] == pytest.approx(m["avg_log_likelihood"])


def test_log_likelihood_metrics_perplexity_one_at_zero_likelihood():
    """``avg_log_likelihood == 0`` corresponds to perfect prediction (perplexity = 1)."""
    m = log_likelihood_metrics(avg_log_likelihood=0.0, seq_len=10)
    assert m["log_likelihood"] == 0.0
    assert m["perplexity"] == pytest.approx(1.0)
