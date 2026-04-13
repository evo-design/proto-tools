"""Unit tests for the ``Metrics`` container in ``proto_tools/utils/tool_io.py``.

Covers dual attribute/mapping access, ``None``-stripping at construction,
``primary_value``, ``update``, and the two halves of ``validate_against_spec``
(presence of always-available metrics, and element-wise ``min``/``max`` bounds
including list-valued and gap-tolerant per-position metrics).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from proto_tools.utils.tool_io import Metrics, MetricSpec


class _SampleMetrics(Metrics):
    """Subclass with mixed scalar and per-position metric specs."""

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "perplexity": {"availability": "always", "type": "float", "min": 1.0, "max": None},
        "log_likelihood": {"availability": "always", "type": "float", "min": None, "max": 0.0},
        "optional_score": {"availability": "depends on input", "type": "float", "min": 0.0, "max": 1.0},
        "per_position": {"availability": "always", "type": "list[float|None]", "min": -10.0, "max": 0.0},
    }
    primary_metric: str | None = "perplexity"


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
    """``None``-valued extras should be absent, not stored as ``None`` sentinels."""
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
    """``primary_value`` returns ``None`` if the named metric isn't present."""

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
    m.validate_against_spec()  # should not raise


def test_validate_raises_when_always_metric_missing():
    """Omitting a metric declared ``availability='always'`` should fail."""
    m = _SampleMetrics(perplexity=2.5, per_position=[-1.0])  # missing log_likelihood
    with pytest.raises(AssertionError, match=r"log_likelihood.*always-available"):
        m.validate_against_spec()


def test_validate_skips_optional_availability_when_absent():
    """Optional metrics (non-always availability) may be absent without failure."""
    m = _SampleMetrics(perplexity=2.5, log_likelihood=-3.0, per_position=[-1.0])
    m.validate_against_spec()  # ``optional_score`` absent, OK


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
    """Metric keys not in ``metric_spec`` shouldn't trip the bounds check."""
    m = _SampleMetrics(
        perplexity=2.0,
        log_likelihood=-1.0,
        per_position=[-1.0],
        undeclared=999.0,  # no spec entry → skipped
    )
    m.validate_against_spec()


def test_validate_skips_bool_values():
    """Booleans have no min/max semantics and should be skipped."""

    class _BoolMetrics(Metrics):
        metric_spec: ClassVar[dict[str, MetricSpec]] = {
            "flag": {"availability": "always", "type": "bool", "min": None, "max": None},
        }

    _BoolMetrics(flag=True).validate_against_spec()
    _BoolMetrics(flag=False).validate_against_spec()


def test_validate_list_metric_element_wise():
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[-1.0, -100.0])
    with pytest.raises(AssertionError, match=r"per_position.*index 1.*below declared min"):
        m.validate_against_spec()


def test_validate_list_metric_tolerates_none_gaps():
    """Per-position lists may contain ``None`` for gap positions; these must be skipped."""
    m = _SampleMetrics(perplexity=2.0, log_likelihood=-1.0, per_position=[None, -1.0, None, -2.0])
    m.validate_against_spec()


def test_validate_nested_list_metric_element_wise():
    """``list[list[float]]`` metrics validate recursively."""

    class _MatrixMetrics(Metrics):
        metric_spec: ClassVar[dict[str, MetricSpec]] = {
            "matrix": {"availability": "always", "type": "list[list[float]]", "min": 0.0, "max": 1.0},
        }

    _MatrixMetrics(matrix=[[0.1, 0.5], [0.2, 0.9]]).validate_against_spec()
    with pytest.raises(AssertionError, match="matrix"):
        _MatrixMetrics(matrix=[[0.1, 0.5], [0.2, 1.5]]).validate_against_spec()
