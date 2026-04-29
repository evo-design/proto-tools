"""tests/style_consistency_tests/test_metric_spec_consistency.py.

Static consistency checks for ``Metrics`` subclasses across the repo.

Validates that every ``metric_spec`` declaration is well-formed — every entry
has a recognized ``type``, ``min``/``max`` are numeric or ``None``,
``availability`` is a non-empty string, etc. This is a structural check on
spec declarations themselves; value-vs-spec validation on actual tool outputs
lives in each tool's e2e test via ``assert_metrics_in_spec``.

Runs under ``--cpu-only`` in CI — no tool execution, no env builds.
"""

from __future__ import annotations

from typing import Any

import pytest

from proto_tools.utils.tool_io import Metrics

# Recognized spec value types (must match the ``type`` strings in MetricSpec).
_VALID_TYPE_STRINGS = {
    "float",
    "int",
    "bool",
    "list[float]",
    "list[float|None]",
    "list[list[float]]",
}


def _discover_metrics_subclasses() -> list[type[Metrics]]:
    """Walk ``Metrics.__subclasses__`` recursively (importing ``proto_tools`` first)."""
    import proto_tools  # noqa: F401  — triggers registration of every subclass via tool module imports

    seen: set[type[Metrics]] = set()
    stack: list[type[Metrics]] = [Metrics]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
    return sorted(seen, key=lambda c: c.__qualname__)


_METRIC_SUBCLASSES = _discover_metrics_subclasses()
_IDS = [c.__qualname__ for c in _METRIC_SUBCLASSES]


def _check_spec_entry(metric_name: str, spec: Any, subclass_name: str) -> None:
    """Validate one ``metric_spec`` entry; raise ``AssertionError`` on any issue."""
    assert isinstance(spec, dict), f"{subclass_name}.metric_spec[{metric_name!r}] must be a dict, got {type(spec)}"

    # type: recognized string
    ty = spec.get("type")
    assert ty in _VALID_TYPE_STRINGS, (
        f"{subclass_name}.metric_spec[{metric_name!r}] has unrecognized type {ty!r}; "
        f"must be one of {sorted(_VALID_TYPE_STRINGS)}"
    )

    # availability: non-empty string when present
    if "availability" in spec:
        avail = spec["availability"]
        assert isinstance(avail, str) and avail.strip(), (
            f"{subclass_name}.metric_spec[{metric_name!r}].availability must be a non-empty string"
        )

    # min / max: numeric or None
    for bound in ("min", "max"):
        if bound in spec:
            value = spec[bound]
            assert value is None or isinstance(value, (int, float)), (
                f"{subclass_name}.metric_spec[{metric_name!r}].{bound} must be numeric or None, got {value!r}"
            )

    # If both bounds present and numeric, min <= max
    lo, hi = spec.get("min"), spec.get("max")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        assert lo <= hi, f"{subclass_name}.metric_spec[{metric_name!r}] has min={lo} > max={hi}"

    # description / unit: string when present
    for key in ("description", "unit"):
        if key in spec:
            value = spec[key]
            assert value is None or isinstance(value, str), (
                f"{subclass_name}.metric_spec[{metric_name!r}].{key} must be a string or None, got {value!r}"
            )


@pytest.mark.parametrize("metrics_subclass", _METRIC_SUBCLASSES, ids=_IDS)
def test_metric_spec_is_well_formed(metrics_subclass: type[Metrics]) -> None:
    """Every ``Metrics`` subclass has a well-formed ``metric_spec`` ClassVar."""
    spec_map = metrics_subclass.metric_spec
    assert isinstance(spec_map, dict), (
        f"{metrics_subclass.__qualname__}.metric_spec must be a dict, got {type(spec_map)}"
    )
    for metric_name, spec in spec_map.items():
        assert isinstance(metric_name, str) and metric_name, (
            f"{metrics_subclass.__qualname__}.metric_spec has a non-string key: {metric_name!r}"
        )
        _check_spec_entry(metric_name, spec, metrics_subclass.__qualname__)


def test_at_least_one_metrics_subclass_registered() -> None:
    """Sanity: the refactor added real ``Metrics`` subclasses across the repo."""
    assert len(_METRIC_SUBCLASSES) >= 10, (
        f"expected at least 10 Metrics subclasses across the repo, found {len(_METRIC_SUBCLASSES)}: {_IDS}"
    )
