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

from typing import Any, get_args

import pytest

from proto_tools.utils.tool_io import Directionality, Metrics

# Recognized spec value types (must match the ``type`` strings in MetricSpec).
_VALID_TYPE_STRINGS = {
    "float",
    "int",
    "bool",
    "list[float]",
    "list[float|None]",
    "list[list[float]]",
}

# Recognized optimization directions, sourced from the ``Directionality`` literal.
_VALID_BETTER_VALUES = set(get_args(Directionality))


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

    # better_values_are: recognized direction when present
    if "better_values_are" in spec:
        direction = spec["better_values_are"]
        assert direction in _VALID_BETTER_VALUES, (
            f"{subclass_name}.metric_spec[{metric_name!r}].better_values_are is {direction!r}; "
            f"must be one of {sorted(_VALID_BETTER_VALUES)}"
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


@pytest.mark.parametrize("metrics_subclass", _METRIC_SUBCLASSES, ids=_IDS)
def test_repo_metrics_declare_direction(metrics_subclass: type[Metrics]) -> None:
    """Every in-repo tool metric declares a valid ``better_values_are`` direction.

    Scoped to ``proto_tools.tools`` subclasses so test-only fixtures are exempt.
    Directionality is metadata consumers rely on, so it must be a deliberate
    per-metric choice rather than silently omitted.
    """
    if not metrics_subclass.__module__.startswith("proto_tools.tools"):
        pytest.skip(f"{metrics_subclass.__qualname__} is not a proto_tools.tools metric")
    for metric_name, spec in metrics_subclass.metric_spec.items():
        direction = spec.get("better_values_are")
        assert direction in _VALID_BETTER_VALUES, (
            f"{metrics_subclass.__qualname__}.metric_spec[{metric_name!r}] is missing a valid "
            f"better_values_are (got {direction!r}); must be one of {sorted(_VALID_BETTER_VALUES)}"
        )


@pytest.mark.parametrize("direction", sorted(_VALID_BETTER_VALUES))
def test_check_spec_entry_accepts_valid_direction(direction: str) -> None:
    """Every recognized ``better_values_are`` value passes validation."""
    _check_spec_entry("m", {"type": "float", "better_values_are": direction}, "T")


def test_check_spec_entry_rejects_unknown_direction() -> None:
    """An unrecognized ``better_values_are`` value raises."""
    with pytest.raises(AssertionError, match="better_values_are"):
        _check_spec_entry("m", {"type": "float", "better_values_are": "up"}, "T")


def test_at_least_one_metrics_subclass_registered() -> None:
    """Sanity: the refactor added real ``Metrics`` subclasses across the repo."""
    assert len(_METRIC_SUBCLASSES) >= 10, (
        f"expected at least 10 Metrics subclasses across the repo, found {len(_METRIC_SUBCLASSES)}: {_IDS}"
    )
