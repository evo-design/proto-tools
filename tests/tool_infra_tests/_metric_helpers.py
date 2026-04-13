"""tests/tool_infra_tests/_metric_helpers.py.

Shared helper for validating tool-emitted metrics against declared specs.

Used by e2e tool tests across every category. Import into an existing test and
add ``assert_metrics_in_spec(result)`` after the tool call to verify that every
``Metrics`` instance in the output satisfies its declared ``metric_spec``.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel

from proto_tools.utils.tool_io import Metrics


def _walk_metrics(obj: object) -> Iterator[Metrics]:
    """Yield every ``Metrics`` instance reachable from ``obj``.

    Recurses through ``BaseModel`` fields and ``list``/``tuple`` containers,
    so it finds metrics on direct-store outputs (``output.metrics``), iterable
    outputs (``output.scores[i]`` where each item IS a ``Metrics``), and
    structures-with-metrics (``output.structures[i].metrics``).
    """
    if isinstance(obj, Metrics):
        yield obj
        return
    if isinstance(obj, BaseModel):
        for field_name in type(obj).model_fields:
            try:
                value = getattr(obj, field_name)
            except AttributeError:
                continue
            yield from _walk_metrics(value)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_metrics(item)


def assert_metrics_in_spec(output: object) -> None:
    """Assert that every ``Metrics`` instance in ``output`` satisfies its spec.

    Walks ``output`` for any :class:`Metrics` instance (direct field, nested
    inside a list, or attached to a ``Structure``) and delegates to each
    container's :meth:`Metrics.validate_against_spec` method. The method
    raises ``AssertionError`` with a precise message on the first violation
    (metric name, value, and the bound it violated).

    Args:
        output: A tool output — typically a ``BaseToolOutput`` subclass, but
            any object whose ``model_fields`` / ``list`` / ``tuple`` structure
            contains ``Metrics`` instances will be walked.

    Raises:
        AssertionError: From ``Metrics.validate_against_spec``, naming the
            offending metric and bound.
    """
    for metrics in _walk_metrics(output):
        metrics.validate_against_spec()


__all__ = ["assert_metrics_in_spec"]
