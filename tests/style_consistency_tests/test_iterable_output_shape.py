"""tests/style_consistency_tests/test_iterable_output_shape.py.

Tests that an iterable tool returns exactly one self-contained object per input item.

Everything a tool produces per item must live inside the element of its
``iterable_output_field``, never in a second top-level list running parallel to it. The
framework splits a batch across workers and serves items from cache individually, and it
reassembles only that one field. A parallel per-item list therefore comes back holding one
worker's slice, or nothing at all, with no error raised.
"""

import types
import typing

import pytest
from pydantic import BaseModel

from proto_tools.tools.tool_registry import ToolRegistry

_ITERABLE_SPECS = [spec for spec in ToolRegistry.list_all() if spec.iterable_output_field]


def _element_models(spec) -> list[type] | None:
    """The per-item model(s) of a tool's iterable output field, or None if it is a bare value.

    A tool may return one model per item, or a union of them when the item shape depends on
    the task (a classification result versus a regression result). Both give per-item data a
    home, so both count. Only a union is unpacked into members: the arguments of any other
    generic, such as the inner ``list`` of a ``list[list[Orf]]``, are not alternative item
    shapes and must not be mistaken for one.
    """
    annotation = spec.output_model.model_fields[spec.iterable_output_field].annotation
    args = typing.get_args(annotation)
    element = args[0] if args else None
    is_union = typing.get_origin(element) in (types.UnionType, typing.Union)
    members = typing.get_args(element) if is_union else (element,)
    if members and all(isinstance(m, type) and issubclass(m, BaseModel) for m in members):
        return list(members)
    return None


def test_iterable_tools_found():
    """The scan finds iterable tools, so a passing suite is not an empty scan."""
    assert len(_ITERABLE_SPECS) > 1


@pytest.mark.parametrize("spec", _ITERABLE_SPECS, ids=lambda s: s.key)
def test_iterable_output_element_is_a_model(spec):
    """A new iterable tool returns one object per item, so per-item fields have a home."""
    assert _element_models(spec), (
        f"{spec.key}: {spec.iterable_output_field!r} should be a list of per-item objects (or of a "
        f"union of them). Anything the tool produces per item then has a place to live."
    )
