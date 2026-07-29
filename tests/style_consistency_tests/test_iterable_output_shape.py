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
from proto_tools.utils.tool_io import BaseToolOutput

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
def test_per_item_data_lives_in_the_item(spec):
    """No output field runs parallel to the per-item list; per-item data belongs in the item."""
    extra = [
        name
        for name in spec.output_model.model_fields
        if name not in BaseToolOutput.model_fields and name != spec.iterable_output_field
    ]
    assert not extra or _element_models(spec), (
        f"{spec.key}: {spec.output_model.__name__} has {extra} alongside the per-item list "
        f"{spec.iterable_output_field!r}, whose element is a bare value. Per-item values must "
        f"live inside that element, since only {spec.iterable_output_field!r} is split and "
        f"reassembled when a batch is fanned out or served from cache."
    )


@pytest.mark.parametrize("spec", _ITERABLE_SPECS, ids=lambda s: s.key)
def test_iterable_output_element_is_a_model(spec):
    """A new iterable tool returns one object per item, so per-item fields have a home."""
    assert _element_models(spec), (
        f"{spec.key}: {spec.iterable_output_field!r} should be a list of per-item objects (or of a "
        f"union of them). Anything the tool produces per item then has a place to live."
    )


# ── The element-model helper ────────────────────────────────────────────────


class _MemberA(BaseModel):
    """A per-item model."""

    value: str = ""


class _MemberB(BaseModel):
    """A different per-item model."""

    count: int = 0


def _spec_returning(annotation):
    """Build a stand-in spec whose iterable output field has the given annotation."""
    field = type("_Field", (), {"annotation": annotation})
    model = type("_Model", (), {"model_fields": {"results": field}})
    return type("_Spec", (), {"iterable_output_field": "results", "output_model": model})


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (list[_MemberA], ["_MemberA"]),
        (list[_MemberA | _MemberB], ["_MemberA", "_MemberB"]),
        (list[list[_MemberA]], None),
        (list[str], None),
        (list[_MemberA | str], None),
    ],
    ids=["model", "union_of_models", "nested_list", "bare", "union_with_non_model"],
)
def test_element_models_sees_through_unions_only(annotation, expected):
    """A union of per-item models counts; other generics are not alternative item shapes."""
    found = _element_models(_spec_returning(annotation))

    assert (found is not None) == (expected is not None)
    if expected is not None:
        assert [m.__name__ for m in found] == expected
