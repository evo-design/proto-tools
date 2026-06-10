"""tests/style_consistency_tests/test_tool_io_round_trip.py.

Universal contract: every registered tool's Input and Config models must
reconstruct cleanly from their own ``model_dump`` output.

This is the property the hosted execution service relies on — it serializes an
Input or Config to a dict, sends it over the wire, and reconstructs it on the
other side via ``Model(**dumped_dict)``. Pydantic's default field coercion handles this for
free, but a tool with a ``@field_validator(..., mode="before")`` that takes
manual control over normalization can accidentally drop the dict shape.

If a tool fails one of these tests, the fix is to add a ``dict`` branch to
its custom normalizer so it can rebuild the nested BaseModel from the dict
the same way Pydantic would have.
"""

from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

import pytest
from pydantic import BaseModel, SerializeAsAny, ValidationError

from proto_tools.tools.tool_registry import ToolRegistry

_TOOL_KEYS = sorted(spec.key for spec in ToolRegistry.list_all())


@pytest.mark.parametrize("tool_key", _TOOL_KEYS)
def test_tool_input_round_trips_through_dict(tool_key: str) -> None:
    """``Input(**Input.model_dump()) == Input`` for every registered tool's example_input."""
    spec = ToolRegistry.get(tool_key)
    assert spec is not None, f"Registry returned None for tool_key={tool_key!r}"

    example = ToolRegistry.get_example_input(tool_key)
    assert example is not None, (
        f"Tool {tool_key!r} has no example_input registered. "
        "Every tool needs one — see proto_tools/tools/tool_registry.py."
    )

    dumped = example.model_dump(exclude_none=True)
    reconstructed = spec.input_model(**dumped)
    assert reconstructed.model_dump(exclude_none=True) == dumped, (
        f"Round-trip mismatch for {tool_key!r} input. "
        "Likely cause: a @field_validator(mode='before') on a nested BaseModel field "
        "doesn't accept the dict shape Pydantic produces from model_dump."
    )


@pytest.mark.parametrize("tool_key", _TOOL_KEYS)
def test_tool_config_round_trips_through_dict(tool_key: str) -> None:
    """``Config(**Config.model_dump()) == Config`` for every registered tool's default config."""
    spec = ToolRegistry.get(tool_key)
    assert spec is not None, f"Registry returned None for tool_key={tool_key!r}"

    try:
        default_cfg = spec.config_model()
    except ValidationError as exc:
        pytest.skip(f"{tool_key!r} default config does not construct on this platform: {exc}")
    dumped = default_cfg.model_dump(exclude_none=True)
    reconstructed = spec.config_model(**dumped)
    assert reconstructed.model_dump(exclude_none=True) == dumped, (
        f"Round-trip mismatch for {tool_key!r} config. "
        "Likely cause: a @field_validator(mode='before') on a nested BaseModel field "
        "doesn't accept the dict shape Pydantic produces from model_dump."
    )


# ── Polymorphic output safety ─────────────────────────────────────────────────


def _all_subclasses(cls: type) -> set[type]:
    """Return every transitive subclass of ``cls``."""
    seen: set[type] = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        stack.extend(sub.__subclasses__())
    return seen


def _serialize_as_any_inner(annotation: Any) -> type[BaseModel] | None:
    """Return the inner BaseModel type if ``annotation`` contains ``SerializeAsAny[T]``."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if any(isinstance(meta, SerializeAsAny) for meta in args[1:]):
            inner = args[0]
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                return inner
        # Recurse into the bare type carried by Annotated[T, ...].
        return _serialize_as_any_inner(args[0])
    for arg in get_args(annotation):
        found = _serialize_as_any_inner(arg)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("tool_key", _TOOL_KEYS)
def test_tool_output_does_not_drop_subclass_fields_on_validate(tool_key: str) -> None:
    """Output ``SerializeAsAny[T]`` fields must be narrowed when T has subclasses with extras."""
    spec = ToolRegistry.get(tool_key)
    assert spec is not None, f"Registry returned None for tool_key={tool_key!r}"

    for field_name, field_info in spec.output_model.model_fields.items():
        inner = _serialize_as_any_inner(field_info.annotation)
        if inner is None:
            continue
        risky = sorted(
            (s.__name__ for s in _all_subclasses(inner) if set(s.model_fields) - set(inner.model_fields)),
        )
        assert not risky, (
            f"{tool_key!r}: {spec.output_model.__name__}.{field_name} is "
            f"SerializeAsAny[{inner.__name__}], but these subclasses declare extra fields "
            f"that model_validate will drop on the inbound JSON path: {risky}. "
            f"Narrow {field_name} on the concrete output class to the specific subclass type."
        )
