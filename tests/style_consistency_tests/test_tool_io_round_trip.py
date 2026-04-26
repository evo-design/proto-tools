"""tests/style_consistency_tests/test_tool_io_round_trip.py.

Universal contract: every registered tool's Input and Config models must
reconstruct cleanly from their own ``model_dump`` output.

This is the property that ``proto_tools.cloud._route_to_cloud`` and every
proto-tools-api Modal service rely on — they serialize an Input or Config to
a dict, send it over the wire, and reconstruct it on the other side via
``Model(**dumped_dict)``. Pydantic's default field coercion handles this for
free, but a tool with a ``@field_validator(..., mode="before")`` that takes
manual control over normalization can accidentally drop the dict shape.

If a tool fails one of these tests, the fix is to add a ``dict`` branch to
its custom normalizer so it can rebuild the nested BaseModel from the dict
the same way Pydantic would have.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
