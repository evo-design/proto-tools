"""Tests for MCP schema and example tools."""

from __future__ import annotations

import pytest

from bio_tools_mcp.tools import get_tool_example, get_tool_schema


# ── get_tool_schema ─────────────────────────────────────────────────────────


def test_get_tool_schema_returns_three_schemas(sample_tool_key):
    result = get_tool_schema(sample_tool_key)
    assert "inputs" in result
    assert "config" in result
    assert "output" in result


def test_get_tool_schema_has_properties(sample_tool_key):
    result = get_tool_schema(sample_tool_key)
    for schema_name in ("inputs", "config", "output"):
        schema = result[schema_name]
        assert "properties" in schema or "$defs" in schema, (
            f"{schema_name} schema for {sample_tool_key} missing properties"
        )


def test_get_tool_schema_all_tools(all_tool_keys):
    for key in all_tool_keys:
        result = get_tool_schema(key)
        assert isinstance(result, dict)
        assert len(result) == 3


def test_get_tool_schema_invalid_key():
    with pytest.raises(ValueError, match="Unknown tool"):
        get_tool_schema("nonexistent-tool-xyz")


# ── get_tool_example ────────────────────────────────────────────────────────


def test_get_tool_example_returns_dict_or_none(all_tool_keys):
    for key in all_tool_keys:
        result = get_tool_example(key)
        assert result is None or isinstance(result, dict), (
            f"Expected dict or None for {key}, got {type(result)}"
        )


def test_get_tool_example_invalid_key():
    with pytest.raises(ValueError, match="Unknown tool"):
        get_tool_example("nonexistent-tool-xyz")
