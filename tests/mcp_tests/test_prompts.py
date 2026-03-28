"""tests/mcp_tests/test_prompts.py

Tests for MCP prompts."""

from __future__ import annotations

from bio_tools_mcp.prompts import find_tool, tool_walkthrough


def test_find_tool_returns_string():
    result = find_tool("predict protein structure")
    assert isinstance(result, str)
    assert "predict protein structure" in result


def test_find_tool_includes_workflow_steps():
    result = find_tool("score DNA")
    assert "list_categories" in result
    assert "search_tools" in result
    assert "get_tool_schema" in result


def test_tool_walkthrough_returns_string():
    result = tool_walkthrough("esmfold-prediction")
    assert isinstance(result, str)
    assert "esmfold-prediction" in result


def test_tool_walkthrough_includes_schema_steps():
    result = tool_walkthrough("blast-search")
    assert "get_tool_schema" in result
    assert "get_tool_example" in result
    assert "get_tool_citation" in result
