"""Tests for MCP resources."""

from __future__ import annotations

import json

import pytest

from bio_tools_mcp.resources import tool_citation, tool_doc, tool_example, tool_schemas


# ── tool_doc ────────────────────────────────────────────────────────────────


def test_tool_doc_returns_markdown(sample_tool_key):
    result = tool_doc(sample_tool_key)
    assert isinstance(result, str)
    assert result.startswith("#")
    assert "## Description" in result


def test_tool_doc_contains_source_file(sample_tool_key):
    result = tool_doc(sample_tool_key)
    assert "**Source file:**" in result


def test_tool_doc_contains_schemas(sample_tool_key):
    result = tool_doc(sample_tool_key)
    assert "## Inputs Schema" in result or "## Config Schema" in result


def test_tool_doc_invalid_key():
    with pytest.raises(ValueError, match="Unknown tool"):
        tool_doc("nonexistent-tool-xyz")


# ── tool_schemas ────────────────────────────────────────────────────────────


def test_tool_schemas_returns_valid_json(sample_tool_key):
    result = tool_schemas(sample_tool_key)
    parsed = json.loads(result)
    assert "inputs" in parsed
    assert "config" in parsed
    assert "output" in parsed


def test_tool_schemas_invalid_key():
    with pytest.raises(ValueError, match="Unknown tool"):
        tool_schemas("nonexistent-tool-xyz")


# ── tool_citation ───────────────────────────────────────────────────────────


def test_tool_citation_returns_bibtex(tool_registry):
    # Find a tool that has a citation
    for spec in tool_registry.list_all():
        citation = tool_registry.get_citation(spec.key)
        if citation is not None:
            result = tool_citation(spec.key)
            assert "@" in result
            return
    pytest.skip("No tools with citations found")


def test_tool_citation_missing_raises():
    # Try to find a tool without a citation, or use a fake key
    with pytest.raises(ValueError):
        tool_citation("nonexistent-tool-xyz")


# ── tool_example ───────────────────────────────────────────────────────────


def test_tool_example_returns_markdown(tool_registry):
    # Find a tool that has an example notebook
    for spec in tool_registry.list_all():
        notebook_path = spec.source_file.parent / "examples" / "example.ipynb"
        if notebook_path.is_file():
            result = tool_example(spec.key)
            assert isinstance(result, str)
            assert result.startswith("# Example:")
            assert "**Notebook path:**" in result
            return
    pytest.skip("No tools with example notebooks found")


def test_tool_example_contains_code(tool_registry):
    for spec in tool_registry.list_all():
        notebook_path = spec.source_file.parent / "examples" / "example.ipynb"
        if notebook_path.is_file():
            result = tool_example(spec.key)
            assert "```python" in result, "Expected code cells in example"
            return
    pytest.skip("No tools with example notebooks found")


def test_tool_example_invalid_key():
    with pytest.raises(ValueError):
        tool_example("nonexistent-tool-xyz")
