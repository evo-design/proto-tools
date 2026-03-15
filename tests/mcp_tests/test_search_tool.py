"""Tests for MCP search tool."""

from __future__ import annotations

from bio_tools_mcp.tools import search_tools


def test_search_by_tool_key():
    result = search_tools("blast")
    assert len(result) > 0
    assert any("blast" in t["key"] for t in result)


def test_search_by_category():
    result = search_tools("structure_prediction")
    assert len(result) > 0
    assert any(t["category"] == "structure_prediction" for t in result)


def test_search_by_description_keyword():
    result = search_tools("protein")
    assert len(result) > 0


def test_search_no_results():
    result = search_tools("xyznonexistenttoolxyz")
    assert result == []


def test_search_max_results():
    result = search_tools("protein", max_results=2)
    assert len(result) <= 2


def test_search_results_have_required_fields():
    result = search_tools("blast")
    required = {"key", "label", "category", "description", "uses_gpu"}
    for t in result:
        assert required.issubset(t.keys())


def test_search_ranking_prefers_key_match():
    result = search_tools("blast")
    if len(result) >= 2:
        # First result should have "blast" in its key
        assert "blast" in result[0]["key"]
