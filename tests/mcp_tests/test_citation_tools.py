"""tests/mcp_tests/test_citation_tools.py

Tests for MCP citation tools."""

from __future__ import annotations

import pytest

from bio_tools_mcp.tools import get_tool_citation, list_citations

# ── get_tool_citation ───────────────────────────────────────────────────────


def test_get_tool_citation_returns_string_or_none(all_tool_keys):
    for key in all_tool_keys:
        result = get_tool_citation(key)
        assert result is None or isinstance(result, str), (
            f"Expected str or None for {key}, got {type(result)}"
        )


def test_get_tool_citation_bibtex_format(all_tool_keys):
    for key in all_tool_keys:
        result = get_tool_citation(key)
        if result is not None:
            assert "@" in result, f"Citation for {key} doesn't look like BibTeX"


def test_get_tool_citation_invalid_key():
    with pytest.raises(ValueError, match="Unknown tool"):
        get_tool_citation("nonexistent-tool-xyz")


# ── list_citations ──────────────────────────────────────────────────────────


def test_list_citations_returns_dict():
    result = list_citations()
    assert isinstance(result, dict)
    assert len(result) > 0, "Expected at least one citation"


def test_list_citations_all_bibtex():
    result = list_citations()
    for key, bibtex in result.items():
        assert isinstance(bibtex, str)
        assert "@" in bibtex, f"Citation for {key} doesn't look like BibTeX"
