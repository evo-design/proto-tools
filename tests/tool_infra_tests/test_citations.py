"""tests/tool_infra_tests/test_citations.py

Tests for ToolRegistry citation retrieval."""

import pytest

from bio_programming_tools.tools.tool_registry import ToolRegistry


# ── get_citation ─────────────────────────────────────────────────────────────


def test_get_citation_returns_bibtex_string():
    """get_citation returns a BibTeX string for tools with cite.bib"""
    citation = ToolRegistry.get_citation("alphafold3-prediction")
    assert citation is not None
    assert "@article{" in citation or "@misc{" in citation
    assert "title=" in citation
    assert "year=" in citation


def test_get_citation_raises_for_unknown_tool():
    """get_citation raises ValueError for unknown tool keys"""
    with pytest.raises(ValueError, match="Unknown tool"):
        ToolRegistry.get_citation("nonexistent-tool-key")


def test_get_citation_contains_doi():
    """Citations should contain DOI when available"""
    citation = ToolRegistry.get_citation("blast-search")
    assert "doi=" in citation


def test_get_citation_multiple_tools_same_directory():
    """Multiple tool keys from same directory return same citation"""
    citation1 = ToolRegistry.get_citation("pyhmmer-hmmscan")
    citation2 = ToolRegistry.get_citation("pyhmmer-hmmsearch")
    citation3 = ToolRegistry.get_citation("pyhmmer-phmmer")
    assert citation1 is not None
    assert citation1 == citation2 == citation3
    assert "pyhmmer" in citation1.lower() or "pyHMMER" in citation1


# ── list_citations ───────────────────────────────────────────────────────────


def test_list_citations_values_are_bibtex():
    """All values in list_citations are valid BibTeX strings"""
    citations = ToolRegistry.list_citations()
    for key, bibtex in citations.items():
        assert isinstance(bibtex, str)
        assert "@" in bibtex, f"Citation for {key} missing @ symbol"
        assert "title" in bibtex, f"Citation for {key} missing title"


def test_list_citations_keys_match_registry():
    """All keys in list_citations exist in the tool registry"""
    citations = ToolRegistry.list_citations()
    for key in citations:
        spec = ToolRegistry.get(key)
        assert spec.key == key
