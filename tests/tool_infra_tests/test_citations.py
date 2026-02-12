"""
test_citations.py

Tests for ToolRegistry citation retrieval methods.
"""
from __future__ import annotations

import pytest

from bio_programming_tools.tools.tool_registry import ToolRegistry


class TestGetCitation:
    """Tests for ToolRegistry.get_citation()"""

    def test_get_citation_returns_bibtex_string(self):
        """get_citation returns a BibTeX string for tools with cite.bib"""
        # alphafold3-prediction has a cite.bib file
        citation = ToolRegistry.get_citation("alphafold3-prediction")
        assert citation is not None
        assert "@article{" in citation or "@misc{" in citation
        assert "title=" in citation
        assert "year=" in citation

    def test_get_citation_raises_for_unknown_tool(self):
        """get_citation raises ValueError for unknown tool keys"""
        with pytest.raises(ValueError, match="Unknown tool"):
            ToolRegistry.get_citation("nonexistent-tool-key")

    def test_get_citation_contains_doi(self):
        """Citations should contain DOI when available"""
        citation = ToolRegistry.get_citation("blast-local-search")
        assert "doi=" in citation

    def test_get_citation_multiple_tools_same_directory(self):
        """Multiple tool keys from same directory return same citation"""
        # pyhmmer has multiple tools that share the same cite.bib
        citation1 = ToolRegistry.get_citation("pyhmmer-hmmscan")
        citation2 = ToolRegistry.get_citation("pyhmmer-hmmsearch")
        citation3 = ToolRegistry.get_citation("pyhmmer-phmmer")
        assert citation1 is not None
        assert citation1 == citation2 == citation3
        assert "pyhmmer" in citation1.lower() or "pyHMMER" in citation1


class TestListCitations:
    """Tests for ToolRegistry.list_citations()"""

    def test_list_citations_returns_dict(self):
        """list_citations returns a dictionary"""
        citations = ToolRegistry.list_citations()
        assert isinstance(citations, dict)

    def test_list_citations_has_expected_count(self):
        """list_citations returns citations for all tools with cite.bib files"""
        citations = ToolRegistry.list_citations()
        # Should have at least 26 citations (one per tool directory)
        # May have more if there are multiple tools per directory
        assert len(citations) >= 26

    def test_list_citations_values_are_bibtex(self):
        """All values in list_citations are valid BibTeX strings"""
        citations = ToolRegistry.list_citations()
        for key, bibtex in citations.items():
            assert isinstance(bibtex, str)
            assert "@" in bibtex, f"Citation for {key} missing @ symbol"
            assert "title=" in bibtex, f"Citation for {key} missing title"

    def test_list_citations_keys_match_registry(self):
        """All keys in list_citations exist in the tool registry"""
        citations = ToolRegistry.list_citations()
        for key in citations:
            # This should not raise
            spec = ToolRegistry.get(key)
            assert spec.key == key
