"""Tests for the NCBI Entrez fetch tool."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.database_retrieval import (
    NCBIFetchConfig,
    NCBIFetchInput,
    run_ncbi_fetch,
)
from bio_programming_tools.tools.database_retrieval.ncbi_fetch.ncbi_fetch import (
    _accession_from_header,
    _parse_fasta_records,
)
from bio_programming_tools.tools.tool_registry import ToolRegistry


def test_ncbi_fetch_is_registered():
    tool_keys = [spec.key for spec in ToolRegistry.list_all()]
    assert "ncbi-fetch" in tool_keys

    schema = ToolRegistry.get_config_schema("ncbi-fetch")
    assert "properties" in schema
    assert "request_timeout_seconds" in schema["properties"]


def test_ncbi_fetch_has_citation():
    citation = ToolRegistry.get_citation("ncbi-fetch")
    assert citation is not None
    assert "@article{" in citation


def test_ncbi_fetch_input_esearch_requires_term():
    with pytest.raises(ValidationError, match="esearch requires search_term"):
        NCBIFetchInput(db="protein", operation="esearch")


def test_ncbi_fetch_input_efetch_requires_identifier():
    with pytest.raises(ValidationError, match="efetch requires identifier"):
        NCBIFetchInput(db="protein", operation="efetch")


def test_parse_fasta_records():
    text = (
        ">sp|P04637|P53_HUMAN Cellular tumor antigen p53\n"
        "MEEPQSDPSVEPPLSQETFSD\n"
        "LWKLLPENNVLSPLPS\n"
        ">sp|P0A6X3|LACI_ECOLI Lac repressor\n"
        "MKPVTLYDVAEYAGVSYQTV\n"
    )
    records = _parse_fasta_records(text)
    assert len(records) == 2
    assert records[0].accession == "P04637"
    assert records[0].sequence == "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS"
    assert records[1].accession == "P0A6X3"


def test_accession_from_header_pipe_delimited():
    assert _accession_from_header("sp|P04637|P53_HUMAN") == "P04637"


def test_accession_from_header_simple():
    assert _accession_from_header("NP_000537.3 tumor protein p53") == "NP_000537.3"


def test_accession_from_header_empty():
    assert _accession_from_header("") is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ncbi_fetch_esearch_protein():
    """Search NCBI protein database for lacI."""
    inputs = NCBIFetchInput(
        db="protein",
        operation="esearch",
        search_term='"lacI"[Gene] AND "Escherichia coli"[Organism]',
        max_results=3,
    )
    output = run_ncbi_fetch(inputs, NCBIFetchConfig())
    assert output.operation == "esearch"
    assert len(output.ids) > 0


@pytest.mark.integration
def test_ncbi_fetch_efetch_fasta():
    """Fetch protein FASTA from NCBI by accession."""
    inputs = NCBIFetchInput(
        db="protein",
        operation="efetch",
        identifier="NP_000537.3",
        rettype="fasta",
    )
    output = run_ncbi_fetch(inputs, NCBIFetchConfig())
    assert output.operation == "efetch"
    assert len(output.fasta_records) > 0
    assert output.fasta_records[0].sequence.startswith("M")
