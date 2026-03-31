"""tests/database_retrieval_tests/test_ncbi_fetch.py

Tests for the NCBI Entrez tools (esearch, esummary, efetch)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    NCBIEfetchInput,
    NCBIEsearchInput,
    NCBIEsummaryInput,
    NCBIFetchConfig,
    run_ncbi_efetch,
    run_ncbi_esearch,
    run_ncbi_esummary,
)
from proto_tools.tools.database_retrieval.ncbi.shared_data_models import (
    _accession_from_header,
    _parse_fasta_records,
)


def test_ncbi_esearch_requires_search_term():
    with pytest.raises(ValidationError):
        NCBIEsearchInput(db="protein")


def test_ncbi_esummary_requires_identifier():
    with pytest.raises(ValidationError):
        NCBIEsummaryInput(db="protein")


def test_ncbi_efetch_requires_identifier():
    with pytest.raises(ValidationError):
        NCBIEfetchInput(db="protein")


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
def test_ncbi_esearch_protein():
    """Search NCBI protein database for lacI."""
    inputs = NCBIEsearchInput(
        db="protein",
        search_term='"lacI"[Gene] AND "Escherichia coli"[Organism]',
        max_results=3,
    )
    output = run_ncbi_esearch(inputs, NCBIFetchConfig())
    assert len(output.ids) > 0


@pytest.mark.integration
def test_ncbi_efetch_fasta():
    """Fetch protein FASTA from NCBI by accession."""
    inputs = NCBIEfetchInput(
        db="protein",
        identifier="NP_000537.3",
        return_format="fasta",
    )
    output = run_ncbi_efetch(inputs, NCBIFetchConfig())
    assert len(output.fasta_records) > 0
    assert output.fasta_records[0].sequence.startswith("M")


@pytest.mark.integration
def test_ncbi_esummary_gene():
    """Retrieve gene summary from NCBI by gene ID."""
    inputs = NCBIEsummaryInput(
        db="gene",
        identifier="7157",
    )
    output = run_ncbi_esummary(inputs, NCBIFetchConfig())
    assert output.summary
    assert output.source_url is not None
