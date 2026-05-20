"""tests/database_retrieval_tests/test_ncbi_fetch.py.

Tests for the NCBI Entrez tools (esearch, esummary, efetch).
"""

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


@pytest.mark.skip_ci
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


@pytest.mark.skip_ci
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


@pytest.mark.integration
def test_ncbi_esearch_retstart_paginates():
    """`retstart` shifts the result window so a paginated second page returns different IDs."""
    page1 = run_ncbi_esearch(
        NCBIEsearchInput(db="protein", search_term="kinase[Title]", max_results=5, retstart=0),
        NCBIFetchConfig(),
    )
    page2 = run_ncbi_esearch(
        NCBIEsearchInput(db="protein", search_term="kinase[Title]", max_results=5, retstart=5),
        NCBIFetchConfig(),
    )
    assert page1.success and page2.success
    assert len(page1.ids) == 5 and len(page2.ids) == 5
    # The two windows must not overlap — that's the whole point of retstart.
    assert set(page1.ids).isdisjoint(set(page2.ids)), (
        f"retstart did not shift the result window: page1={page1.ids}, page2={page2.ids}"
    )


def test_ncbi_config_env_var_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """NCBI_API_KEY / NCBI_EMAIL env vars populate the config defaults."""
    monkeypatch.setenv("NCBI_API_KEY", "env-key-123")
    monkeypatch.setenv("NCBI_EMAIL", "env@example.org")
    cfg = NCBIFetchConfig()
    assert cfg.ncbi_api_key == "env-key-123"
    assert cfg.ncbi_email == "env@example.org"


def test_ncbi_config_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit config value overrides the env var."""
    monkeypatch.setenv("NCBI_API_KEY", "env-key-123")
    monkeypatch.setenv("NCBI_EMAIL", "env@example.org")
    cfg = NCBIFetchConfig(ncbi_api_key="explicit-key", ncbi_email="explicit@example.org")
    assert cfg.ncbi_api_key == "explicit-key"
    assert cfg.ncbi_email == "explicit@example.org"


def test_ncbi_config_no_env_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """With both env vars unset, the config fields default to None."""
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    cfg = NCBIFetchConfig()
    assert cfg.ncbi_api_key is None
    assert cfg.ncbi_email is None
