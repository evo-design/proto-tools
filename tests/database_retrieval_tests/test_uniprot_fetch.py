"""tests/database_retrieval_tests/test_uniprot_fetch.py.

Tests for the UniProt fetch tool.
"""

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    UniProtFetchConfig,
    UniProtFetchInput,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.uniprot import uniprot_fetch
from proto_tools.tools.database_retrieval.uniprot.uniprot_fetch import (
    _entry_priority,
    _extract_gene_names,
    _extract_pdb_crossrefs,
    _fetch_entry,
)


def test_uniprot_fetch_input_requires_id_or_name():
    with pytest.raises(
        ValidationError,
        match="Provide either uniprot_id or both target_name and organism",
    ):
        UniProtFetchInput()


def test_uniprot_fetch_input_name_without_organism():
    with pytest.raises(
        ValidationError,
        match="Provide either uniprot_id or both target_name and organism",
    ):
        UniProtFetchInput(target_name="TP53")


def test_extract_gene_names():
    entry = {
        "genes": [
            {"geneName": {"value": "TP53"}},
            {"geneName": {"value": "P53"}},
        ]
    }
    names = _extract_gene_names(entry)
    assert "tp53" in names
    assert "p53" in names


def test_extract_gene_names_empty():
    assert _extract_gene_names({}) == set()
    assert _extract_gene_names({"genes": "not_a_list"}) == set()


def test_extract_pdb_crossrefs():
    entry = {
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": "1TSR"},
            {"database": "PDB", "id": "2OCJ"},
            {"database": "EMBL", "id": "M14694"},
        ]
    }
    pdb_ids = _extract_pdb_crossrefs(entry)
    assert pdb_ids == ["1TSR", "2OCJ"]


def test_extract_pdb_crossrefs_empty():
    assert _extract_pdb_crossrefs({}) == []


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"primaryAccession": "P13569"}


class _ResetThenOkSession:
    """A session whose ``get`` mimics a reused keep-alive connection the server already closed."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.exceptions.ConnectionError(
                "Connection aborted.", ConnectionResetError(104, "Connection reset by peer")
            )
        return _FakeResponse()


def test_fetch_entry_retries_connection_reset(monkeypatch):
    """A connection reset on a reused connection is retried rather than raised.

    Reproduces the ``rest.uniprot.org`` failure directly: urllib3's adapter-level ``Retry`` does
    not catch a reset that happens while sending on a pooled connection, so ``_fetch_entry`` must.
    """
    monkeypatch.setattr(uniprot_fetch, "_BACKOFF_SECONDS", 0.0)
    session = _ResetThenOkSession(failures=2)
    entry = _fetch_entry("P13569", session)  # type: ignore[arg-type]
    assert entry == {"primaryAccession": "P13569"}
    assert session.calls == 3


def test_fetch_entry_gives_up_after_configured_retries(monkeypatch):
    monkeypatch.setattr(uniprot_fetch, "_BACKOFF_SECONDS", 0.0)
    session = _ResetThenOkSession(failures=5)
    with pytest.raises(requests.exceptions.ConnectionError):
        _fetch_entry("P13569", session)  # type: ignore[arg-type]
    assert session.calls == 3


def test_entry_priority_distinguishes_reviewed_from_unreviewed():
    """Regression: substring "reviewed" in entryType used to match TrEMBL too.

    The pre-fix substring check `"reviewed" in entry_type` was True for both
    "UniProtKB reviewed (Swiss-Prot)" and "UniProtKB unreviewed (TrEMBL)",
    silently inverting the ranking and promoting longer TrEMBL accessions
    over the canonical Swiss-Prot entry.
    """
    swissprot = {
        "primaryAccession": "P04637",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "genes": [{"geneName": {"value": "TP53"}}],
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "1TUP"}],
    }
    trembl = {
        "primaryAccession": "A0A0U1RQC9",
        "entryType": "UniProtKB unreviewed (TrEMBL)",
        "genes": [{"geneName": {"value": "TP53"}}],
        "uniProtKBCrossReferences": [],
    }
    # Position 2 of the priority tuple is the `reviewed` flag (0/1).
    assert _entry_priority(swissprot, "TP53", prefer_pdb_crossref=False)[2] == 1
    assert _entry_priority(trembl, "TP53", prefer_pdb_crossref=False)[2] == 0
    # And max() picks the Swiss-Prot entry, not the longer TrEMBL accession.
    winner = max(
        [swissprot, trembl],
        key=lambda e: _entry_priority(e, "TP53", prefer_pdb_crossref=False),
    )
    assert winner["primaryAccession"] == "P04637"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_uniprot_fetch_by_accession():
    """Fetch TP53 protein by UniProt accession."""
    inputs = UniProtFetchInput(uniprot_id="P04637")
    output = run_uniprot_fetch(inputs, UniProtFetchConfig())
    assert output.accession == "P04637"
    assert output.sequence is not None
    assert output.sequence.startswith("M")
    assert len(output.pdb_crossrefs) > 0


@pytest.mark.integration
def test_uniprot_fetch_by_name():
    """Search UniProt for lacI in E. coli."""
    inputs = UniProtFetchInput(
        target_name="lacI",
        organism="Escherichia coli",
    )
    output = run_uniprot_fetch(inputs, UniProtFetchConfig())
    assert output.accession
    assert output.sequence is not None
    assert output.sequence.startswith("M")


@pytest.mark.integration
@pytest.mark.parametrize(
    "target_name, organism, expected_accession",
    [
        ("TP53", "Homo sapiens", "P04637"),
        ("KRAS", "Homo sapiens", "P01116"),
        ("BRCA1", "Homo sapiens", "P38398"),
    ],
    ids=["TP53", "KRAS", "BRCA1"],
)
def test_uniprot_fetch_by_name_returns_canonical_swissprot(target_name, organism, expected_accession):
    """Default symbol search returns the canonical Swiss-Prot entry, not a TrEMBL hit.

    Pre-fix, the substring bug surfaced long TrEMBL accessions for these
    common human genes; the ranker now picks Swiss-Prot without needing
    `prefer_pdb_crossref=True`.
    """
    output = run_uniprot_fetch(
        UniProtFetchInput(target_name=target_name, organism=organism),
        UniProtFetchConfig(),
    )
    assert output.success
    assert output.accession == expected_accession
    assert "swiss-prot" in (output.entry_type or "").lower()


@pytest.mark.integration
def test_uniprot_fetch_with_fields_shrinks_response_and_populates_typed_output():
    """`fields` config restricts the API response and only populates requested typed fields.

    Without `fields`, UniProt returns ~880 KB for human TP53 with every
    annotation populated. A targeted selection collapses the response to
    well under 10% of that size while still populating the typed Output
    fields the caller requested.
    """
    minimal = run_uniprot_fetch(
        UniProtFetchInput(uniprot_id="P04637"),
        UniProtFetchConfig(fields=["accession", "sequence", "gene_names", "xref_pdb"]),
    )
    full = run_uniprot_fetch(
        UniProtFetchInput(uniprot_id="P04637"),
        UniProtFetchConfig(),
    )

    # Same canonical sequence is returned in both cases — the API filter only
    # restricts what comes back, not which entry is selected.
    assert minimal.success and full.success
    assert minimal.accession == full.accession == "P04637"
    assert minimal.sequence == full.sequence
    assert minimal.gene_names == full.gene_names
    assert minimal.pdb_crossrefs == full.pdb_crossrefs

    # The minimal raw_entry must be meaningfully smaller — payload reduction
    # is the whole point of this setting. UniProt always returns `entryType`,
    # `extraAttributes`, and `primaryAccession` in the response envelope, so
    # the floor is ~hundreds of bytes; the full TP53 record is ~880 KB.
    assert len(str(minimal.raw_entry)) < len(str(full.raw_entry)) / 5, (
        "minimal `fields` request should shrink raw_entry by >5x; got "
        f"minimal={len(str(minimal.raw_entry))} vs full={len(str(full.raw_entry))}"
    )

    # Fields not requested are absent from raw_entry. `comments` carries
    # function annotations / disease info / subcellular location etc. — none
    # of which we asked for.
    assert "comments" not in minimal.raw_entry
    assert "comments" in full.raw_entry
    assert "features" not in minimal.raw_entry
    assert "features" in full.raw_entry


@pytest.mark.integration
def test_uniprot_fetch_with_fields_in_search_mode():
    """`fields` works in search mode and the ranker still picks the canonical Swiss-Prot."""
    output = run_uniprot_fetch(
        UniProtFetchInput(target_name="TP53", organism="Homo sapiens"),
        UniProtFetchConfig(fields=["accession", "gene_names", "reviewed", "xref_pdb"]),
    )
    assert output.success
    assert output.accession == "P04637"
    assert "swiss-prot" in (output.entry_type or "").lower()
    # The ranker reads `entryType`, `genes`, and PDB cross-refs — including
    # the corresponding fields keeps ranking quality intact.
    assert output.gene_names
    assert len(output.pdb_crossrefs) > 0
