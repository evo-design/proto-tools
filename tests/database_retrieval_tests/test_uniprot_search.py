"""Tests for ranked UniProtKB search."""

import csv
import json
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    UniProtSearchConfig,
    UniProtSearchInput,
    run_uniprot_search,
)
from proto_tools.tools.database_retrieval.uniprot import uniprot_search as module

_RESULTS = {
    "results": [
        {
            "entryType": "UniProtKB reviewed (Swiss-Prot)",
            "primaryAccession": "P19821",
            "uniProtkbId": "DPO1_THEAQ",
            "annotationScore": 5.0,
            "organism": {"scientificName": "Thermus aquaticus", "taxonId": 271},
            "proteinDescription": {"recommendedName": {"fullName": {"value": "DNA polymerase I, thermostable"}}},
            "genes": [{"geneName": {"value": "polA"}}],
            "sequence": {"value": "MRGMLPLFEPKG", "length": 832},
        },
        {
            "entryType": "UniProtKB unreviewed (TrEMBL)",
            "primaryAccession": "A0A0X1",
            "uniProtkbId": "A0A0X1_THEAQ",
            "organism": {"scientificName": "Thermus aquaticus", "taxonId": 271},
            "proteinDescription": {"submissionNames": [{"fullName": {"value": "DNA polymerase"}}]},
            "sequence": {"value": "MKKL", "length": 4},
        },
    ]
}


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, total: str | None = "10") -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = {"X-Total-Results": total} if total is not None else {}
        self.text = "synthetic body"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    """Fake UniProt session that records GET parameters."""

    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None, total: str | None = "10"):
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.status_code = status_code
        self.payload = _RESULTS if payload is None else payload
        self.total = total

    def get(self, url: str, **kwargs: Any) -> _Response:
        assert url == module._SEARCH_URL
        self.calls.append(kwargs["params"])
        return _Response(self.payload, self.status_code, self.total)

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, fake: _Session) -> _Session:
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: fake)
    return fake


# ── Input validation ─────────────────────────────────────────────────────────


def test_bare_string_is_normalized_and_empty_rejected() -> None:
    assert UniProtSearchInput(queries="  firefly luciferase ").queries == ["firefly luciferase"]
    with pytest.raises(ValidationError, match="must not be empty"):
        UniProtSearchInput(queries=["TP53", " "])


# ── Query construction and parsing ───────────────────────────────────────────


def test_filters_are_combined_and_hits_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session())
    output = run_uniprot_search(
        UniProtSearchInput(queries="DNA polymerase"),
        UniProtSearchConfig(organism="Thermus aquaticus", taxon_id=271, max_results=2),
    )
    assert session.closed is True
    (params,) = session.calls
    assert params["query"] == (
        '(DNA polymerase) AND (organism_name:"Thermus aquaticus") AND (taxonomy_id:271) AND (reviewed:true)'
    )
    assert params["size"] == 2 and params["format"] == "json" and "sequence" not in params["fields"]

    (result,) = output.results
    assert result.total_count == 10 and result.query == params["query"]
    top, second = result.hits
    assert (top.accession, top.entry_name, top.gene_names, top.reviewed) == ("P19821", "DPO1_THEAQ", ["polA"], True)
    assert (top.protein_name, top.taxon_id, top.length, top.sequence) == (
        "DNA polymerase I, thermostable",
        271,
        832,
        None,
    )
    assert second.protein_name == "DNA polymerase" and second.reviewed is False and second.gene_names == []


def test_unfiltered_query_is_sent_verbatim_with_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session(total=None))
    output = run_uniprot_search(
        UniProtSearchInput(queries=["gene:polA", "taq"]),
        UniProtSearchConfig(reviewed_only=False, include_sequence=True),
    )
    assert [params["query"] for params in session.calls] == ["gene:polA", "taq"]
    assert session.calls[0]["fields"].endswith(",sequence")
    assert output.results[0].hits[0].sequence == "MRGMLPLFEPKG"
    assert output.results[0].total_count == 2  # falls back to the hit count without the header


def test_no_hits_is_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _Session(payload={"results": []}, total="0"))
    output = run_uniprot_search(UniProtSearchInput(queries="zzzqqq"), UniProtSearchConfig())
    assert output.results[0].hits == [] and output.results[0].total_count == 0


def test_syntax_error_propagates_with_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session(status_code=400))
    with pytest.raises(requests.HTTPError, match=r"UniProt search failed.*synthetic body"):
        run_uniprot_search(UniProtSearchInput(queries="gene:((("), UniProtSearchConfig())
    assert session.closed is True


# ── Export ───────────────────────────────────────────────────────────────────


def test_csv_fasta_and_json_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _install(monkeypatch, _Session())
    output = run_uniprot_search(UniProtSearchInput(queries="taq"), UniProtSearchConfig(include_sequence=True))
    output._export_output(tmp_path / "hits", "csv")
    rows = list(csv.DictReader((tmp_path / "hits.csv").open()))
    assert [(row["rank"], row["accession"]) for row in rows] == [("1", "P19821"), ("2", "A0A0X1")]
    assert rows[0]["gene_names"] == "polA" and rows[0]["query_index"] == "0"
    output._export_output(tmp_path / "hits", "fasta")
    assert (tmp_path / "hits.fasta").read_text().startswith(">P19821 query=0 DNA polymerase I, thermostable\nMRGM")
    output._export_output(tmp_path / "hits", "json")
    assert json.loads((tmp_path / "hits.json").read_text())["results"][0]["total_count"] == 10


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_uniprot_search_resolves_a_description():
    """A free-text description returns Taq polymerase among the top reviewed hits."""
    output = run_uniprot_search(
        UniProtSearchInput(queries="DNA polymerase Thermus aquaticus"), UniProtSearchConfig(max_results=5)
    )
    hits = output.results[0].hits
    assert "P19821" in [hit.accession for hit in hits]
    assert all(hit.reviewed for hit in hits)


@pytest.mark.integration
def test_uniprot_search_taxon_filter_includes_descendants():
    """taxon_id=40674 (Mammalia) keeps only mammalian p53 entries."""
    output = run_uniprot_search(
        UniProtSearchInput(queries="gene:TP53"), UniProtSearchConfig(taxon_id=40674, max_results=5)
    )
    hits = output.results[0].hits
    assert "P04637" in [hit.accession for hit in hits]
    assert all(hit.organism != "Danio rerio" for hit in hits)
