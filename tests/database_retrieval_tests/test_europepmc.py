"""Tests for Europe PMC search and full-text retrieval."""

import csv
from collections.abc import Callable
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EuropePMCFullTextInput,
    EuropePMCSearchConfig,
    EuropePMCSearchInput,
    run_europepmc_fulltext,
    run_europepmc_search,
)
from proto_tools.tools.database_retrieval.europepmc import europepmc_fulltext as fulltext_module
from proto_tools.tools.database_retrieval.europepmc import europepmc_search as search_module

_SEARCH_PAYLOAD = {
    "hitCount": 175,
    "resultList": {
        "result": [
            {
                "id": "42967907",
                "source": "MED",
                "pmid": "42967907",
                "pmcid": "PMC13724736",
                "doi": "10.1038/s41467-026-77740-5",
                "title": "A peptide bicyclization approach for phage display libraries.",
                "authorString": "Jin G, Liu WR, Binder Consortium.",
                "authorList": {
                    "author": [
                        {"fullName": "Jin G", "firstName": "Guangxu", "lastName": "Jin"},
                        {"fullName": "Liu WR", "lastName": "Liu"},
                        {"collectiveName": "Binder Consortium"},
                    ]
                },
                "journalInfo": {"journal": {"title": "Nature communications"}},
                "pubYear": "2026",
                "firstPublicationDate": "2026-10-03",
                "abstractText": "<h4>Background</h4>Binds with K<sub>d</sub> &lt; 10 nM (n<i>=</i>3).",
                "isOpenAccess": "Y",
                "inEPMC": "Y",
                "citedByCount": 2,
            },
            {
                "id": "PPR1349248",
                "source": "PPR",
                "doi": "10.64898/2026.10.22.760942",
                "title": "Projected Gradient Descent for Hallucination-Based Protein Binder Design",
                "authorString": "Ross M, Miller D, Wells J.",
                "pubYear": "2026",
                "bookOrReportDetails": {"publisher": "bioRxiv"},
                "isOpenAccess": "N",
                "inEPMC": "N",
                "citedByCount": 0,
                "commentCorrectionList": {
                    "commentCorrection": [{"source": "MED", "id": "41000001", "type": "Preprint of"}]
                },
            },
        ]
    },
}

_JATS = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:mml="http://www.w3.org/1998/Math/MathML">
<front><article-meta>
  <article-id pub-id-type="pmcid">PMC123</article-id>
  <article-id pub-id-type="pmid">999</article-id>
  <article-id pub-id-type="doi">10.1/x</article-id>
  <title-group><article-title>Designed <italic>PD-L1</italic> binders</article-title></title-group>
  <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY</license-p></license></permissions>
  <abstract><sec><title>Results</title><p>Binder B1 bound with K<sub>D</sub> = 3 nM.</p></sec></abstract>
  <abstract abstract-type="teaser"><p>Short summary.</p></abstract>
</article-meta></front>
<body>
  <sec><title>Results</title>
    <p>Alanine scanning identified hotspots (Fig. 1).
      <fig id="F1"><label>Figure 1</label><caption><p>Alanine scan of the interface.</p></caption>
        <graphic xlink:href="f1.jpg"/></fig>
    </p>
    <sec><title>Binding assays</title>
      <p>Affinity was 2 <inline-formula><alternatives><mml:math><mml:mo>&#215;</mml:mo></mml:math><tex-math>\\times</tex-math></alternatives></inline-formula> higher.</p>
      <table-wrap id="T1"><label>Table 1</label><caption><p>Affinities.</p></caption>
        <table><thead><tr><th>Binder</th><th>K<sub>D</sub> (nM)</th></tr></thead>
        <tbody><tr><td>B1</td><td>3</td></tr></tbody></table>
        <table-wrap-foot><p>Measured by BLI.</p></table-wrap-foot>
      </table-wrap>
    </sec>
  </sec>
</body>
<back><ack><title>Acknowledgements</title><p>We thank the reviewers.</p></ack>
  <ref-list><ref><mixed-citation>Should not appear.</mixed-citation></ref></ref-list></back>
</article>"""


class _Response:
    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.content = content
        self.text = "synthetic body"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    """Fake session that records requested URLs and GET parameters.

    Responses are returned in order, the last one repeating; a callable picks one per request URL.
    """

    def __init__(self, *responses: _Response | Callable[[str], _Response]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.headers: dict[str, str] = {}
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs.get("params") or {}))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return response(url) if callable(response) else response

    def close(self) -> None:
        self.closed = True


def _lookup_payload(*records: tuple[str, str]) -> dict[str, Any]:
    """Lite search payload listing (pmcid, isOpenAccess) records."""
    return {"resultList": {"result": [{"pmcid": pmcid, "isOpenAccess": oa, "inEPMC": oa} for pmcid, oa in records]}}


# ── Input validation ─────────────────────────────────────────────────────────


def test_fulltext_input_normalizes_pmcids_and_rejects_others() -> None:
    assert EuropePMCFullTextInput(pmcids=" pmc6034017").pmcids == ["PMC6034017"]
    assert EuropePMCFullTextInput(pmcids=["6034017"]).pmcids == ["PMC6034017"]
    with pytest.raises(ValidationError, match="not a PMCID"):
        EuropePMCFullTextInput(pmcids=["10.1038/x"])


# ── Search ───────────────────────────────────────────────────────────────────


def test_search_sends_query_and_parses_core_and_preprint_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Response(payload=_SEARCH_PAYLOAD))
    monkeypatch.setattr(search_module, "build_http_session", lambda **_kwargs: session)
    output = run_europepmc_search(EuropePMCSearchInput(queries="binder SRC:PPR"), EuropePMCSearchConfig(max_results=2))

    assert session.closed is True
    ((url, params),) = session.calls
    assert url == search_module._SEARCH_URL
    assert params["query"] == "binder SRC:PPR" and params["pageSize"] == 2 and params["resultType"] == "core"

    (result,) = output.results
    assert result.total_count == 175
    paper, preprint = result.hits
    assert (paper.pmcid, paper.journal, paper.year, paper.is_open_access) == (
        "PMC13724736",
        "Nature communications",
        2026,
        True,
    )
    assert paper.abstract == "Background Binds with Kd < 10 nM (n=3)."
    assert paper.source_url == "https://europepmc.org/article/MED/42967907"
    assert (preprint.journal, preprint.pmid, preprint.has_full_text) == ("bioRxiv", None, False)
    assert (paper.published_pmid, preprint.published_pmid) == (None, "41000001")
    assert paper.authors == ["Jin G", "Liu WR", "Binder Consortium"]
    assert preprint.authors == ["Ross M", "Miller D", "Wells J"]


def test_search_rejected_query_is_an_item_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Response(status_code=400), _Response(payload=_SEARCH_PAYLOAD))
    monkeypatch.setattr(search_module, "build_http_session", lambda **_kwargs: session)
    output = run_europepmc_search(EuropePMCSearchInput(queries=["TITLE:(((", "binder"]), EuropePMCSearchConfig())
    bad, good = output.results
    assert bad.query == "TITLE:(((" and bad.hits == [] and bad.total_count is None
    assert bad.error == "Europe PMC search failed: 400 Client Error; body='synthetic body'"
    assert good.error is None and len(good.hits) == 2
    assert session.closed is True


@pytest.mark.parametrize("status_code", [429, 500])
def test_search_server_error_raises(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    session = _Session(_Response(status_code=status_code))
    monkeypatch.setattr(search_module, "build_http_session", lambda **_kwargs: session)
    with pytest.raises(requests.HTTPError, match=r"Europe PMC search failed.*synthetic body"):
        run_europepmc_search(EuropePMCSearchInput(queries="binder"), EuropePMCSearchConfig())
    assert session.closed is True


def test_search_csv_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    session = _Session(_Response(payload=_SEARCH_PAYLOAD), _Response(status_code=400))
    monkeypatch.setattr(search_module, "build_http_session", lambda **_kwargs: session)
    output = run_europepmc_search(EuropePMCSearchInput(queries=["binder", "TITLE:((("]), EuropePMCSearchConfig())
    output._export_output(tmp_path / "hits", "csv")
    rows = list(csv.DictReader((tmp_path / "hits.csv").open()))
    assert [(row["rank"], row["id"]) for row in rows] == [("1", "42967907"), ("2", "PPR1349248"), ("", "")]
    assert rows[0]["authors"] == "Jin G;Liu WR;Binder Consortium"
    assert rows[2]["error"].startswith("Europe PMC search failed: 400")


# ── Full text ────────────────────────────────────────────────────────────────


def test_parse_jats_sections_floats_and_tables() -> None:
    document = fulltext_module._parse_jats(_JATS, pmcid="PMC123", source_url="u")
    assert (document.pmid, document.doi, document.title) == ("999", "10.1/x", "Designed PD-L1 binders")
    assert document.license == "https://creativecommons.org/licenses/by/4.0/"

    sections = [(s.kind, s.heading) for s in document.sections]
    assert sections == [
        ("title", None),
        ("abstract", "Abstract"),
        ("abstract", "Abstract (teaser)"),
        ("section", "Results"),
        ("figure", "Figure 1"),
        ("section", "Results > Binding assays"),
        ("table", "Table 1"),
        ("section", "Acknowledgements"),
    ]
    by_heading = {s.heading: s.text for s in document.sections}
    assert by_heading["Abstract"] == "Results Binder B1 bound with KD = 3 nM."
    assert by_heading["Results"] == "Alanine scanning identified hotspots (Fig. 1)."
    assert by_heading["Figure 1"] == "Alanine scan of the interface."
    assert by_heading["Results > Binding assays"] == "Affinity was 2 \u00d7 higher."
    assert by_heading["Table 1"] == "Affinities.\nBinder | KD (nM)\nB1 | 3\nMeasured by BLI."
    assert "Should not appear" not in document.text


def test_fulltext_fetches_each_pmcid(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    session = _Session(_Response(payload=_lookup_payload(("PMC123", "Y"))), _Response(content=_JATS))
    monkeypatch.setattr(fulltext_module, "build_http_session", lambda **_kwargs: session)
    output = run_europepmc_fulltext(EuropePMCFullTextInput(pmcids=["PMC123"]))
    assert session.closed is True
    assert session.calls[1][0] == "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123/fullTextXML"
    assert output.documents[0].source_url == session.calls[1][0]
    output._export_output(tmp_path / "doc", "txt")
    assert "## [table] Table 1\nAffinities." in (tmp_path / "doc.txt").read_text()


def test_fulltext_mixed_batch_returns_item_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Missing, non-open-access, and 404 articles get errors; the open-access one is still read."""
    lookup = _Response(payload=_lookup_payload(("PMC123", "Y"), ("PMC7612244", "N"), ("PMC404", "Y")))

    def respond(url: str) -> _Response:
        if url == search_module._SEARCH_URL:
            return lookup
        return _Response(status_code=404) if "PMC404" in url else _Response(content=_JATS)

    session = _Session(respond)
    monkeypatch.setattr(fulltext_module, "build_http_session", lambda **_kwargs: session)
    pmcids = ["PMC999", "PMC123", "PMC7612244", "PMC404"]
    output = run_europepmc_fulltext(EuropePMCFullTextInput(pmcids=pmcids))

    assert [doc.pmcid for doc in output.documents] == pmcids
    missing, found, closed, not_found = output.documents
    assert missing.error == "PMC999 is not in Europe PMC" and missing.sections == []
    assert closed.error is not None and "not open access" in closed.error and closed.title is None
    assert not_found.error == "Europe PMC full text for PMC404 failed: 404 Client Error; body='synthetic body'"
    assert found.error is None and found.title == "Designed PD-L1 binders"
    lookups = [params for url, params in session.calls if url == search_module._SEARCH_URL]
    assert [params["pageSize"] for params in lookups] == [1000]
    fetched = [url for url, _params in session.calls if url != search_module._SEARCH_URL]
    assert fetched == [fulltext_module._FULLTEXT_URL.format(pmcid=p) for p in ("PMC123", "PMC404")]
    output._export_output(tmp_path / "doc", "txt")
    assert "# PMC999 (error: PMC999 is not in Europe PMC)" in (tmp_path / "doc.txt").read_text()


def test_fulltext_open_access_lookup_is_one_request_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(_Response(payload=_lookup_payload()))
    monkeypatch.setattr(fulltext_module, "build_http_session", lambda **_kwargs: session)
    monkeypatch.setattr(fulltext_module, "_LOOKUP_BATCH_SIZE", 3)
    output = run_europepmc_fulltext(EuropePMCFullTextInput(pmcids=[f"PMC{i}" for i in range(1, 6)]))

    assert [params for _url, params in session.calls] == [
        {"query": "PMCID:PMC1 OR PMCID:PMC2 OR PMCID:PMC3", "resultType": "lite", "format": "json", "pageSize": 1000},
        {"query": "PMCID:PMC4 OR PMCID:PMC5", "resultType": "lite", "format": "json", "pageSize": 1000},
    ]
    assert all(doc.error for doc in output.documents)


@pytest.mark.parametrize("lookup_fails", [True, False])
def test_fulltext_server_error_raises(monkeypatch: pytest.MonkeyPatch, lookup_fails: bool) -> None:
    lookup = _Response(status_code=500) if lookup_fails else _Response(payload=_lookup_payload(("PMC123", "Y")))
    session = _Session(lookup, _Response(status_code=500))
    monkeypatch.setattr(fulltext_module, "build_http_session", lambda **_kwargs: session)
    with pytest.raises(requests.HTTPError, match="synthetic body"):
        run_europepmc_fulltext(EuropePMCFullTextInput(pmcids=["PMC123"]))
    assert session.closed is True


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_europepmc_search_finds_published_rfdiffusion_by_doi():
    """A DOI field query resolves the RFdiffusion paper to its PMID and PMCID."""
    output = run_europepmc_search(
        EuropePMCSearchInput(queries='DOI:"10.1038/s41586-023-06415-8"'), EuropePMCSearchConfig(max_results=1)
    )
    (hit,) = output.results[0].hits
    assert (hit.pmid, hit.pmcid, hit.is_open_access) == ("37433327", "PMC10468394", True)
    assert "diffusion" in (hit.abstract or "").lower()


@pytest.mark.integration
def test_europepmc_search_preprints_only():
    """SRC:PPR restricts hits to preprints."""
    output = run_europepmc_search(
        EuropePMCSearchInput(queries='TITLE:"protein binder" AND SRC:PPR'), EuropePMCSearchConfig(max_results=5)
    )
    hits = output.results[0].hits
    assert hits and all(hit.source == "PPR" for hit in hits)


@pytest.mark.integration
def test_europepmc_fulltext_reads_tables_from_an_open_access_article():
    """An open-access article returns its title, abstract, body sections and tables."""
    document = run_europepmc_fulltext(EuropePMCFullTextInput(pmcids=["PMC6034017"])).documents[0]
    kinds = {section.kind for section in document.sections}
    assert {"title", "abstract", "section", "figure", "table"} <= kinds
    assert document.pmid == "29961454"
    assert any(s.heading == "Table 1" and "53.86" in s.text for s in document.sections)
