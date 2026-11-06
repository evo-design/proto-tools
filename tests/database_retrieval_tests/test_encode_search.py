"""Tests for ENCODE portal search."""

import csv
import json
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EncodeSearchConfig,
    EncodeSearchInput,
    run_encode_search,
)
from proto_tools.tools.database_retrieval.encode import encode_search as module

_FILE = {
    "@id": "/files/ENCFF654RRF/",
    "accession": "ENCFF654RRF",
    "dataset": "/experiments/ENCSR000EEB/",
    "href": "/files/ENCFF654RRF/@@download/ENCFF654RRF.bed.gz",
    "assay_title": "TF ChIP-seq",
    "assembly": "GRCh38",
    "biological_replicates": [1, 2],
    "biosample_ontology": {"term_name": "HepG2"},
    "date_created": "2020-12-27T08:00:00.000000+00:00",
    "file_format": "bed",
    "file_size": 1238443,
    "file_type": "bed narrowPeak",
    "lab": {"title": "ENCODE Processing Pipeline"},
    "md5sum": "2dde26453e92db5c7101410349de9a9c",
    "output_type": "IDR thresholded peaks",
    "preferred_default": True,
    "status": "released",
    "target": {"label": "MAFK"},
}
_EXPERIMENT = {
    "@id": "/experiments/ENCSR000EEB/",
    "accession": "ENCSR000EEB",
    "assay_title": "TF ChIP-seq",
    "assembly": ["hg19", "GRCh38"],
    "biosample_ontology": "/biosample-types/cell_line_EFO_0001187/",
    "target": "/targets/MAFK-human/",
    "lab": "/labs/michael-snyder/",
    "date_released": "2011-10-29",
    "description": "MAFK ChIP-seq on human HepG2",
}


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    """Fake ENCODE session that records request query parameters."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.closed = False
        self.payload = payload
        self.status_code = status_code

    def get(self, url: str, **kwargs: Any) -> _Response:
        assert url.startswith(module._SEARCH_URL + "?")
        assert kwargs["headers"]["Accept"] == "application/json"
        self.calls.append(parse_qsl(urlsplit(url).query))
        return _Response(self.payload, self.status_code)

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, fake: _Session) -> _Session:
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: fake)
    return fake


# ── Input validation ─────────────────────────────────────────────────────────


def test_bare_filter_mapping_is_one_query() -> None:
    inputs = EncodeSearchInput(queries={"target.label": " CTCF ", "assembly": "GRCh38"})
    (query,) = inputs.queries
    assert query.filters == {"target.label": "CTCF", "assembly": "GRCh38"} and query.search_term is None
    assert EncodeSearchInput(queries=[{"search_term": " MAFK HepG2 "}]).queries[0].search_term == "MAFK HepG2"


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({}, "at least one filter"),
        ({"search_term": "  "}, "at least one filter"),
        ({"type": "Experiment"}, "must not set"),
        ({"target.label": ""}, "must not be empty"),
    ],
)
def test_invalid_queries_are_rejected(query: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EncodeSearchInput(queries=[query])


# ── Request construction and parsing ─────────────────────────────────────────


def test_file_search_params_and_hit_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session({"@graph": [_FILE], "total": 7}))
    output = run_encode_search(
        EncodeSearchInput(queries={"target.label": "MAFK", "biosample_ontology.term_name": "HepG2"}),
        EncodeSearchConfig(default_only=True, max_results=5),
    )
    assert session.closed is True
    (params,) = session.calls
    assert params[:5] == [
        ("type", "File"),
        ("status", "released"),
        ("preferred_default", "true"),
        ("target.label", "MAFK"),
        ("biosample_ontology.term_name", "HepG2"),
    ]
    assert ("limit", "5") in params and ("field", "md5sum") in params

    (result,) = output.results
    assert result.total_count == 7 and "preferred_default=true" in result.request_url
    (hit,) = result.hits
    assert (hit.accession, hit.dataset_accession, hit.target, hit.biosample) == (
        "ENCFF654RRF",
        "ENCSR000EEB",
        "MAFK",
        "HepG2",
    )
    assert hit.download_url == "https://www.encodeproject.org/files/ENCFF654RRF/@@download/ENCFF654RRF.bed.gz"
    assert hit.portal_url == "https://www.encodeproject.org/files/ENCFF654RRF/"
    assert hit.date_created == "2020-12-27" and hit.biological_replicates == [1, 2]


def test_experiment_search_handles_linked_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session({"@graph": [_EXPERIMENT], "total": 1}))
    output = run_encode_search(
        EncodeSearchInput(queries={"search_term": "MAFK HepG2", "filters": {"status": "archived"}}),
        EncodeSearchConfig(item_type="Experiment", default_only=True),
    )
    (params,) = session.calls
    keys = [key for key, _ in params]
    assert params[0] == ("type", "Experiment") and ("searchTerm", "MAFK HepG2") in params
    assert ("status", "archived") in params and ("status", "released") not in params  # query status wins
    assert "preferred_default" not in keys  # files only
    hit = output.results[0].hits[0]
    assert (hit.accession, hit.dataset_accession, hit.assembly) == ("ENCSR000EEB", "ENCSR000EEB", "hg19, GRCh38")
    assert (hit.target, hit.biosample, hit.lab) == ("MAFK-human", "cell_line_EFO_0001187", "michael-snyder")
    assert hit.download_url is None


def test_no_matches_404_is_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _Session({"@graph": [], "notification": "No results found"}, status_code=404))
    output = run_encode_search(EncodeSearchInput(queries={"target.label": "ZZZ"}), EncodeSearchConfig())
    assert output.results[0].hits == [] and output.results[0].total_count == 0


def test_http_error_propagates_with_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session({"description": "bad request"}, status_code=400))
    with pytest.raises(requests.HTTPError, match=r"ENCODE search failed.*bad request"):
        run_encode_search(EncodeSearchInput(queries={"assembly": "GRCh38"}), EncodeSearchConfig())
    assert session.closed is True


# ── Export ───────────────────────────────────────────────────────────────────


def test_csv_and_json_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _install(monkeypatch, _Session({"@graph": [_FILE, _FILE], "total": 2}))
    output = run_encode_search(EncodeSearchInput(queries={"target.label": "MAFK"}), EncodeSearchConfig())
    output._export_output(tmp_path / "hits", "csv")
    rows = list(csv.DictReader((tmp_path / "hits.csv").open()))
    assert [(row["rank"], row["accession"]) for row in rows] == [("1", "ENCFF654RRF"), ("2", "ENCFF654RRF")]
    assert rows[0]["biological_replicates"] == "1;2" and rows[0]["query_index"] == "0"
    output._export_output(tmp_path / "hits", "json")
    assert json.loads((tmp_path / "hits.json").read_text())["results"][0]["total_count"] == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_encode_search_finds_default_idr_peaks():
    """MAFK HepG2 TF ChIP-seq has one default GRCh38 IDR thresholded peak file, from ENCSR000EEB."""
    output = run_encode_search(
        EncodeSearchInput(
            queries={
                "assay_title": "TF ChIP-seq",
                "target.label": "MAFK",
                "biosample_ontology.term_name": "HepG2",
                "output_type": "IDR thresholded peaks",
                "assembly": "GRCh38",
                "file_format": "bed",
            }
        ),
        EncodeSearchConfig(default_only=True),
    )
    hits = output.results[0].hits
    assert hits and all(hit.preferred_default and hit.assembly == "GRCh38" for hit in hits)
    assert "ENCSR000EEB" in [hit.dataset_accession for hit in hits]
    assert all(hit.download_url and hit.download_url.endswith(".bed.gz") for hit in hits)


@pytest.mark.integration
def test_encode_search_experiments_by_free_text():
    """Free-text experiment search for MAFK HepG2 returns ENCSR000EEB."""
    output = run_encode_search(
        EncodeSearchInput(queries={"search_term": "MAFK HepG2", "filters": {"assay_title": "TF ChIP-seq"}}),
        EncodeSearchConfig(item_type="Experiment", max_results=10),
    )
    assert "ENCSR000EEB" in [hit.accession for hit in output.results[0].hits]
