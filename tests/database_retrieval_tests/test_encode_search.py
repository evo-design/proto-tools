"""Tests for ENCODE portal search."""

import csv
import hashlib
import json
from collections.abc import Callable
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
    """Fake ENCODE session that records request query parameters.

    ``route`` answers requests by their parameters (e.g. diagnosis counts); otherwise every request
    gets ``payload``.
    """

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        status_code: int = 200,
        route: Callable[[dict[str, str]], _Response] | None = None,
    ) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.closed = False
        self.payload = payload or {}
        self.status_code = status_code
        self.route = route

    def get(self, url: str, **kwargs: Any) -> _Response:
        assert url.startswith(module._SEARCH_URL + "?")
        assert kwargs["headers"]["Accept"] == "application/json"
        params = parse_qsl(urlsplit(url).query)
        self.calls.append(params)
        if self.route is not None:
            return self.route(dict(params))
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
        ({"target.label": ["CTCF", " "]}, "must not be empty"),
        ({"target.label": []}, "at least one value"),
    ],
)
def test_invalid_queries_are_rejected(query: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EncodeSearchInput(queries=[query])


# ── Request construction and parsing ─────────────────────────────────────────


def test_list_filters_repeat_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session({"@graph": [_FILE], "total": 14}))
    output = run_encode_search(
        EncodeSearchInput(queries={"target.label": [" CTCF ", "RAD21"], "assembly!": ["hg19", "GRCh37"]}),
        EncodeSearchConfig(),
    )
    query = output.results[0].query
    assert query.filters == {"target.label": ["CTCF", "RAD21"], "assembly!": ["hg19", "GRCh37"]}
    params = session.calls[0]
    assert [pair for pair in params if pair[0] in {"target.label", "assembly!"}] == [
        ("target.label", "CTCF"),
        ("target.label", "RAD21"),
        ("assembly!", "hg19"),
        ("assembly!", "GRCh37"),
    ]


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


# ── Organism ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            {"replicates": [{"library": {"biosample": {"organism": {"scientific_name": "Mus musculus"}}}}]},
            "Mus musculus",
        ),
        ({"donors": ["/human-donors/ENCDO000ABC/"], "assembly": "GRCh38"}, "Homo sapiens"),
        ({"donors": ["/human-donors/A/", "/mouse-donors/B/"]}, "Homo sapiens, Mus musculus"),
        ({"donors": ["/fly-donors/ENCDO000FLY/"], "assembly": "dm6"}, "Drosophila melanogaster"),
        ({"donors": ["/fly-donors/ENCDO000FLY/"]}, None),  # fly donors span several species
        ({"assembly": ["hg19", "GRCh38"]}, "Homo sapiens"),
        ({"assembly": "unknown-build"}, None),
    ],
)
def test_organism_comes_from_biosamples_donors_or_assembly(item: dict[str, Any], expected: str | None) -> None:
    assert module._organism(item, item.get("assembly")) == expected


# ── Filter diagnosis ─────────────────────────────────────────────────────────

_FACETS = {
    "facets": [
        {"field": "biosample_ontology.term_name", "terms": [{"key": "K562"}, {"key": "HepG2"}]},
        {"field": "target.label", "terms": [{"key": "CTCF"}, {"key": "MAFK"}]},
        {"field": "assembly", "terms": [{"key": "GRCh38"}, {"key": "mm10"}]},
    ]
}
_KNOWN = {("biosample_ontology.term_name", "K562"), ("target.label", "CTCF"), ("assembly", "GRCh38")}


def _portal(search: dict[str, Any]) -> Callable[[dict[str, str]], _Response]:
    """Answer the search with ``search``, and count requests (limit=0) by whether the filter is known."""

    def route(params: dict[str, str]) -> _Response:
        if params["limit"] != "0":
            return _Response(search, 200 if search["@graph"] else 404)
        filters = {(key, val) for key, val in params.items() if key not in {"type", "status", "format", "limit"}}
        if not filters:
            return _Response({"@graph": [], "total": 50, **_FACETS})
        return _Response(
            {"@graph": [], "total": 9} if filters <= _KNOWN else {"@graph": []}, 200 if filters <= _KNOWN else 404
        )

    return route


_NO_HITS = {"@graph": [], "notification": "No results found"}


@pytest.mark.parametrize(
    ("filters", "fragments"),
    [
        (
            {"target.label": "CTCF", "biosample_ontology.term_name": "k562"},
            ["biosample_ontology.term_name='k562' matches no released File", "(did you mean 'K562'?)"],
        ),
        ({"target.lable": "CTCF"}, ["'target.lable' is not a File search facet; did you mean 'target.label'?"]),
        (
            {"assembly": "GRCh38", "target.label": "ZZZ9"},
            ["target.label='ZZZ9' matches no released File", "(values must match the portal's terms exactly)"],
        ),
    ],
)
def test_filters_matching_nothing_are_explained(
    monkeypatch: pytest.MonkeyPatch, filters: dict[str, str], fragments: list[str]
) -> None:
    _install(monkeypatch, _Session(route=_portal(_NO_HITS)))
    output = run_encode_search(EncodeSearchInput(queries=filters), EncodeSearchConfig())
    (warning,) = output.warnings
    assert all(fragment in warning for fragment in fragments), warning


def test_valid_filters_with_no_common_match_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session(route=_portal(_NO_HITS)))
    output = run_encode_search(
        EncodeSearchInput(queries={"target.label": "CTCF", "assembly": "GRCh38"}), EncodeSearchConfig()
    )
    assert output.results[0].total_count == 0 and output.warnings == []
    assert len(session.calls) == 3  # the search, then one count per filter; no facet request


def test_negated_filter_that_excludes_nothing_warns_even_with_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session(route=_portal({"@graph": [_FILE], "total": 1})))
    output = run_encode_search(
        EncodeSearchInput(queries={"target.label": "CTCF", "assembly!": "grch38"}), EncodeSearchConfig()
    )
    assert output.results[0].total_count == 1
    (warning,) = output.warnings
    assert "assembly!='grch38' excludes nothing" in warning and "did you mean 'GRCh38'?" in warning
    assert dict(session.calls[1])["assembly"] == "grch38"  # counted in its positive form

    session.calls.clear()
    output = run_encode_search(
        EncodeSearchInput(queries={"target.label": "CTCF", "assembly!": "GRCh38"}), EncodeSearchConfig()
    )
    assert output.warnings == [] and len(session.calls) == 2  # only the negated filter is counted


def test_list_value_that_matches_nothing_warns_even_with_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session(route=_portal({"@graph": [_FILE], "total": 1})))
    output = run_encode_search(
        EncodeSearchInput(queries={"target.label": ["CTCF", "ctcf"], "assembly": "GRCh38"}), EncodeSearchConfig()
    )
    (warning,) = output.warnings
    assert "target.label='ctcf' in a list matches no released File, so it adds no hits" in warning
    assert "did you mean 'CTCF'?" in warning
    counts = [dict(params) for params in session.calls[1:]]
    assert [params.get("target.label") for params in counts] == ["CTCF", "ctcf", None]  # then the facet lookup
    assert not any("assembly" in params for params in counts)  # single values are not counted when there are hits


# ── Download ─────────────────────────────────────────────────────────────────

_CONTENT = b"chr1\t100\t200\tpeak1\n" * 1000


class _Stream:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 200
        self.text = ""
        self.closed = False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int) -> Any:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class _DownloadSession:
    """Fake session serving file content and recording requested URLs."""

    def __init__(self, content: bytes = _CONTENT) -> None:
        self.urls: list[str] = []
        self.content = content

    def get(self, url: str, **kwargs: Any) -> _Stream:
        assert kwargs["stream"] is True
        self.urls.append(url)
        return _Stream(self.content)

    def close(self) -> None:
        pass


def _search_output(*files: dict[str, Any]) -> Any:
    search = _Session({"@graph": list(files), "total": len(files)})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "build_http_session", lambda **_kwargs: search)
        return run_encode_search(EncodeSearchInput(queries={"target.label": "MAFK"}), EncodeSearchConfig())


def _file(accession: str, content: bytes = _CONTENT) -> dict[str, Any]:
    return {
        **_FILE,
        "@id": f"/files/{accession}/",
        "accession": accession,
        "href": f"/files/{accession}/@@download/{accession}.bed.gz",
        "file_size": len(content),
        "md5sum": hashlib.md5(content).hexdigest(),  # noqa: S324 -- test fixture checksum
    }


def test_download_verifies_md5_and_skips_files_already_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    output = _search_output(_file("ENCFF000AAA"), _file("ENCFF000BBB"))
    session = _DownloadSession()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: session)

    paths = output.download(tmp_path / "encode")
    assert list(paths) == ["ENCFF000AAA", "ENCFF000BBB"]
    assert paths["ENCFF000AAA"] == tmp_path / "encode" / "ENCFF000AAA.bed.gz"
    assert paths["ENCFF000AAA"].read_bytes() == _CONTENT
    assert session.urls == [f"https://www.encodeproject.org/files/{a}/@@download/{a}.bed.gz" for a in paths]
    assert sorted(path.name for path in (tmp_path / "encode").iterdir()) == ["ENCFF000AAA.bed.gz", "ENCFF000BBB.bed.gz"]

    session.urls.clear()
    assert output.download(tmp_path / "encode", ["ENCFF000BBB"]) == {"ENCFF000BBB": paths["ENCFF000BBB"]}
    assert session.urls == []  # already present with the right MD5


def test_download_rejects_a_corrupt_file_and_leaves_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    output = _search_output(_file("ENCFF000AAA"))
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: _DownloadSession(b"truncated"))
    with pytest.raises(OSError, match=r"ENCFF000AAA MD5 .* does not match"):
        output.download(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_refuses_over_budget_before_any_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    output = _search_output(_file("ENCFF000AAA"), _file("ENCFF000BBB"))
    session = _DownloadSession()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: session)
    with pytest.raises(ValueError, match=r"Downloading 2 files needs .* over max_total_bytes"):
        output.download(tmp_path, max_total_bytes=len(_CONTENT))
    assert session.urls == []
    assert len(output.download(tmp_path, max_total_bytes=None)) == 2


def test_download_rejects_unknown_accessions_and_experiment_hits(tmp_path: Any) -> None:
    output = _search_output(_file("ENCFF000AAA"))
    with pytest.raises(ValueError, match=r"Not file hits .*ENCFF999ZZZ"):
        output.download(tmp_path, ["ENCFF999ZZZ"])
    with pytest.raises(ValueError, match="experiment searches have no download URLs"):
        _search_output(_EXPERIMENT).download(tmp_path)


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


@pytest.mark.network
@pytest.mark.uses_env
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


@pytest.mark.network
@pytest.mark.uses_env
def test_encode_search_experiments_by_free_text():
    """Free-text experiment search for MAFK HepG2 returns ENCSR000EEB."""
    output = run_encode_search(
        EncodeSearchInput(queries={"search_term": "MAFK HepG2", "filters": {"assay_title": "TF ChIP-seq"}}),
        EncodeSearchConfig(item_type="Experiment", max_results=10),
    )
    assert "ENCSR000EEB" in [hit.accession for hit in output.results[0].hits]
