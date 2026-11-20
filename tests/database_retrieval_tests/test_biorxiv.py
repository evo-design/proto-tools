"""Tests for bioRxiv preprint details."""

from typing import Any

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import BiorxivDetailsConfig, BiorxivDetailsInput, run_biorxiv_details
from proto_tools.tools.database_retrieval.biorxiv import biorxiv_details as module

_DOI = "10.1101/2022.12.09.519842"
_MEDRXIV_DOI = "10.1101/2020.03.04.20031047"
_NOT_FOUND = {"messages": [{"status": "no posts found"}], "collection": []}


def _record(version: str, date: str, title: str) -> dict[str, Any]:
    return {
        "title": title,
        "authors": "Watson, J. L.; Juergens, D.; Baker, D.",
        "author_corresponding": "David Baker",
        "author_corresponding_institution": "University of Washington",
        "doi": _DOI,
        "date": date,
        "version": version,
        "type": "new results",
        "license": "cc_by_nd",
        "category": "biochemistry",
        "jatsxml": f"https://www.biorxiv.org/content/{version}.source.xml",
        "abstract": f"Abstract v{version}",
        "published": "10.1038/s41586-023-06415-8",
        "server": "bioRxiv",
    }


def _url(server: str, doi: str) -> str:
    return f"https://api.biorxiv.org/details/{server}/{doi}"


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    """Fake bioRxiv session that records requested URLs; ``routes`` overrides the default reply per URL."""

    def __init__(
        self,
        payload: dict[str, Any] = _NOT_FOUND,
        status_code: int = 200,
        routes: dict[str, tuple[dict[str, Any], int]] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.routes = routes or {}
        self.urls: list[str] = []
        self.closed = False

    def get(self, url: str, **_kwargs: Any) -> _Response:
        self.urls.append(url)
        return _Response(*self.routes.get(url, (self.payload, self.status_code)))

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, fake: _Session) -> _Session:
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: fake)
    return fake


# ── Input validation ─────────────────────────────────────────────────────────


def test_doi_prefixes_are_stripped_and_non_dois_rejected() -> None:
    assert BiorxivDetailsInput(dois=f"https://doi.org/{_DOI}").dois == [_DOI]
    assert BiorxivDetailsInput(dois=[f"doi: {_DOI}"]).dois == [_DOI]
    with pytest.raises(ValidationError, match="not a DOI"):
        BiorxivDetailsInput(dois=["2022.12.09.519842"])


# ── Parsing ──────────────────────────────────────────────────────────────────


def test_versions_are_ordered_and_latest_metadata_used(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"collection": [_record("2", "2022-12-14", "Title v2"), _record("1", "2022-12-10", "Title v1")]}
    session = _install(monkeypatch, _Session(payload))
    output = run_biorxiv_details(BiorxivDetailsInput(dois=_DOI), BiorxivDetailsConfig(server="medrxiv"))

    assert session.closed is True
    assert session.urls == [_url("medrxiv", _DOI)]
    (preprint,) = output.preprints
    assert [(v.version, v.date) for v in preprint.versions] == [(1, "2022-12-10"), (2, "2022-12-14")]
    assert (preprint.title, preprint.abstract) == ("Title v2", "Abstract v2")
    assert preprint.authors == ["Watson, J. L.", "Juergens, D.", "Baker, D."]
    assert preprint.published_doi == "10.1038/s41586-023-06415-8"
    assert preprint.source_url == session.urls[0]


def test_unpublished_preprint_has_no_published_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {**_record("1", "2020-08-07", "T"), "published": "NA"}
    _install(monkeypatch, _Session({"collection": [record]}))
    assert run_biorxiv_details(BiorxivDetailsInput(dois=_DOI)).preprints[0].published_doi is None


# ── Server selection ─────────────────────────────────────────────────────────


def test_auto_falls_back_to_medrxiv(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {**_record("1", "2020-03-06", "T"), "doi": _MEDRXIV_DOI, "server": "medRxiv"}
    routes = {_url("medrxiv", _MEDRXIV_DOI): ({"collection": [record]}, 200)}
    session = _install(monkeypatch, _Session(routes=routes))
    (preprint,) = run_biorxiv_details(BiorxivDetailsInput(dois=_MEDRXIV_DOI)).preprints

    assert session.urls == [_url("biorxiv", _MEDRXIV_DOI), _url("medrxiv", _MEDRXIV_DOI)]
    assert (preprint.error, preprint.server, preprint.source_url) == (None, "medRxiv", session.urls[1])


def test_explicit_server_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _install(monkeypatch, _Session())
    (preprint,) = run_biorxiv_details(BiorxivDetailsInput(dois=_DOI), BiorxivDetailsConfig(server="medrxiv")).preprints

    assert session.urls == [_url("medrxiv", _DOI)]
    assert preprint.error == f"No preprint found for DOI {_DOI} on medrxiv (no posts found)"


# ── Item errors ──────────────────────────────────────────────────────────────


def test_unknown_doi_in_batch_is_an_item_error(monkeypatch: pytest.MonkeyPatch) -> None:
    found = {"collection": [_record("1", "2022-12-10", "T")]}
    routes = {_url("biorxiv", _DOI): (found, 200), _url("biorxiv", "10.1101/bad"): ({}, 400)}
    _install(monkeypatch, _Session(routes=routes))
    output = run_biorxiv_details(BiorxivDetailsInput(dois=[_DOI, "10.1101/0000", "10.1101/bad"]))

    assert [p.doi for p in output.preprints] == [_DOI, "10.1101/0000", "10.1101/bad"]
    hit, missing, rejected = output.preprints
    assert (hit.error, hit.title) == (None, "T")
    assert missing.error == "No preprint found for DOI 10.1101/0000 on biorxiv or medrxiv (no posts found)"
    assert (missing.title, missing.server, missing.versions) == (None, None, [])
    assert rejected.error is not None and rejected.error.startswith(
        "biorxiv details request for DOI 10.1101/bad failed: 400"
    )


@pytest.mark.parametrize("status_code", [429, 503])
def test_rate_limit_and_server_errors_raise(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    _install(monkeypatch, _Session({}, status_code))
    with pytest.raises(requests.HTTPError, match=f"biorxiv details request for DOI {_DOI} failed: {status_code}"):
        run_biorxiv_details(BiorxivDetailsInput(dois=_DOI))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.uses_env
def test_biorxiv_details_reports_versions_and_published_doi():
    """The RFdiffusion preprint has two versions and links to its Nature paper."""
    (preprint,) = run_biorxiv_details(BiorxivDetailsInput(dois=_DOI)).preprints
    assert [v.version for v in preprint.versions][:2] == [1, 2]
    assert preprint.versions[0].date == "2022-12-10"
    assert preprint.published_doi == "10.1038/s41586-023-06415-8"
    assert "Watson, J. L." in preprint.authors


@pytest.mark.network
@pytest.mark.uses_env
def test_biorxiv_details_auto_finds_medrxiv_and_reports_wrong_server():
    """Auto finds a medRxiv DOI; an explicit wrong server gives an item error."""
    (preprint,) = run_biorxiv_details(BiorxivDetailsInput(dois=_MEDRXIV_DOI)).preprints
    assert (preprint.error, preprint.server) == (None, "medRxiv")
    (missing,) = run_biorxiv_details(BiorxivDetailsInput(dois=_DOI), BiorxivDetailsConfig(server="medrxiv")).preprints
    assert missing.error is not None and "on medrxiv" in missing.error
