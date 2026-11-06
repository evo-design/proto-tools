"""Tests for RCSB MMseqs2 sequence search over PDB polymer entities."""

import csv
import json
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    PdbSequenceSearchConfig,
    PdbSequenceSearchInput,
    PdbSequenceSearchQuery,
    run_pdb_sequence_search,
)
from proto_tools.tools.database_retrieval.pdb import sequence_search as module

_QUERY = "VRGSPAINVAVHVFRKAADDTWEPFASGKT"


def _search_item(identifier: str, score: float, identity: float) -> dict[str, Any]:
    return {
        "identifier": identifier,
        "score": score,
        "services": [
            {
                "service_type": "sequence",
                "nodes": [
                    {
                        "node_id": 1,
                        "match_context": [
                            {
                                "sequence_identity": identity,
                                "evalue": 3.4e-13,
                                "bitscore": 65,
                                "alignment_length": 30,
                                "mismatches": 0,
                                "gaps_opened": 0,
                                "query_beg": 1,
                                "query_end": 30,
                                "subject_beg": 20,
                                "subject_end": 49,
                                "query_length": 30,
                                "subject_length": 127,
                                "query_aligned_seq": _QUERY,
                                "subject_aligned_seq": _QUERY,
                            }
                        ],
                    }
                ],
            }
        ],
    }


_SEARCH_PAYLOAD = {
    "total_count": 496,
    "result_set": [_search_item("1BM7_1", 1.0, 1.0), _search_item("4ZQK_1", 0.5, 0.4)],
}

# RCSB's GraphQL endpoint does not return entities in request order.
_GRAPHQL_PAYLOAD = {
    "data": {
        "polymer_entities": [
            {
                "rcsb_id": "4ZQK_1",
                "rcsb_polymer_entity": {"pdbx_description": "Programmed cell death 1 ligand 1"},
                "rcsb_entity_source_organism": [{"ncbi_scientific_name": "Homo sapiens"}],
                "rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["Q9NZQ7"], "auth_asym_ids": ["A"]},
                "entity_poly": {"pdbx_seq_one_letter_code_can": "AFTVTVPK"},
            },
            {
                "rcsb_id": "1BM7_1",
                "rcsb_polymer_entity": {"pdbx_description": "PROTEIN (TRANSTHYRETIN)"},
                "rcsb_entity_source_organism": [{"ncbi_scientific_name": "Homo sapiens"}],
                "rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["P02766"], "auth_asym_ids": ["A", "B"]},
                "entity_poly": {"pdbx_seq_one_letter_code_can": "GPTGTGESKCPLM"},
            },
        ]
    }
}


class _Response:
    def __init__(self, payload: dict[str, Any] | None, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = "synthetic body"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self) -> dict[str, Any]:
        if self.payload is None:
            raise ValueError("no JSON body")
        return self.payload


class _Session:
    """Fake RCSB session that records calls and serves search and GraphQL responses."""

    def __init__(self, search_status: int = 200, resets: int = 0) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.search_status = search_status
        self.resets = resets

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if self.resets:
            self.resets -= 1
            raise requests.exceptions.ConnectionError("Connection reset by peer")
        if url == module._GRAPHQL_URL:
            return _Response(_GRAPHQL_PAYLOAD)
        if self.search_status == 204:
            return _Response(None, status_code=204)
        return _Response(_SEARCH_PAYLOAD, status_code=self.search_status)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _Session:
    fake = _Session()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: fake)
    return fake


# ── Input validation ─────────────────────────────────────────────────────────


def test_query_is_cleaned_and_short_queries_rejected() -> None:
    assert PdbSequenceSearchQuery(sequence=" vrgspa inva\nvhvf ").sequence == "VRGSPAINVAVHVF"
    with pytest.raises(ValidationError, match="at least 10 residues"):
        PdbSequenceSearchQuery(sequence="MKV")


def test_bare_strings_and_single_queries_are_normalized() -> None:
    assert [q.sequence for q in PdbSequenceSearchInput(queries=_QUERY).queries] == [_QUERY]
    inputs = PdbSequenceSearchInput(queries=[_QUERY, {"sequence": "ACGTACGTACGT", "sequence_type": "dna"}])
    assert [q.sequence_type for q in inputs.queries] == ["protein", "dna"]


# ── Search and annotation ────────────────────────────────────────────────────


def test_payload_parsing_and_annotation_joined_by_id(session: _Session) -> None:
    output = run_pdb_sequence_search(
        PdbSequenceSearchInput(queries=_QUERY),
        PdbSequenceSearchConfig(identity_cutoff=0.3, evalue_cutoff=1.0, max_hits=2),
    )

    assert session.closed is True
    assert [url for url, _kwargs in session.calls] == [module._SEARCH_URL, module._GRAPHQL_URL]
    payload = session.calls[0][1]["json"]
    assert payload["query"]["service"] == "sequence"
    assert payload["query"]["parameters"] == {
        "value": _QUERY,
        "sequence_type": "protein",
        "identity_cutoff": 0.3,
        "evalue_cutoff": 1.0,
    }
    assert payload["return_type"] == "polymer_entity"
    assert payload["request_options"]["paginate"]["rows"] == 2
    assert payload["request_options"]["results_content_type"] == ["experimental"]
    assert session.calls[1][1]["json"]["variables"] == {"ids": ["1BM7_1", "4ZQK_1"]}

    (result,) = output.results
    assert result.total_count == 496
    top, second = result.hits
    assert (top.entity_id, top.pdb_id, top.sequence_identity) == ("1BM7_1", "1BM7", 1.0)
    assert (top.subject_beg, top.subject_end) == (20, 49)
    assert top.uniprot_ids == ["P02766"] and top.chain_ids == ["A", "B"]
    assert second.description == "Programmed cell death 1 ligand 1" and second.uniprot_ids == ["Q9NZQ7"]
    assert output.source_url == module._SEARCH_URL


def test_annotation_off_makes_one_request(session: _Session) -> None:
    output = run_pdb_sequence_search(PdbSequenceSearchInput(queries=_QUERY), PdbSequenceSearchConfig(annotate=False))
    assert [url for url, _kwargs in session.calls] == [module._SEARCH_URL]
    assert output.results[0].hits[0].description is None and output.results[0].hits[0].uniprot_ids is None


def test_computed_models_widen_content_type(session: _Session) -> None:
    run_pdb_sequence_search(
        PdbSequenceSearchInput(queries=_QUERY), PdbSequenceSearchConfig(include_computed_models=True)
    )
    assert session.calls[0][1]["json"]["request_options"]["results_content_type"] == ["experimental", "computational"]


def test_no_content_is_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = _Session(search_status=204)
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: empty)
    output = run_pdb_sequence_search(PdbSequenceSearchInput(queries=_QUERY), PdbSequenceSearchConfig())
    assert output.results[0].hits == [] and output.results[0].total_count == 0
    assert [url for url, _kwargs in empty.calls] == [module._SEARCH_URL]  # nothing to annotate


def test_connection_reset_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    flaky = _Session(resets=2)
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: flaky)
    monkeypatch.setattr(module, "_BACKOFF_SECONDS", 0.0)
    output = run_pdb_sequence_search(PdbSequenceSearchInput(queries=_QUERY), PdbSequenceSearchConfig(annotate=False))
    assert [url for url, _kwargs in flaky.calls] == [module._SEARCH_URL] * 3
    assert output.results[0].total_count == 496


def test_http_error_propagates_with_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = _Session(search_status=500)
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: failing)
    with pytest.raises(requests.HTTPError, match=r"RCSB sequence search failed.*synthetic body"):
        run_pdb_sequence_search(PdbSequenceSearchInput(queries=_QUERY), PdbSequenceSearchConfig())
    assert failing.closed is True


# ── Export ───────────────────────────────────────────────────────────────────


def test_csv_and_fasta_export(session: _Session, tmp_path: Any) -> None:
    output = run_pdb_sequence_search(PdbSequenceSearchInput(queries=_QUERY), PdbSequenceSearchConfig())
    output._export_output(tmp_path / "hits", "csv")
    rows = list(csv.DictReader((tmp_path / "hits.csv").open()))
    assert [row["entity_id"] for row in rows] == ["1BM7_1", "4ZQK_1"]
    assert rows[0]["chain_ids"] == "A;B" and rows[0]["query_index"] == "0"
    output._export_output(tmp_path / "hits", "fasta")
    assert (tmp_path / "hits.fasta").read_text().startswith(">1BM7_1 query=0 PROTEIN (TRANSTHYRETIN)\nGPTGTGESKCPLM\n")
    output._export_output(tmp_path / "hits", "json")
    assert json.loads((tmp_path / "hits.json").read_text())["results"][0]["total_count"] == 496


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pdb_sequence_search_finds_transthyretin_from_a_mutated_fragment():
    """A 30-residue transthyretin fragment with three substitutions still ranks transthyretin first."""
    output = run_pdb_sequence_search(
        PdbSequenceSearchInput(queries="VRGSPAVNVAVHIFRKAADDTWEPFASGRT"),
        PdbSequenceSearchConfig(identity_cutoff=0.5, evalue_cutoff=1.0, max_hits=5),
    )
    top = output.results[0].hits[0]
    assert "P02766" in (top.uniprot_ids or [])
    assert top.sequence_identity is not None and 0.8 <= top.sequence_identity < 1.0
    assert top.sequence and top.chain_ids
