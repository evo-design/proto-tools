"""Tests for RCSB learned structure-similarity search."""

from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    PdbStructureSimilarityConfig,
    PdbStructureSimilarityInput,
    PdbStructureSimilarityQuery,
    run_pdb_structure_similarity,
)
from proto_tools.tools.database_retrieval.pdb import structure_similarity as module

_FIXTURE = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = "synthetic body"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    """Fake RCSB session: echoes the queried chain into the top hit's identifier."""

    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.status_code = status_code

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if url == module._UPLOAD_URL:
            return _Response({"key": "synthetic/query.bcif"})
        chain = kwargs["json"]["query"]["parameters"]["value"].get("asym_id", "X")
        return _Response(
            {
                "total_count": 7,
                "result_set": [
                    {
                        "identifier": f"1YCR.{chain}",
                        "score": 1.0,
                        "services": [
                            {
                                "service_type": "structure",
                                "nodes": [{"node_id": 0, "original_score": 0.978926}],
                            }
                        ],
                    },
                    {"identifier": "4ABC.B", "score": 0.8, "original_score": 0.71},
                ],
            },
            status_code=self.status_code,
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _Session:
    fake = _Session()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: fake)
    return fake


def test_query_requires_exactly_one_source() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        PdbStructureSimilarityQuery()
    with pytest.raises(ValidationError, match="exactly one"):
        PdbStructureSimilarityQuery(structure=_FIXTURE, pdb_id="1YCR", chain_id="A")


def test_public_query_requires_chain_or_assembly() -> None:
    with pytest.raises(ValidationError, match="requires chain_id or assembly_id"):
        PdbStructureSimilarityQuery(pdb_id="1YCR")
    with pytest.raises(ValidationError, match="at most one"):
        PdbStructureSimilarityQuery(pdb_id="1YCR", chain_id="A", assembly_id="1")


def test_single_query_is_normalized_to_list() -> None:
    inputs = PdbStructureSimilarityInput(queries={"pdb_id": "1YCR", "chain_id": "A"})
    assert [query.pdb_id for query in inputs.queries] == ["1YCR"]


def test_public_query_payload_and_scores(session: _Session) -> None:
    output = run_pdb_structure_similarity(
        PdbStructureSimilarityInput(queries=[PdbStructureSimilarityQuery(pdb_id="1ycr", chain_id="A")]),
        PdbStructureSimilarityConfig(number_of_candidates=25, predicted_tm_score_cutoff=0.4),
    )

    assert session.closed is True
    assert [url for url, _kwargs in session.calls] == [module._SEARCH_URL]
    payload = session.calls[0][1]["json"]
    assert payload["query"]["parameters"]["value"] == {"entry_id": "1YCR", "asym_id": "A"}
    assert payload["query"]["parameters"]["ptmscore_cutoff"] == 0.4
    assert payload["request_options"]["paginate"]["rows"] == 25
    assert payload["request_options"]["results_content_type"] == ["experimental"]
    assert "request_info" not in payload
    (result,) = output.results
    assert result.total_count == 7
    assert result.hits[0].predicted_tm_score == pytest.approx(0.978926)
    assert result.hits[1].predicted_tm_score == pytest.approx(0.71)
    assert output.source_url == module._SEARCH_URL


def test_queries_share_one_session_and_keep_input_order(session: _Session) -> None:
    output = run_pdb_structure_similarity(
        PdbStructureSimilarityInput(
            queries=[
                PdbStructureSimilarityQuery(pdb_id="1YCR", chain_id="B"),
                PdbStructureSimilarityQuery(pdb_id="1YCR", chain_id="A"),
            ]
        ),
        PdbStructureSimilarityConfig(),
    )

    assert [result.hits[0].identifier for result in output.results] == ["1YCR.B", "1YCR.A"]
    assert len(session.calls) == 2
    assert session.closed is True


def test_custom_structure_is_uploaded_before_search(session: _Session) -> None:
    run_pdb_structure_similarity(
        PdbStructureSimilarityInput(queries=[PdbStructureSimilarityQuery(structure=_FIXTURE)]),
        PdbStructureSimilarityConfig(target_search_space="assembly"),
    )

    assert [url for url, _kwargs in session.calls] == [module._UPLOAD_URL, module._SEARCH_URL]
    search_payload = session.calls[1][1]["json"]
    assert search_payload["query"]["parameters"]["value"] == {
        "url": module._UPLOAD_DOWNLOAD_BASE + "synthetic/query.bcif",
        "format": "bcif",
    }
    assert search_payload["return_type"] == "assembly"


def test_custom_chain_search_returns_chain_level_hits(session: _Session) -> None:
    run_pdb_structure_similarity(
        PdbStructureSimilarityInput(queries=[PdbStructureSimilarityQuery(structure=_FIXTURE)]),
        PdbStructureSimilarityConfig(target_search_space="polymer_entity_instance"),
    )

    assert session.calls[1][1]["json"]["return_type"] == "polymer_instance"


def test_http_error_propagates_with_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = _Session(status_code=503)
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: failing)

    with pytest.raises(requests.HTTPError, match=r"RCSB structure search failed.*synthetic body"):
        run_pdb_structure_similarity(
            PdbStructureSimilarityInput(queries=[PdbStructureSimilarityQuery(pdb_id="1YCR", chain_id="A")]),
            PdbStructureSimilarityConfig(),
        )
    assert failing.closed is True
