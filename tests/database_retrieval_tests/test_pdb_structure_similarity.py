"""Tests for RCSB learned structure-similarity search."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools import Structure
from proto_tools.tools.database_retrieval import (
    PdbStructureSimilarityConfig,
    PdbStructureSimilarityInput,
    run_pdb_structure_similarity,
)
from proto_tools.tools.database_retrieval.pdb import structure_similarity as module


class _Response:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if url == module._UPLOAD_URL:
            return _Response({"key": "synthetic/query.bcif"})
        return _Response(
            {
                "total_count": 2,
                "result_set": [
                    {
                        "identifier": "1YCR.A",
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
            }
        )

    def close(self) -> None:
        self.closed = True


def test_input_requires_exactly_one_source() -> None:
    fixture = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
    with pytest.raises(ValidationError, match="exactly one"):
        PdbStructureSimilarityInput()
    with pytest.raises(ValidationError, match="exactly one"):
        PdbStructureSimilarityInput(structure=Structure(structure=fixture), pdb_id="1YCR", chain_id="A")


def test_public_entry_search_returns_original_structure_score(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session()
    session_config: dict[str, Any] = {}

    def build_session(**kwargs: Any) -> _Session:
        session_config.update(kwargs)
        return session

    monkeypatch.setattr(module, "build_http_session", build_session)

    output = run_pdb_structure_similarity(
        PdbStructureSimilarityInput(pdb_id="1ycr", chain_id="A"),
        PdbStructureSimilarityConfig(number_of_candidates=25, predicted_tm_score_cutoff=0.4),
    )

    assert session.closed is True
    assert session_config["allowed_methods"] == ["POST"]
    assert [url for url, _kwargs in session.calls] == [module._SEARCH_URL]
    payload = session.calls[0][1]["json"]
    parameters = payload["query"]["parameters"]
    assert parameters["value"] == {"entry_id": "1YCR", "asym_id": "A"}
    assert parameters["ptmscore_cutoff"] == 0.4
    assert payload["request_options"]["results_content_type"] == ["experimental"]
    assert output.total_count == 2
    assert output.max_predicted_tm_score == pytest.approx(0.978926)
    assert output.hits[1].predicted_tm_score == pytest.approx(0.71)
    assert output.query_was_uploaded is False


def test_custom_structure_requires_external_upload_acknowledgement() -> None:
    fixture = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
    inputs = PdbStructureSimilarityInput(structure=fixture)
    with pytest.raises(ValueError, match="allow_external_upload=True"):
        run_pdb_structure_similarity(inputs, PdbStructureSimilarityConfig())


def test_acknowledged_custom_structure_is_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
    session = _Session()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: session)

    output = run_pdb_structure_similarity(
        PdbStructureSimilarityInput(structure=fixture),
        PdbStructureSimilarityConfig(allow_external_upload=True, target_search_space="assembly"),
    )

    assert session.closed is True
    assert [url for url, _kwargs in session.calls] == [module._UPLOAD_URL, module._SEARCH_URL]
    search_payload = session.calls[1][1]["json"]
    assert search_payload["query"]["parameters"]["value"] == {
        "url": module._UPLOAD_DOWNLOAD_BASE + "synthetic/query.bcif",
        "format": "bcif",
    }
    assert search_payload["return_type"] == "entry"
    assert output.query_was_uploaded is True
