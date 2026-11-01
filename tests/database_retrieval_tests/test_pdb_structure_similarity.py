"""Tests for RCSB learned structure-similarity search."""

from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from proto_tools import Structure
from proto_tools.tools.database_retrieval import (
    PdbStructureSimilarityBatchConfig,
    PdbStructureSimilarityBatchInput,
    PdbStructureSimilarityConfig,
    PdbStructureSimilarityInput,
    PdbStructureSimilarityOutput,
    run_pdb_structure_similarity,
    run_pdb_structure_similarity_batch,
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
    assert session_config["http_retries"] == 1
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


def test_exhausted_transport_retry_is_not_retried_by_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSession:
        def __init__(self) -> None:
            self.closed = False

        def post(self, *_args: Any, **_kwargs: Any) -> _Response:
            raise requests.ConnectTimeout("synthetic timeout")

        def close(self) -> None:
            self.closed = True

    session = FailingSession()
    seen: dict[str, Any] = {}

    def build_session(**kwargs: Any) -> FailingSession:
        seen.update(kwargs)
        return session

    monkeypatch.setattr(module, "build_http_session", build_session)
    with pytest.raises(RuntimeError, match="bounded HTTP retry budget"):
        run_pdb_structure_similarity(
            PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="A"),
            PdbStructureSimilarityConfig(timeout_seconds=45, http_retries=0),
        )

    assert seen["http_retries"] == 0
    assert session.closed is True


def test_custom_structure_is_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
    session = _Session()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: session)

    output = run_pdb_structure_similarity(
        PdbStructureSimilarityInput(structure=fixture),
        PdbStructureSimilarityConfig(target_search_space="assembly"),
    )

    assert session.closed is True
    assert [url for url, _kwargs in session.calls] == [module._UPLOAD_URL, module._SEARCH_URL]
    search_payload = session.calls[1][1]["json"]
    assert search_payload["query"]["parameters"]["value"] == {
        "url": module._UPLOAD_DOWNLOAD_BASE + "synthetic/query.bcif",
        "format": "bcif",
    }
    assert search_payload["return_type"] == "assembly"
    assert output.query_was_uploaded is True


def test_custom_chain_search_preserves_chain_level_return_type(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
    session = _Session()
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: session)

    run_pdb_structure_similarity(
        PdbStructureSimilarityInput(structure=fixture),
        PdbStructureSimilarityConfig(target_search_space="polymer_entity_instance"),
    )

    search_payload = session.calls[1][1]["json"]
    assert search_payload["return_type"] == "polymer_instance"


def test_batch_preserves_alignment_and_reports_partial_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def search(
        inputs: PdbStructureSimilarityInput,
        config: PdbStructureSimilarityConfig,
    ) -> PdbStructureSimilarityOutput:
        if inputs.pdb_id == "FAIL":
            raise RuntimeError("synthetic unavailable query")
        return PdbStructureSimilarityOutput(
            max_predicted_tm_score=0.75 if inputs.chain_id == "A" else 0.25,
            total_count=1,
            source_url=module._SEARCH_URL,
        )

    monkeypatch.setattr(module, "run_pdb_structure_similarity", search)
    output = run_pdb_structure_similarity_batch(
        PdbStructureSimilarityBatchInput(
            query_ids=["first", "broken", "third"],
            queries=[
                PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="A"),
                PdbStructureSimilarityInput(pdb_id="FAIL", chain_id="A"),
                PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="B"),
            ],
        ),
        PdbStructureSimilarityBatchConfig(max_workers=2),
    )

    assert [result.query_id for result in output.results] == ["first", "broken", "third"]
    assert [result.input_index for result in output.results] == [0, 1, 2]
    assert output.results[0].output is not None
    assert output.results[0].output.max_predicted_tm_score == pytest.approx(0.75)
    assert output.results[1].output is None
    assert output.results[1].error_type == "RuntimeError"
    assert "synthetic unavailable" in (output.results[1].error_message or "")
    assert output.results[2].output is not None
    assert output.results[2].output.max_predicted_tm_score == pytest.approx(0.25)
    assert output.successful_count == 2
    assert output.failed_count == 1
    assert output.metadata["failed_queries_are_not_measurements"] is True


@pytest.mark.parametrize("fail_fast", [False, True])
def test_batch_unexpected_output_respects_failure_policy(monkeypatch: pytest.MonkeyPatch, fail_fast: bool) -> None:
    def invalid_output(*_args: Any, **_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(module, "run_pdb_structure_similarity", invalid_output)
    inputs = PdbStructureSimilarityBatchInput(
        queries=[PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="A")],
    )
    config = PdbStructureSimilarityBatchConfig(fail_fast=fail_fast)
    if fail_fast:
        with pytest.raises(TypeError, match="unexpected output type"):
            run_pdb_structure_similarity_batch(inputs, config)
    else:
        output = run_pdb_structure_similarity_batch(inputs, config)
        assert output.successful_count == 0
        assert output.failed_count == 1
        assert output.results[0].output is None
        assert output.results[0].error_type == "TypeError"


def test_batch_fail_fast_propagates_query_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(
        inputs: PdbStructureSimilarityInput,
        config: PdbStructureSimilarityConfig,
    ) -> PdbStructureSimilarityOutput:
        raise RuntimeError("stop now")

    monkeypatch.setattr(module, "run_pdb_structure_similarity", fail)
    with pytest.raises(RuntimeError, match="stop now"):
        run_pdb_structure_similarity_batch(
            PdbStructureSimilarityBatchInput(queries=[PdbStructureSimilarityInput(pdb_id="1YCR", chain_id="A")]),
            PdbStructureSimilarityBatchConfig(fail_fast=True),
        )
