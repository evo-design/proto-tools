"""Unit tests for AlphaGenome raw-sequence prediction dispatch."""

from __future__ import annotations

from typing import Any

from bio_programming_tools import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)
from bio_programming_tools.utils.tool_instance import ToolInstance


def test_single_sequence_dispatch(monkeypatch):
    sequence = "ACGT" * (16_384 // 4)
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"predictions": [{"signal": 1.23}]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(sequences=[sequence]),
        AlphaGenomePredictSequencesConfig(requested_outputs=["RNA_SEQ"], organism="human"),
    )

    assert len(calls) == 1
    assert calls[0]["payload"]["operation"] == "predict_sequences"
    assert calls[0]["payload"]["sequences"] == [sequence]
    assert len(result) == 1
    assert result[0].interval_end == len(sequence)
    assert result[0].result == {"predictions": {"signal": 1.23}}


def test_multi_sequence_dispatch(monkeypatch):
    sequence_a = "ACGT" * (16_384 // 4)
    sequence_b = "TGCA" * (16_384 // 4)
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"predictions": [{"signal": "a"}, {"signal": "b"}]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(sequences=[sequence_a, sequence_b]),
        AlphaGenomePredictSequencesConfig(requested_outputs=["RNA_SEQ"], organism="human"),
    )

    assert len(calls) == 1
    assert calls[0]["payload"]["operation"] == "predict_sequences"
    assert calls[0]["payload"]["sequences"] == [sequence_a, sequence_b]
    assert len(result) == 2
    assert result[0].result == {"predictions": {"signal": "a"}}
    assert result[1].result == {"predictions": {"signal": "b"}}


def test_single_string_auto_wraps(monkeypatch):
    """A single sequence string should auto-wrap into a list."""
    sequence = "ACGT" * (16_384 // 4)
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"predictions": [{"signal": 0.5}]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_predict_sequences(
        AlphaGenomePredictSequencesInput(sequences=sequence),
        AlphaGenomePredictSequencesConfig(requested_outputs=["RNA_SEQ"], organism="human"),
    )

    assert calls[0]["payload"]["sequences"] == [sequence]
    assert len(result) == 1
