"""Unit tests for AlphaGenome batched tool dispatch paths."""

from __future__ import annotations

from typing import Any

from bio_programming_tools import (
    AlphaGenomeISM,
    AlphaGenomeInterval,
    AlphaGenomePredictIntervalsConfig,
    AlphaGenomePredictIntervalsInput,
    AlphaGenomePredictVariantsConfig,
    AlphaGenomePredictVariantsInput,
    AlphaGenomeScoreISMConfig,
    AlphaGenomeScoreISMInput,
    AlphaGenomeScoreIntervalsConfig,
    AlphaGenomeScoreIntervalsInput,
    AlphaGenomeScoreVariantsConfig,
    AlphaGenomeScoreVariantsInput,
    AlphaGenomeVariant,
    run_alphagenome_predict_intervals,
    run_alphagenome_predict_variants,
    run_alphagenome_score_intervals,
    run_alphagenome_score_ism_variants_batch,
    run_alphagenome_score_variants,
)
from bio_programming_tools.utils.tool_instance import ToolInstance


def test_predict_intervals_dispatch(monkeypatch):
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"predictions": [{"signal": 1.0}, {"signal": 1.5}]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_predict_intervals(
        AlphaGenomePredictIntervalsInput(
            intervals=[
                AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=16_384),
                AlphaGenomeInterval(chromosome="chr2", interval_start=0, interval_end=16_384),
            ],
        ),
        AlphaGenomePredictIntervalsConfig(requested_outputs=["RNA_SEQ"], organism="human"),
    )

    assert calls[0]["payload"]["operation"] == "predict_intervals"
    assert calls[0]["payload"]["intervals"] == [
        {"chromosome": "chr1", "interval_start": 0, "interval_end": 16_384},
        {"chromosome": "chr2", "interval_start": 0, "interval_end": 16_384},
    ]
    assert len(result) == 2
    assert result[0].result == {"predictions": {"signal": 1.0}}
    assert result[1].result == {"predictions": {"signal": 1.5}}
    assert result[0].chromosome == "chr1"
    assert result[1].chromosome == "chr2"


def test_predict_variants_dispatch(monkeypatch):
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"predictions": [{"signal": 2.0}, {"signal": 2.5}]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_predict_variants(
        AlphaGenomePredictVariantsInput(
            variants=[
                AlphaGenomeVariant(
                    chromosome="chr2", interval_start=0, interval_end=16_384,
                    variant_position=1024, reference_bases="A", alternate_bases="G",
                ),
                AlphaGenomeVariant(
                    chromosome="chr3", interval_start=0, interval_end=16_384,
                    variant_position=2048, reference_bases="C", alternate_bases="T",
                ),
            ],
        ),
        AlphaGenomePredictVariantsConfig(requested_outputs=["RNA_SEQ"], organism="human"),
    )

    assert calls[0]["payload"]["operation"] == "predict_variants"
    assert len(calls[0]["payload"]["intervals"]) == 2
    assert len(calls[0]["payload"]["variants"]) == 2
    assert len(result) == 2
    assert result[0].result == {"predictions": {"signal": 2.0}}
    assert result[1].result == {"predictions": {"signal": 2.5}}
    assert result[0].variant["position"] == 1024
    assert result[1].variant["position"] == 2048


def test_score_intervals_dispatch(monkeypatch):
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"scores": [
            [{"track": "RNA_SEQ", "score": 0.7}],
            [{"track": "RNA_SEQ", "score": 0.9}],
        ]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_score_intervals(
        AlphaGenomeScoreIntervalsInput(
            intervals=[
                AlphaGenomeInterval(chromosome="chr3", interval_start=0, interval_end=524_288),
                AlphaGenomeInterval(chromosome="chr4", interval_start=0, interval_end=524_288),
            ],
        ),
        AlphaGenomeScoreIntervalsConfig(interval_scorers=["RNA_SEQ"], organism="human"),
    )

    assert calls[0]["payload"]["operation"] == "score_intervals"
    assert len(calls[0]["payload"]["intervals"]) == 2
    assert len(result) == 2
    assert result[0].scores == [{"track": "RNA_SEQ", "score": 0.7}]
    assert result[1].scores == [{"track": "RNA_SEQ", "score": 0.9}]


def test_score_variants_dispatch(monkeypatch):
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"scores": [
            [{"track": "ATAC", "score": 0.2}],
            [{"track": "ATAC", "score": 0.3}],
        ]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_score_variants(
        AlphaGenomeScoreVariantsInput(
            variants=[
                AlphaGenomeVariant(
                    chromosome="chr4", interval_start=0, interval_end=524_288,
                    variant_position=2048, reference_bases="C", alternate_bases="T",
                ),
                AlphaGenomeVariant(
                    chromosome="chr5", interval_start=0, interval_end=524_288,
                    variant_position=4096, reference_bases="G", alternate_bases="A",
                ),
            ],
        ),
        AlphaGenomeScoreVariantsConfig(variant_scorers=["ATAC"], organism="human"),
    )

    assert calls[0]["payload"]["operation"] == "score_variants"
    assert len(calls[0]["payload"]["intervals"]) == 2
    assert len(calls[0]["payload"]["variants"]) == 2
    assert len(result) == 2
    assert result[0].scores == [{"track": "ATAC", "score": 0.2}]
    assert result[1].scores == [{"track": "ATAC", "score": 0.3}]


def test_score_ism_batch_dispatch(monkeypatch):
    calls: list[dict[str, Any]] = []

    def mock_dispatch(_tool_name, payload, **kwargs):
        calls.append({"payload": payload, "kwargs": kwargs})
        return {"scores": [
            [{"track": "DNASE", "score": 0.4}],
            [{"track": "DNASE", "score": 0.6}],
        ]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(mock_dispatch))

    result = run_alphagenome_score_ism_variants_batch(
        AlphaGenomeScoreISMInput(
            requests=[
                AlphaGenomeISM(
                    chromosome="chr5", interval_start=0, interval_end=524_288,
                    ism_interval_start=100, ism_interval_end=110,
                ),
                AlphaGenomeISM(
                    chromosome="chr6", interval_start=0, interval_end=524_288,
                    ism_interval_start=200, ism_interval_end=210,
                ),
            ],
        ),
        AlphaGenomeScoreISMConfig(variant_scorers=["DNASE"], organism="human"),
    )

    assert calls[0]["payload"]["operation"] == "score_ism_variants_batch"
    assert len(calls[0]["payload"]["requests"]) == 2
    assert len(result) == 2
    assert result[0].scores == [{"track": "DNASE", "score": 0.4}]
    assert result[1].scores == [{"track": "DNASE", "score": 0.6}]
