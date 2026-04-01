"""tests/sequence_scoring_tests/test_alphagenome_batch_paths.py.

Tests for AlphaGenome batch dispatch paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from proto_tools import (
    AlphaGenomeInterval,
    AlphaGenomeISM,
    AlphaGenomePredictIntervalsConfig,
    AlphaGenomePredictIntervalsInput,
    AlphaGenomePredictVariantsConfig,
    AlphaGenomePredictVariantsInput,
    AlphaGenomeScoreIntervalsConfig,
    AlphaGenomeScoreIntervalsInput,
    AlphaGenomeScoreISMConfig,
    AlphaGenomeScoreISMInput,
    AlphaGenomeScoreVariantsConfig,
    AlphaGenomeScoreVariantsInput,
    AlphaGenomeVariant,
    run_alphagenome_predict_intervals,
    run_alphagenome_predict_variants,
    run_alphagenome_score_intervals,
    run_alphagenome_score_ism_variants_batch,
    run_alphagenome_score_variants,
)
from proto_tools.utils.tool_instance import ToolInstance

# ── Module-level constants ────────────────────────────────────────────────────

_PREDICT_LENGTH = 16_384
_SCORE_LENGTH = 524_288
_ORGANISM = "human"


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fake_hf_token(monkeypatch):
    """Bypass require_hf_token() for dispatch-path tests (no real HF access needed)."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_for_unit_tests")


@pytest.fixture
def mock_dispatch_factory(monkeypatch):
    """Return a factory that installs a mock ToolInstance.dispatch and records calls."""

    def _install(return_value: dict[str, Any]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def _mock(_tool_name, payload, **kwargs):
            calls.append({"payload": payload, "kwargs": kwargs})
            return return_value

        monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(_mock))
        return calls

    return _install


# ── Dispatch path tests ───────────────────────────────────────────────────────


def test_predict_intervals_dispatch(mock_dispatch_factory):
    calls = mock_dispatch_factory({"predictions": [{"signal": 1.0}, {"signal": 1.5}]})

    result = run_alphagenome_predict_intervals(
        AlphaGenomePredictIntervalsInput(
            intervals=[
                AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=_PREDICT_LENGTH),
                AlphaGenomeInterval(chromosome="chr2", interval_start=0, interval_end=_PREDICT_LENGTH),
            ],
        ),
        AlphaGenomePredictIntervalsConfig(requested_outputs=["RNA_SEQ"], organism=_ORGANISM),
    )

    payload = calls[0]["payload"]
    assert payload["operation"] == "predict_intervals"
    assert payload["intervals"] == [
        {"chromosome": "chr1", "interval_start": 0, "interval_end": _PREDICT_LENGTH},
        {"chromosome": "chr2", "interval_start": 0, "interval_end": _PREDICT_LENGTH},
    ]

    assert len(result) == 2
    assert result[0].result == {"predictions": {"signal": 1.0}}
    assert result[1].result == {"predictions": {"signal": 1.5}}
    assert result[0].chromosome == "chr1"
    assert result[0].interval_start == 0
    assert result[0].interval_end == _PREDICT_LENGTH
    assert result[1].chromosome == "chr2"


def test_predict_variants_dispatch(mock_dispatch_factory):
    calls = mock_dispatch_factory({"predictions": [{"signal": 2.0}, {"signal": 2.5}]})

    result = run_alphagenome_predict_variants(
        AlphaGenomePredictVariantsInput(
            variants=[
                AlphaGenomeVariant(
                    chromosome="chr2",
                    interval_start=0,
                    interval_end=_PREDICT_LENGTH,
                    variant_position=1024,
                    reference_bases="A",
                    alternate_bases="G",
                ),
                AlphaGenomeVariant(
                    chromosome="chr3",
                    interval_start=0,
                    interval_end=_PREDICT_LENGTH,
                    variant_position=2048,
                    reference_bases="C",
                    alternate_bases="T",
                ),
            ],
        ),
        AlphaGenomePredictVariantsConfig(requested_outputs=["RNA_SEQ"], organism=_ORGANISM),
    )

    payload = calls[0]["payload"]
    assert payload["operation"] == "predict_variants"
    assert len(payload["intervals"]) == 2
    assert len(payload["variants"]) == 2

    assert len(result) == 2
    assert result[0].result == {"predictions": {"signal": 2.0}}
    assert result[1].result == {"predictions": {"signal": 2.5}}
    assert result[0].variant["position"] == 1024
    assert result[0].variant["reference_bases"] == "A"
    assert result[0].variant["alternate_bases"] == "G"
    assert result[1].variant["position"] == 2048


def test_score_intervals_dispatch(mock_dispatch_factory):
    calls = mock_dispatch_factory(
        {
            "scores": [
                [{"track": "RNA_SEQ", "score": 0.7}],
                [{"track": "RNA_SEQ", "score": 0.9}],
            ]
        }
    )

    result = run_alphagenome_score_intervals(
        AlphaGenomeScoreIntervalsInput(
            intervals=[
                AlphaGenomeInterval(chromosome="chr3", interval_start=0, interval_end=_SCORE_LENGTH),
                AlphaGenomeInterval(chromosome="chr4", interval_start=0, interval_end=_SCORE_LENGTH),
            ],
        ),
        AlphaGenomeScoreIntervalsConfig(interval_scorers=["RNA_SEQ"], organism=_ORGANISM),
    )

    payload = calls[0]["payload"]
    assert payload["operation"] == "score_intervals"
    assert len(payload["intervals"]) == 2

    assert len(result) == 2
    assert result[0].scores == [{"track": "RNA_SEQ", "score": 0.7}]
    assert result[1].scores == [{"track": "RNA_SEQ", "score": 0.9}]


def test_score_variants_dispatch(mock_dispatch_factory):
    calls = mock_dispatch_factory(
        {
            "scores": [
                [{"track": "ATAC", "score": 0.2}],
                [{"track": "ATAC", "score": 0.3}],
            ]
        }
    )

    result = run_alphagenome_score_variants(
        AlphaGenomeScoreVariantsInput(
            variants=[
                AlphaGenomeVariant(
                    chromosome="chr4",
                    interval_start=0,
                    interval_end=_SCORE_LENGTH,
                    variant_position=2048,
                    reference_bases="C",
                    alternate_bases="T",
                ),
                AlphaGenomeVariant(
                    chromosome="chr5",
                    interval_start=0,
                    interval_end=_SCORE_LENGTH,
                    variant_position=4096,
                    reference_bases="G",
                    alternate_bases="A",
                ),
            ],
        ),
        AlphaGenomeScoreVariantsConfig(variant_scorers=["ATAC"], organism=_ORGANISM),
    )

    payload = calls[0]["payload"]
    assert payload["operation"] == "score_variants"
    assert len(payload["intervals"]) == 2
    assert len(payload["variants"]) == 2

    assert len(result) == 2
    assert result[0].scores == [{"track": "ATAC", "score": 0.2}]
    assert result[1].scores == [{"track": "ATAC", "score": 0.3}]


def test_score_ism_batch_dispatch(mock_dispatch_factory):
    calls = mock_dispatch_factory(
        {
            "scores": [
                [{"track": "DNASE", "score": 0.4}],
                [{"track": "DNASE", "score": 0.6}],
            ]
        }
    )

    result = run_alphagenome_score_ism_variants_batch(
        AlphaGenomeScoreISMInput(
            requests=[
                AlphaGenomeISM(
                    chromosome="chr5",
                    interval_start=0,
                    interval_end=_SCORE_LENGTH,
                    ism_interval_start=100,
                    ism_interval_end=110,
                ),
                AlphaGenomeISM(
                    chromosome="chr6",
                    interval_start=0,
                    interval_end=_SCORE_LENGTH,
                    ism_interval_start=200,
                    ism_interval_end=210,
                ),
            ],
        ),
        AlphaGenomeScoreISMConfig(variant_scorers=["DNASE"], organism=_ORGANISM),
    )

    payload = calls[0]["payload"]
    assert payload["operation"] == "score_ism_variants_batch"
    assert len(payload["requests"]) == 2

    assert len(result) == 2
    assert result[0].scores == [{"track": "DNASE", "score": 0.4}]
    assert result[1].scores == [{"track": "DNASE", "score": 0.6}]
