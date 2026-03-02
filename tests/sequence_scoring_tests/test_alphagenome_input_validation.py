"""Validation tests for AlphaGenome input models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bio_programming_tools import (
    AlphaGenomeInterval,
    AlphaGenomePredictIntervalsInput,
    AlphaGenomePredictSequencesInput,
    AlphaGenomePredictVariantsInput,
    AlphaGenomeScoreIntervalsInput,
    AlphaGenomeScoreVariantsInput,
    AlphaGenomeVariant,
)


# ============================================================================
# Sequence validation
# ============================================================================


def test_predict_sequences_accepts_single_supported_min_length():
    sequence = "ACGT" * (16_384 // 4)

    parsed = AlphaGenomePredictSequencesInput(sequences=[sequence])

    assert len(parsed.sequences) == 1
    assert len(parsed.sequences[0]) == 16_384


def test_predict_sequences_rejects_unsupported_length():
    sequence = "ACGT" * (2_048 // 4)

    with pytest.raises(
        ValidationError,
        match="supported AlphaGenome context length",
    ):
        AlphaGenomePredictSequencesInput(sequences=[sequence])


def test_predict_sequences_accepts_multiple_supported_lengths():
    inputs = AlphaGenomePredictSequencesInput(
        sequences=[
            "ACGT" * (16_384 // 4),
            "TGCA" * (16_384 // 4),
        ]
    )
    assert len(inputs.sequences) == 2
    assert all(len(sequence) == 16_384 for sequence in inputs.sequences)


def test_predict_sequences_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomePredictSequencesInput(sequences=[])


def test_predict_sequences_auto_wraps_single_string():
    """A single string should auto-wrap into a list."""
    sequence = "ACGT" * (16_384 // 4)
    inputs = AlphaGenomePredictSequencesInput(sequences=sequence)
    assert len(inputs.sequences) == 1
    assert len(inputs.sequences[0]) == 16_384


# ============================================================================
# Interval auto-wrap validation
# ============================================================================


def test_predict_intervals_auto_wraps_single_interval():
    """A single AlphaGenomeInterval dict should auto-wrap into a list."""
    inputs = AlphaGenomePredictIntervalsInput(
        intervals={"chromosome": "chr1", "interval_start": 0, "interval_end": 16_384},
    )
    assert len(inputs.intervals) == 1
    assert inputs.intervals[0].chromosome == "chr1"


def test_predict_intervals_accepts_list():
    inputs = AlphaGenomePredictIntervalsInput(
        intervals=[
            AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=16_384),
            AlphaGenomeInterval(chromosome="chr2", interval_start=0, interval_end=16_384),
        ],
    )
    assert len(inputs.intervals) == 2


def test_predict_intervals_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomePredictIntervalsInput(intervals=[])


def test_predict_intervals_rejects_none():
    with pytest.raises(ValidationError, match="cannot be None"):
        AlphaGenomePredictIntervalsInput(intervals=None)


def test_score_intervals_auto_wraps_single_interval():
    inputs = AlphaGenomeScoreIntervalsInput(
        intervals=AlphaGenomeInterval(chromosome="chr3", interval_start=0, interval_end=524_288),
    )
    assert len(inputs.intervals) == 1


# ============================================================================
# Variant auto-wrap validation
# ============================================================================


def test_predict_variants_auto_wraps_single_variant():
    """A single AlphaGenomeVariant should auto-wrap into a list."""
    inputs = AlphaGenomePredictVariantsInput(
        variants=AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=1024,
            reference_bases="A",
            alternate_bases="G",
        ),
    )
    assert len(inputs.variants) == 1
    assert inputs.variants[0].variant_position == 1024


def test_predict_variants_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomePredictVariantsInput(variants=[])


def test_score_variants_auto_wraps_single_variant():
    inputs = AlphaGenomeScoreVariantsInput(
        variants=AlphaGenomeVariant(
            chromosome="chr4",
            interval_start=0,
            interval_end=524_288,
            variant_position=2048,
            reference_bases="C",
            alternate_bases="T",
        ),
    )
    assert len(inputs.variants) == 1
