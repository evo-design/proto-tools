"""Tests for AlphaGenome."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bio_programming_tools import (
    AlphaGenomeISM,
    AlphaGenomeInterval,
    AlphaGenomePredictIntervalsInput,
    AlphaGenomePredictSequencesInput,
    AlphaGenomePredictVariantsInput,
    AlphaGenomeScoreISMInput,
    AlphaGenomeScoreIntervalsInput,
    AlphaGenomeScoreVariantsInput,
    AlphaGenomeVariant,
)

# ── AlphaGenomeInterval ──────────────────────────────────────────────────────


def test_interval_rejects_end_before_start():
    with pytest.raises(ValidationError, match="end must be greater than start"):
        AlphaGenomeInterval(chromosome="chr1", interval_start=100, interval_end=50)


def test_interval_rejects_end_equal_to_start():
    with pytest.raises(ValidationError, match="end must be greater than start"):
        AlphaGenomeInterval(chromosome="chr1", interval_start=100, interval_end=100)


def test_interval_rejects_negative_start():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AlphaGenomeInterval(chromosome="chr1", interval_start=-1, interval_end=100)


# ── AlphaGenomeVariant ───────────────────────────────────────────────────────


def test_variant_rejects_empty_reference_bases():
    with pytest.raises(ValidationError, match="Allele values cannot be empty"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=100,
            reference_bases="   ",
            alternate_bases="G",
        )


def test_variant_rejects_empty_alternate_bases():
    with pytest.raises(ValidationError, match="Allele values cannot be empty"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=100,
            reference_bases="A",
            alternate_bases="",
        )


def test_variant_rejects_invalid_reference_bases():
    with pytest.raises(ValidationError, match="Allele values must only contain DNA bases"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=100,
            reference_bases="X",
            alternate_bases="G",
        )


def test_variant_rejects_invalid_alternate_bases():
    with pytest.raises(ValidationError, match="Allele values must only contain DNA bases"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=100,
            reference_bases="A",
            alternate_bases="Z",
        )


def test_variant_normalizes_alleles_to_uppercase():
    variant = AlphaGenomeVariant(
        chromosome="chr1",
        interval_start=0,
        interval_end=16_384,
        variant_position=100,
        reference_bases="a",
        alternate_bases="gct",
    )
    assert variant.reference_bases == "A"
    assert variant.alternate_bases == "GCT"


def test_variant_rejects_position_before_interval_start():
    with pytest.raises(ValidationError, match="variant_position must be within"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=1000,
            interval_end=16_384,
            variant_position=500,
            reference_bases="A",
            alternate_bases="G",
        )


def test_variant_rejects_position_at_interval_end():
    with pytest.raises(ValidationError, match="variant_position must be within"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=16_384,
            reference_bases="A",
            alternate_bases="G",
        )


def test_variant_rejects_position_after_interval_end():
    with pytest.raises(ValidationError, match="variant_position must be within"):
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=16_384,
            variant_position=99_999,
            reference_bases="A",
            alternate_bases="G",
        )


# ── AlphaGenomePredictSequencesInput ─────────────────────────────────────────


def test_predict_sequences_accepts_single_supported_min_length():
    inputs = AlphaGenomePredictSequencesInput(sequences=["ACGT" * (16_384 // 4)])
    assert len(inputs.sequences) == 1


def test_predict_sequences_rejects_unsupported_length():
    with pytest.raises(ValidationError, match="supported AlphaGenome context length"):
        AlphaGenomePredictSequencesInput(sequences=["ACGT" * (2_048 // 4)])


def test_predict_sequences_rejects_invalid_characters():
    with pytest.raises(ValidationError, match="sequence must only contain DNA bases"):
        AlphaGenomePredictSequencesInput(sequences=["X" * 16_384])


def test_predict_sequences_accepts_multiple_supported_lengths():
    inputs = AlphaGenomePredictSequencesInput(
        sequences=[
            "ACGT" * (16_384 // 4),
            "TGCA" * (16_384 // 4),
        ]
    )
    assert len(inputs.sequences) == 2


def test_predict_sequences_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomePredictSequencesInput(sequences=[])


def test_predict_sequences_auto_wraps_single_string():
    inputs = AlphaGenomePredictSequencesInput(sequences="ACGT" * (16_384 // 4))
    assert len(inputs.sequences) == 1


def test_predict_sequences_normalizes_to_uppercase():
    inputs = AlphaGenomePredictSequencesInput(sequences=["acgt" * (16_384 // 4)])
    assert inputs.sequences[0] == inputs.sequences[0].upper()


# ── AlphaGenomePredictIntervalsInput ─────────────────────────────────────────


def test_predict_intervals_auto_wraps_single_interval_object():
    inputs = AlphaGenomePredictIntervalsInput(
        intervals=AlphaGenomeInterval(chromosome="chr1", interval_start=0, interval_end=16_384),
    )
    assert len(inputs.intervals) == 1
    assert inputs.intervals[0].chromosome == "chr1"


def test_predict_intervals_auto_wraps_single_interval_dict():
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


# ── AlphaGenomeScoreIntervalsInput ───────────────────────────────────────────


def test_score_intervals_auto_wraps_single_interval():
    inputs = AlphaGenomeScoreIntervalsInput(
        intervals=AlphaGenomeInterval(chromosome="chr3", interval_start=0, interval_end=524_288),
    )
    assert len(inputs.intervals) == 1


def test_score_intervals_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomeScoreIntervalsInput(intervals=[])


def test_score_intervals_rejects_none():
    with pytest.raises(ValidationError, match="cannot be None"):
        AlphaGenomeScoreIntervalsInput(intervals=None)


# ── AlphaGenomePredictVariantsInput ──────────────────────────────────────────

_VALID_VARIANT = dict(
    chromosome="chr1",
    interval_start=0,
    interval_end=16_384,
    variant_position=1024,
    reference_bases="A",
    alternate_bases="G",
)


def test_predict_variants_auto_wraps_single_variant():
    inputs = AlphaGenomePredictVariantsInput(
        variants=AlphaGenomeVariant(**_VALID_VARIANT),
    )
    assert len(inputs.variants) == 1
    assert inputs.variants[0].variant_position == 1024


def test_predict_variants_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomePredictVariantsInput(variants=[])


def test_predict_variants_rejects_none():
    with pytest.raises(ValidationError, match="cannot be None"):
        AlphaGenomePredictVariantsInput(variants=None)


# ── AlphaGenomeScoreVariantsInput ────────────────────────────────────────────


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


def test_score_variants_rejects_empty_list():
    with pytest.raises(ValidationError, match="cannot be empty"):
        AlphaGenomeScoreVariantsInput(variants=[])


def test_score_variants_rejects_none():
    with pytest.raises(ValidationError, match="cannot be None"):
        AlphaGenomeScoreVariantsInput(variants=None)


# ── AlphaGenomeISM ──────────────────────────────────────────────────────────


def test_ism_rejects_ism_end_le_ism_start():
    with pytest.raises(ValidationError, match="ism_interval_end must be greater than ism_interval_start"):
        AlphaGenomeISM(
            chromosome="chr1",
            interval_start=0,
            interval_end=1000,
            ism_interval_start=200,
            ism_interval_end=200,
        )


def test_ism_rejects_ism_interval_outside_parent():
    with pytest.raises(ValidationError, match="ISM interval must be fully contained"):
        AlphaGenomeISM(
            chromosome="chr1",
            interval_start=0,
            interval_end=1000,
            ism_interval_start=900,
            ism_interval_end=1100,
        )


def test_ism_rejects_partial_variant_fields():
    with pytest.raises(
        ValidationError,
        match="variant_position, reference_bases, and alternate_bases must all be",
    ):
        AlphaGenomeISM(
            chromosome="chr1",
            interval_start=0,
            interval_end=1000,
            ism_interval_start=100,
            ism_interval_end=200,
            variant_position=150,
        )


def test_ism_accepts_optional_variant_fields_all_present():
    req = AlphaGenomeISM(
        chromosome="chr1",
        interval_start=0,
        interval_end=1000,
        ism_interval_start=100,
        ism_interval_end=200,
        variant_position=150,
        reference_bases="A",
        alternate_bases="G",
    )
    assert req.variant_position == 150
    assert req.reference_bases == "A"
    assert req.alternate_bases == "G"


def test_ism_accepts_no_variant_fields():
    req = AlphaGenomeISM(
        chromosome="chr1",
        interval_start=0,
        interval_end=1000,
        ism_interval_start=100,
        ism_interval_end=200,
    )
    assert req.variant_position is None
    assert req.reference_bases is None
    assert req.alternate_bases is None


# ── AlphaGenomeScoreISMInput ────────────────────────────────────────────────


def test_score_ism_rejects_empty_list():
    with pytest.raises(ValidationError, match="requests cannot be empty"):
        AlphaGenomeScoreISMInput(requests=[])
