"""Validation tests for AlphaGenome input models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.sequence_scoring.alphagenome import (
    AlphaGenomePredictSequenceInput,
)


def test_predict_sequence_accepts_supported_min_length():
    sequence = "ACGT" * (16_384 // 4)

    parsed = AlphaGenomePredictSequenceInput(sequence=sequence)

    assert len(parsed.sequence) == 16_384


def test_predict_sequence_rejects_unsupported_length():
    sequence = "ACGT" * (2_048 // 4)

    with pytest.raises(
        ValidationError,
        match="supported AlphaGenome context length",
    ):
        AlphaGenomePredictSequenceInput(sequence=sequence)
