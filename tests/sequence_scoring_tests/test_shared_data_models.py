"""Tests for shared sequence-scoring data helpers."""

import pytest

from proto_tools.tools.sequence_scoring.shared_data_models import (
    SequenceTargetRange,
    prepare_model_windows,
)


def test_prepare_model_windows_rejects_target_before_output_window():
    """Targets must be covered by the model's output bins after clamping."""
    with pytest.raises(ValueError, match="outside model output window"):
        prepare_model_windows(
            ["A" * 100],
            context_length=100,
            output_length=20,
            target_ranges=[SequenceTargetRange(start=10, end=20)],
        )


def test_prepare_model_windows_rejects_target_end_after_output_window():
    """A supplied target end must also fit inside the model output window."""
    with pytest.raises(ValueError, match="outside model output window"):
        prepare_model_windows(
            ["A" * 200],
            context_length=100,
            output_length=20,
            target_ranges=[SequenceTargetRange(start=150, end=170)],
        )


def test_prepare_model_windows_rejects_past_end_target_start():
    """A target start must point to a real source-sequence base."""
    with pytest.raises(ValueError, match="< sequence length"):
        prepare_model_windows(
            ["A" * 100],
            context_length=100,
            output_length=20,
            target_ranges=[SequenceTargetRange(start=100, end=100)],
        )


def test_prepare_model_windows_rejects_target_windows_without_full_context():
    """Target-coordinate mode requires enough source sequence for a full model window."""
    with pytest.raises(ValueError, match="at least 100"):
        prepare_model_windows(
            ["A" * 50],
            context_length=100,
            output_length=20,
            target_ranges=[SequenceTargetRange(start=40, end=45)],
        )


def test_sequence_target_range_rejects_reversed_coordinates():
    """Target ranges should keep start and end paired and ordered."""
    with pytest.raises(ValueError, match="integer"):
        SequenceTargetRange(start=True, end=2)
    with pytest.raises(ValueError, match="greater than or equal"):
        SequenceTargetRange(start=5, end=4)
