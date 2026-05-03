"""Shared sequence windowing helpers for sequence-to-function models."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from proto_tools.utils import BaseToolInput, InputField, return_invalid_nucleotide_chars


class SequenceTargetRange(BaseToolInput):
    """Target range to score within a provided source sequence.

    Use this when a sequence is longer than the model's required context length
    and the tool should choose the model input window for you. The range marks
    the sequence-relative span the caller cares about; the tool then extracts a
    fixed-length model context window that keeps this span inside the model's
    output bins.

    Window placement is start-aligned when possible: the output span begins at
    ``start`` unless source-sequence boundaries force the context window left.

    Coordinates are 0-based and use an exclusive end. Unlike AlphaGenome
    intervals, these ranges are relative to each provided sequence rather than
    to a chromosome. ``end == start`` is allowed so generators can score an
    empty target prefix during incremental construction.

    Attributes:
        start (int): Target start coordinate (0-based, inclusive).
        end (int): Target end coordinate (0-based, exclusive). May equal
            ``start`` for an empty target prefix.
    """

    start: int = InputField(ge=0, description="0-based inclusive start in the provided sequence")
    end: int = InputField(ge=0, description="0-based exclusive end in the provided sequence")

    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_coordinate(cls, value: Any, info: ValidationInfo) -> int:
        """Validate target range coordinates."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{info.field_name}: must be an integer, got {type(value).__name__}={value!r}")
        return int(value)

    def model_post_init(self, __context: object) -> None:
        """Validate target range coordinates."""
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")


class PreparedSequenceWindow(BaseModel):
    """Model-ready sequence and source-coordinate metadata.

    ``model_sequence`` is the fixed-length sequence actually sent to the model.
    The coordinate fields describe where that model input window and the model
    output bins land in the original source sequence supplied by the caller.
    """

    model_config = ConfigDict(frozen=True)

    model_sequence: str = Field(description="Fixed-length sequence passed to the model")
    context_start: int = Field(description="0-based model input start in the provided sequence")
    context_end: int = Field(description="0-based exclusive model input end in the provided sequence")
    output_start: int = Field(description="0-based output-bin span start in the provided sequence")
    output_end: int = Field(description="0-based exclusive output-bin span end in the provided sequence")
    target_start: int | None = Field(default=None, description="Requested target start in the provided sequence")
    target_end: int | None = Field(default=None, description="Requested target end in the provided sequence")


def _validate_target_output_overlap(
    *,
    idx: int,
    target_start: int,
    target_end: int,
    output_start: int,
    output_length: int,
) -> None:
    output_end = output_start + output_length
    if target_start < output_start or target_start >= output_end:
        raise ValueError(
            f"target_ranges[{idx}].start={target_start} falls outside model output window "
            f"[{output_start}, {output_end})."
        )
    if target_end > output_end:
        raise ValueError(
            f"target_ranges[{idx}].end={target_end} falls outside model output window [{output_start}, {output_end})."
        )


def validate_dna_sequence(sequence: str) -> str:
    """Validate and normalize a DNA sequence."""
    if not sequence or not sequence.strip():
        raise ValueError("Sequence cannot be empty")
    sequence = sequence.upper()
    invalid_chars = return_invalid_nucleotide_chars(sequence, additional_valid_chars="N")
    if invalid_chars:
        raise ValueError(f"Invalid nucleotide characters in sequence: {', '.join(sorted(invalid_chars))}")
    return sequence


def prepare_model_windows(
    sequences: list[str],
    *,
    context_length: int,
    output_length: int,
    target_ranges: list[SequenceTargetRange] | None = None,
) -> list[PreparedSequenceWindow]:
    """Prepare exact model windows or extract them from longer source sequences.

    Without ``target_ranges``, every sequence must already be exactly the model
    context length. With ``target_ranges``, each provided sequence may be
    longer; the returned window is the fixed-length slice sent to the model,
    plus sequence-relative coordinates for the context window, output span, and
    requested target range.
    """
    if target_ranges is not None and len(target_ranges) != len(sequences):
        raise ValueError(f"Expected {len(sequences)} target_ranges, got {len(target_ranges)}.")

    output_flank = (context_length - output_length) // 2
    prepared: list[PreparedSequenceWindow] = []
    for idx, raw_sequence in enumerate(sequences):
        sequence = validate_dna_sequence(raw_sequence)
        if target_ranges is None:
            if len(sequence) != context_length:
                raise ValueError(f"Input sequence must have length {context_length}, got {len(sequence)}")
            prepared.append(
                PreparedSequenceWindow(
                    model_sequence=sequence,
                    context_start=0,
                    context_end=context_length,
                    output_start=output_flank,
                    output_end=output_flank + output_length,
                    target_start=None,
                    target_end=None,
                )
            )
            continue

        target_range = target_ranges[idx]
        target_start = target_range.start
        target_end = target_range.end
        if len(sequence) < context_length:
            raise ValueError(
                f"Input sequence must have length at least {context_length} when target coordinates are provided, "
                f"got {len(sequence)}."
            )
        if target_start < 0:
            raise ValueError(f"target_ranges[{idx}].start must be >= 0.")
        if target_start >= len(sequence):
            raise ValueError(f"target_ranges[{idx}].start must be < sequence length.")
        if target_end < target_start:
            raise ValueError(f"target_ranges[{idx}].end must be >= target_ranges[{idx}].start.")
        if target_end > len(sequence):
            raise ValueError(f"target_ranges[{idx}].end must be <= sequence length.")

        context_start = target_start - output_flank
        if context_start < 0:
            context_start = 0
        max_start = max(0, len(sequence) - context_length)
        if context_start > max_start:
            context_start = max_start
        context_end = context_start + context_length
        model_sequence = sequence[context_start:context_end]

        output_start = context_start + output_flank
        _validate_target_output_overlap(
            idx=idx,
            target_start=target_start,
            target_end=target_end,
            output_start=output_start,
            output_length=output_length,
        )

        prepared.append(
            PreparedSequenceWindow(
                model_sequence=model_sequence,
                context_start=context_start,
                context_end=context_end,
                output_start=output_start,
                output_end=output_start + output_length,
                target_start=target_start,
                target_end=target_end,
            )
        )

    return prepared
