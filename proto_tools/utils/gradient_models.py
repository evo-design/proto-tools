"""Shared base models for relaxed-sequence gradient tools."""

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField


class GradientInput(BaseToolInput):
    """Relaxed sequence state for one gradient evaluation step.

    Attributes:
        logits (list[list[float]]): Relaxed sequence logits with shape
            ``(sequence_length, 20)``. Column order is backend-specific and
            must match the tool contract or output ``vocab``.
        temperature (float): Softmax temperature for relaxing logits into a
            continuous sequence distribution.
    """

    logits: list[list[float]] = InputField(
        description="Relaxed sequence logits with shape (L, 20) in backend-specific amino-acid order.",
        examples=[[[0.0] * 20, [0.0] * 20]],
    )
    temperature: float = InputField(
        default=1.0,
        description="Softmax temperature used to convert logits into a relaxed amino-acid probability distribution.",
        gt=0.0,
    )

    @field_validator("logits")
    @classmethod
    def validate_logits(cls, logits: list[list[float]]) -> list[list[float]]:
        """Ensure logits are a non-empty rectangular ``L x 20`` matrix."""
        if not logits:
            raise ValueError("logits must contain at least one position")

        expected_width = len(PROTEIN_AMINO_ACIDS)
        for idx, row in enumerate(logits):
            if len(row) != expected_width:
                raise ValueError(f"logits row {idx} must have {expected_width} columns, got {len(row)}")
        return logits


class GradientOutput(BaseToolOutput):
    """Gradient of a backend objective with respect to relaxed sequence logits.

    Attributes:
        gradient (list[list[float]]): Gradient matrix matching the input logits
            shape.
        loss (float): Backend-local scalar objective value.
        metrics (dict[str, Any]): Auxiliary metrics reported alongside the
            scalar objective.
        vocab (list[str]): Column ordering for both the input logits and the
            returned gradient.
    """

    gradient: list[list[float]] = Field(
        description="Gradient matrix with the same shape and amino-acid column order as the input logits."
    )
    loss: float = Field(description="Scalar objective value returned by the backend for this relaxed sequence.")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metrics reported alongside the scalar objective value.",
    )
    vocab: list[str] = Field(
        description="Amino-acid symbols defining the column ordering for both the input logits and the returned gradient."
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export the gradient bundle as JSON."""
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        path = Path(export_path).with_suffix(".json")
        with open(path, "w") as f:
            json.dump(
                self.model_dump(include={"gradient", "loss", "metrics", "vocab"}),
                f,
                indent=2,
            )
