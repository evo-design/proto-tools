"""proto_tools/utils/gradient/metadata.py.

What a returned gradient was taken with respect to, and whether two of them can be combined.

A gradient is only meaningful alongside the state it differentiates: the same numbers taken at a
different point, in a different coordinate space or against a different column order do not add
up. :class:`GradientMetadata` records those quantities so a caller can check before combining.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from proto_tools.utils.tool_io import Directionality

# Which variable a gradient is a derivative with respect to.
CoordinateSpace = Literal["logits", "probabilities"]


def gradient_l2_norm(gradient: Sequence[Sequence[float]]) -> float:
    """Return the Euclidean norm of a gradient matrix, flattened.

    The matrix is treated as one long vector, so this is the Frobenius norm rather than the
    spectral norm a matrix-aware ``norm`` would give.

    Args:
        gradient (Sequence[Sequence[float]]): An ``(L, V)`` gradient matrix.

    Returns:
        float: Square root of the summed squares of every entry.
    """
    return math.sqrt(math.fsum(value * value for row in gradient for value in row))


class GradientMetadata(BaseModel):
    """Metadata tracking the quantities that determine whether a gradient can be combined with others.

    Adding two gradients is only meaningful when they are derivatives of different scalars with
    respect to the same variable, at the same point, and in the same columns. Shape alone
    establishes none of that: two ``(60, 4)`` matrices can describe different loci of different
    molecules in different parameterizations. Each field below pins one of those axes.

    Attributes:
        description (str): What the scalar measures.
        units (str | None): Physical units where the quantity has them, such as ``"angstroms"``.
            Declared by the term class as ``UNITS`` and copied in, so a caller cannot state it.
        suggested_optimization_direction (Directionality): Which way this quantity is generally
            better, as advisory metadata. Copied from the term's class-level declaration, so a
            caller cannot state it. It never affects the sign of the returned gradient.
        vocab (list[str]): Column identity and order. Two tools may both use 20 columns and order
            them differently, so this compares as an ordered list.
        coordinate_space (CoordinateSpace): Which variable the derivative is with respect to.
            ``"logits"`` for unnormalized scores that the backend softmaxes inside the graph;
            ``"probabilities"`` when the rows were consumed directly as a distribution. The two
            differ by the softmax Jacobian, so they never combine.
        length (int): Number of differentiated rows, matching the gradient's row count.
        positions (list[int] | None): Which input position each row refers to, 1-indexed, or
            ``None`` for the contiguous ``1..length``. Tools that relax a window inside a longer
            context report explicit, possibly non-contiguous positions.
        input_fingerprint (str): Digest of the differentiated matrix, identifying the *point* the
            gradient was taken at. Catches combining a stale gradient from an earlier step with a
            fresh one, which every other field would pass.
        hard (float): Straight-through coefficient. Above zero the gradient is a surrogate rather
            than the true derivative of the forward pass, which composition reports but allows.
        soft (float): Relaxation blend in force, recorded rather than compared.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(default="", title="Description", description="What this scalar quantity measures.")
    units: str | None = Field(
        default=None, title="Units", description="Physical units of the value, where it has them."
    )
    suggested_optimization_direction: Directionality = Field(
        default="context-dependent",
        title="Optimization Direction",
        description="Advisory direction for this quantity; does not affect the sign of the gradient.",
    )
    vocab: list[str] = Field(title="Vocabulary", description="Column identity and order for this gradient.")
    coordinate_space: CoordinateSpace = Field(
        title="Coordinate Space",
        description="Whether the derivative is w.r.t. logits or probabilities.",
    )
    length: int = Field(ge=1, title="Length", description="Number of differentiated rows.")
    positions: list[int] | None = Field(
        default=None,
        title="Positions",
        description="1-indexed input position per row; None means contiguous 1..length.",
    )
    input_fingerprint: str = Field(
        title="Input Fingerprint",
        description="Digest of the differentiated matrix, identifying the point the gradient was taken at.",
    )
    hard: float = Field(default=0.0, title="Hard Mixing", description="Straight-through coefficient in force.")
    soft: float = Field(default=1.0, title="Soft Mixing", description="Relaxation blend in force.")

    @classmethod
    def from_input(
        cls,
        matrix: Sequence[Sequence[float]],
        *,
        vocab: Sequence[str],
        temperature: float | None,
        description: str = "",
        positions: Sequence[int] | None = None,
        hard: float = 0.0,
        soft: float = 1.0,
    ) -> Self:
        """Describe a gradient taken with respect to ``matrix``.

        Args:
            matrix (Sequence[Sequence[float]]): The differentiated relaxed state, ``(L, V)``.
            vocab (Sequence[str]): Column identity and order.
            temperature (float | None): The input's temperature, which fixes the coordinate space.
            description (str): What the scalar measures.
            positions (Sequence[int] | None): 1-indexed input position per row, where the rows are
                not simply ``1..L``.
            hard (float): Straight-through coefficient in force.
            soft (float): Relaxation blend in force.

        Returns:
            Self: Metadata describing this gradient.
        """
        return cls(
            description=description,
            vocab=list(vocab),
            coordinate_space="probabilities" if temperature is None else "logits",
            length=len(matrix),
            positions=list(positions) if positions is not None else None,
            input_fingerprint=input_fingerprint(matrix),
            hard=hard,
            soft=soft,
        )

    def _rows(self) -> list[int]:
        """Return the explicit per-row positions, materializing the contiguous default."""
        return list(self.positions) if self.positions is not None else list(range(1, self.length + 1))

    def incompatibility_with(self, other: GradientMetadata, *, strict: bool = True) -> str | None:
        """Explain why these two gradients cannot be combined, or return ``None`` if they can.

        ``vocab``, ``coordinate_space`` and the row positions are never negotiable: a mismatch
        there makes the sum meaningless rather than approximate. The fingerprint is: reusing a
        gradient from an earlier step is a deliberate approximation some optimizers make, so
        ``strict=False`` permits it.

        Args:
            other (GradientMetadata): The metadata to compare against.
            strict (bool): Whether a differing point is also an incompatibility.

        Returns:
            str | None: A reason, or ``None`` when the two may be combined.
        """
        if self.vocab != other.vocab:
            return f"vocabularies differ: {self.vocab} vs {other.vocab}"
        if self.coordinate_space != other.coordinate_space:
            return (
                f"one gradient is with respect to {self.coordinate_space} and the other to "
                f"{other.coordinate_space}; they differ by the softmax Jacobian"
            )
        if self.length != other.length:
            return f"row counts differ: {self.length} vs {other.length}"
        rows, other_rows = self._rows(), other._rows()
        if rows != other_rows:
            return f"rows refer to different input positions: {rows[0]}-{rows[-1]} vs {other_rows[0]}-{other_rows[-1]}"
        if strict and self.input_fingerprint != other.input_fingerprint:
            return (
                f"gradients were taken at different points ({self.input_fingerprint} vs "
                f"{other.input_fingerprint}); pass strict=False to combine them anyway"
            )
        return None


def input_fingerprint(matrix: Sequence[Sequence[float]]) -> str:
    """Return a short digest identifying the exact values of a differentiated matrix.

    Hashes the float64 bytes rather than a formatted string, so the digest is stable across
    processes. It identifies the *values*, not where they live: two windows of one context
    initialized identically share a digest, which is why positions are compared separately.

    Args:
        matrix (Sequence[Sequence[float]]): The differentiated relaxed state.

    Returns:
        str: First 12 hex characters of the SHA-256 over the matrix's float64 bytes.
    """
    packed = b"".join(struct.pack("<d", float(value)) for row in matrix for value in row)
    return hashlib.sha256(packed).hexdigest()[:12]
