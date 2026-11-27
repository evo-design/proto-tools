"""proto_tools/utils/gradient/goals.py.

What a caller wants from a single scalar quantity, expressed as a minimizable loss.

A goal maps a term's raw value to a loss and provides the derivative of that loss.
It works with scalars alone. :mod:`proto_tools.utils.gradient.objectives` uses those
derivatives to apply the chain rule to terms from any tool or model.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Shorthand for the two goals that need no parameters.
GoalDirection = Literal["minimize", "maximize"]


class Goal(BaseModel):
    """What a caller wants from one scalar quantity, as a minimizable loss.

    A goal supplies a scalar loss over a term's value and that loss's derivative with respect to
    the value. Composition applies the chain rule, so no goal needs to know anything about
    gradients, inputs or models.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    def loss(self, value: float) -> float:
        """Return the minimizable loss for a term value.

        Args:
            value (float): The term's raw scalar value.

        Returns:
            float: Loss to be minimized.

        Raises:
            NotImplementedError: If a subclass does not define a loss.
        """
        raise NotImplementedError(f"{type(self).__name__} must define loss()")

    def dloss_dvalue(self, value: float) -> float:
        """Return the derivative of :meth:`loss` with respect to the term value.

        Args:
            value (float): The term's raw scalar value.

        Returns:
            float: ``d(loss)/d(value)``, the coefficient the raw gradient is scaled by.

        Raises:
            NotImplementedError: If a subclass does not define a derivative.
        """
        raise NotImplementedError(f"{type(self).__name__} must define dloss_dvalue()")


class MinimizeGoal(Goal):
    """Drive a quantity down without limit."""

    def loss(self, value: float) -> float:
        """Return the value itself, so minimizing the loss minimizes the quantity."""
        return value

    def dloss_dvalue(self, _value: float) -> float:
        """Return the constant derivative of the identity."""
        return 1.0


class MaximizeGoal(Goal):
    """Drive a quantity up without limit."""

    def loss(self, value: float) -> float:
        """Return the negated value, so minimizing the loss maximizes the quantity."""
        return -value

    def dloss_dvalue(self, _value: float) -> float:
        """Return the constant derivative of the negation."""
        return -1.0


class AtLeastGoal(Goal):
    """Require a quantity to reach a threshold, with no pressure once it has.

    Attributes:
        threshold (float): Value at or above which the goal is satisfied.
    """

    threshold: float = Field(allow_inf_nan=False, description="Value at or above which this goal is satisfied.")

    def loss(self, value: float) -> float:
        """Return the squared shortfall below the threshold, or zero once satisfied."""
        return max(0.0, self.threshold - value) ** 2

    def dloss_dvalue(self, value: float) -> float:
        """Return the squared-shortfall derivative, or zero once the goal is satisfied."""
        return -2.0 * max(0.0, self.threshold - value)


class AtMostGoal(Goal):
    """Require a quantity to stay under a threshold, with no pressure while it does.

    Attributes:
        threshold (float): Value at or below which the goal is satisfied.
    """

    threshold: float = Field(allow_inf_nan=False, description="Value at or below which this goal is satisfied.")

    def loss(self, value: float) -> float:
        """Return the squared excess above the threshold, or zero while satisfied."""
        return max(0.0, value - self.threshold) ** 2

    def dloss_dvalue(self, value: float) -> float:
        """Return the squared-excess derivative, or zero while the goal is satisfied."""
        return 2.0 * max(0.0, value - self.threshold)


class RangeGoal(Goal):
    """Require a quantity to sit within a band, with no pressure inside it.

    Attributes:
        lower (float): Lower end of the satisfied band, inclusive.
        upper (float): Upper end of the satisfied band, inclusive.
    """

    lower: float = Field(allow_inf_nan=False, description="Lower end of the satisfied band.")
    upper: float = Field(allow_inf_nan=False, description="Upper end of the satisfied band.")

    def model_post_init(self, _context: object) -> None:
        """Reject an inverted band, which no value could satisfy."""
        if self.upper < self.lower:
            raise ValueError(f"RangeGoal upper ({self.upper}) must be >= lower ({self.lower})")

    def loss(self, value: float) -> float:
        """Return the squared distance outside the band, or zero within it."""
        return max(0.0, self.lower - value) ** 2 + max(0.0, value - self.upper) ** 2

    def dloss_dvalue(self, value: float) -> float:
        """Return the active hinge's derivative, or zero within the band."""
        return -2.0 * max(0.0, self.lower - value) + 2.0 * max(0.0, value - self.upper)


class SigmoidGoal(Goal):
    """Push a quantity one way with saturating pressure, bounding its contribution.

    The loss flattens a few multiples of ``scale`` away from ``center``. This bounds the
    term's contribution to a weighted sum even when its raw value is extreme.

    Attributes:
        direction (GoalDirection): Whether higher or lower raw values are wanted.
        center (float): Value the sigmoid is centered on, where pressure is greatest.
        scale (float): Width of the transition; larger values saturate more slowly.
    """

    direction: GoalDirection = Field(description="Whether to drive the raw value up or down.")
    center: float = Field(allow_inf_nan=False, description="Value the sigmoid is centered on.")
    scale: float = Field(default=1.0, gt=0.0, allow_inf_nan=False, description="Width of the sigmoid transition.")

    def _probability(self, value: float) -> float:
        """Return the sigmoid of the centered, scaled value, without overflowing.

        Both branches exponentiate a non-positive number, so a value far from ``center`` in
        either direction underflows to a saturated 0.0 or 1.0 instead of raising. The naive
        ``1 / (1 + exp(-z))`` overflows for ``z`` below about -709, which a quantity in raw
        physical units reaches easily.
        """
        z = (value - self.center) / self.scale
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exponential = math.exp(z)
        return exponential / (1.0 + exponential)

    def loss(self, value: float) -> float:
        """Return ``1 - sigmoid`` when maximizing, ``sigmoid`` when minimizing."""
        # Invert when maximizing, so a higher raw value lands at a lower loss
        probability = self._probability(value)
        return 1.0 - probability if self.direction == "maximize" else probability

    def dloss_dvalue(self, value: float) -> float:
        """Return the sigmoid derivative, signed by the direction."""
        # Differentiate sigmoid((v - c) / a): p(1 - p) / a, which flattens as p saturates
        probability = self._probability(value)
        slope = probability * (1.0 - probability) / self.scale

        # Negate when maximizing, matching the inverted loss above
        return -slope if self.direction == "maximize" else slope


class ClippedLinearGoal(Goal):
    """Rescale a quantity affinely and clip the result, capping how much it can contribute.

    Use this goal to put quantities with different units on comparable scales before
    summing them. Clipping limits the effect of extreme values. A negative ``scale``
    drives the quantity up.

    At least one bound is required. For an unbounded linear loss, use
    :class:`MinimizeGoal` or :class:`MaximizeGoal` with an objective weight. The offset
    changes the loss value without changing the gradient.

    Attributes:
        scale (float): Multiplier applied to the value; negative drives the quantity up.
        offset (float): Constant added after scaling. Shifts the loss, never the gradient.
        floor (float | None): Lower bound on the loss, or ``None`` for unbounded below.
        ceiling (float | None): Upper bound on the loss, or ``None`` for unbounded above.
    """

    scale: float = Field(allow_inf_nan=False, description="Multiplier applied to the value.")
    offset: float = Field(default=0.0, allow_inf_nan=False, description="Constant added after scaling.")
    floor: float | None = Field(default=None, allow_inf_nan=False, description="Lower bound on the loss.")
    ceiling: float | None = Field(default=None, allow_inf_nan=False, description="Upper bound on the loss.")

    def model_post_init(self, _context: object) -> None:
        """Require a bound to clip against, and reject bounds that exclude every value."""
        if self.floor is None and self.ceiling is None:
            raise ValueError(
                "ClippedLinearGoal needs a floor, a ceiling, or both; an unclipped affine rescaling is "
                "MinimizeGoal or MaximizeGoal under a weight."
            )
        if self.floor is not None and self.ceiling is not None and self.ceiling < self.floor:
            raise ValueError(f"ClippedLinearGoal ceiling ({self.ceiling}) must be >= floor ({self.floor})")

    def loss(self, value: float) -> float:
        """Return the affine transform of the value, clamped to the bounds."""
        # Rescale onto a comparable range. A negative scale drives the quantity up.
        raw = self.scale * value + self.offset

        # Clamp so an extreme value stops adding pressure
        if self.floor is not None:
            raw = max(raw, self.floor)
        if self.ceiling is not None:
            raw = min(raw, self.ceiling)
        return raw

    def dloss_dvalue(self, value: float) -> float:
        """Return ``scale`` while unclamped, zero once a bound is exceeded.

        A value sitting exactly on a bound still passes gradient, matching the inclusive
        convention of a clamp's backward pass.
        """
        raw = self.scale * value + self.offset

        # A clamped value is on a flat stretch of the loss, so it exerts no pressure
        if self.floor is not None and raw < self.floor:
            return 0.0
        if self.ceiling is not None and raw > self.ceiling:
            return 0.0

        # The offset shifts the loss but never its slope
        return self.scale


__all__ = [
    "AtLeastGoal",
    "AtMostGoal",
    "ClippedLinearGoal",
    "Goal",
    "GoalDirection",
    "MaximizeGoal",
    "MinimizeGoal",
    "RangeGoal",
    "SigmoidGoal",
]
