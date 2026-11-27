"""tests/tool_infra_tests/test_gradient_goals.py.

Unit tests for the scalar goals a caller applies to one differentiated quantity.

Every goal's stated derivative is checked against a finite difference of its own loss, so a sign
error cannot pass. Pure scalar mathematics: no terms, no gradients, no models, no environments.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from proto_tools.utils.gradient.goals import (
    AtLeastGoal,
    AtMostGoal,
    ClippedLinearGoal,
    Goal,
    MaximizeGoal,
    MinimizeGoal,
    RangeGoal,
    SigmoidGoal,
)

# Values chosen to sit away from any hinge kink, where the derivative is genuinely defined.
_GOALS_AND_POINTS = [
    (MinimizeGoal(), 1.7),
    (MaximizeGoal(), 1.7),
    (AtLeastGoal(threshold=2.0), 0.5),
    (AtMostGoal(threshold=2.0), 3.5),
    (RangeGoal(lower=1.0, upper=3.0), 0.25),
    (RangeGoal(lower=1.0, upper=3.0), 4.0),
    (SigmoidGoal(direction="maximize", center=2.0, scale=1.0), 1.3),
    (SigmoidGoal(direction="minimize", center=4.0, scale=2.0), 5.1),
    (SigmoidGoal(direction="maximize", center=0.0, scale=1.0), -3.0),
    (SigmoidGoal(direction="maximize", center=0.0, scale=1.0), 3.0),
    (SigmoidGoal(direction="minimize", center=-5.0, scale=0.5), -6.2),
    (ClippedLinearGoal(scale=-1.0, offset=1.0, floor=0.0), 0.4),
    (ClippedLinearGoal(scale=1 / 31.75, ceiling=1.0), 12.0),
]


@pytest.mark.parametrize(("goal", "point"), _GOALS_AND_POINTS, ids=lambda item: str(item)[:40])
def test_goal_derivative_matches_a_finite_difference_of_its_loss(goal: Goal, point: float) -> None:
    """Each goal's stated derivative is the derivative of the loss it reports."""
    epsilon = 1e-6
    numeric = (goal.loss(point + epsilon) - goal.loss(point - epsilon)) / (2 * epsilon)
    assert goal.dloss_dvalue(point) == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_threshold_goals_are_inert_once_satisfied() -> None:
    """A satisfied constraint contributes no loss and no gradient."""
    for goal, satisfied in (
        (AtLeastGoal(threshold=2.0), 2.5),
        (AtMostGoal(threshold=2.0), 1.5),
        (RangeGoal(lower=1.0, upper=3.0), 2.0),
    ):
        assert goal.loss(satisfied) == 0.0
        assert goal.dloss_dvalue(satisfied) == 0.0


def test_sigmoid_goal_reproduces_the_saturating_objective() -> None:
    """The maximize form is ``1 - sigmoid((v - c) / a)``, matching the shape it replaces."""
    goal = SigmoidGoal(direction="maximize", center=2.0, scale=1.0)
    expected = 1.0 - 1.0 / (1.0 + math.exp(-(3.0 - 2.0) / 1.0))

    assert goal.loss(3.0) == pytest.approx(expected)
    assert SigmoidGoal(direction="minimize", center=2.0, scale=1.0).loss(3.0) == pytest.approx(1.0 - expected)


def test_the_normalized_and_capped_confidence_objective() -> None:
    """The PAE objective this goal exists for: angstroms onto a unit scale, capped at 1.

    A quantity in physical units is divided onto a comparable scale and clipped so an extreme
    value stops adding pressure, with a value exactly on the bound still passing gradient.
    """
    goal = ClippedLinearGoal(scale=1 / 31.75, ceiling=1.0)

    assert goal.loss(0.0) == pytest.approx(0.0)
    assert goal.loss(15.875) == pytest.approx(0.5)
    assert goal.loss(31.75) == pytest.approx(1.0)
    assert goal.loss(63.5) == pytest.approx(1.0), "loss must cap at the ceiling"
    assert goal.dloss_dvalue(63.5) == 0.0, "a clipped value exerts no pressure"
    assert goal.dloss_dvalue(31.75) == pytest.approx(1 / 31.75), "a value exactly on the bound passes gradient"


def test_clipped_linear_goal_reproduces_an_inverted_confidence_objective() -> None:
    """``1 - confidence`` is a negative scale with a unit offset, gradient independent of offset."""
    goal = ClippedLinearGoal(scale=-1.0, offset=1.0, floor=0.0)

    assert goal.loss(0.8) == pytest.approx(0.2)
    assert goal.dloss_dvalue(0.8) == pytest.approx(-1.0)


def test_goals_validate_their_own_configuration() -> None:
    """Inverted bands, inverted bounds and a non-positive sigmoid width are rejected."""
    with pytest.raises(ValidationError):
        RangeGoal(lower=3.0, upper=1.0)
    with pytest.raises(ValidationError):
        ClippedLinearGoal(scale=1.0, floor=1.0, ceiling=0.0)
    with pytest.raises(ValidationError):
        SigmoidGoal(direction="maximize", center=0.0, scale=0.0)


def test_clipped_linear_goal_requires_something_to_clip_against() -> None:
    """Unclipped affine rescaling is MinimizeGoal or MaximizeGoal under a weight, so it is refused."""
    with pytest.raises(ValidationError, match="needs a floor, a ceiling, or both"):
        ClippedLinearGoal(scale=1 / 31.75)

    assert ClippedLinearGoal(scale=1.0, floor=0.0, ceiling=1.0).loss(5.0) == pytest.approx(1.0)


def test_goals_are_frozen_and_closed() -> None:
    """A goal is a value: it cannot be mistyped into silence or edited after the fact."""
    with pytest.raises(ValidationError):
        AtLeastGoal(threshold=1.0, threshhold=2.0)
    with pytest.raises(ValidationError):
        AtLeastGoal(threshold=1.0).threshold = 2.0
