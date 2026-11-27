"""proto_tools/utils/gradient/objectives.py.

Composition of raw gradient terms into one minimizable objective.

A gradient tool returns each requested scalar's raw value and gradient, with no direction, weight
or loss shaping applied. This module combines a selection of them into one objective, keeping
``ObjectiveResult.gradient`` the gradient of ``ObjectiveResult.objective_loss``.

Composition depends on the term models. The goals are independent and live in
:mod:`proto_tools.utils.gradient.goals`.

NumPy and plain Python only, no PyTorch, so composition runs in the main process.

Shapes use ``(L, V)`` for relaxed input positions by vocabulary columns, matching
the gradients returned by the tool.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field, SerializeAsAny

from proto_tools.utils.gradient.goals import Goal, GoalDirection, MaximizeGoal, MinimizeGoal

# Field types and isinstance targets, so the classes are needed rather than the annotations.
from proto_tools.utils.gradient.models import GradientResult
from proto_tools.utils.gradient.terms import DifferentiableTerm, TermGradient

#: What can be composed: answered terms, whole evaluated items, or a mixture of the two.
ComposableGradients = TermGradient | GradientResult | Sequence[TermGradient | GradientResult]

#: How a goal or weight names the term it applies to: by the term's ``id``, or by the term itself.
TermKey = str | DifferentiableTerm

#: Goals for a selection of terms: one per term in order, or keyed by the term they apply to.
GoalsArgument = Mapping[TermKey, Goal | GoalDirection | None] | Sequence[Goal | GoalDirection | None]

#: Weights for a selection of terms, in the same two forms.
WeightsArgument = Mapping[TermKey, float | None] | Sequence[float | None]


class TermContribution(BaseModel):
    """What one term contributed to a composed objective.

    Record the term's loss, the coefficient applied to its gradient, and the norm of its
    weighted contribution. The raw value stays on the original :class:`TermGradient`,
    and the caller keeps the goal used to transform it.

    Serialization preserves a concrete term's selectors, such as
    ``{"id": "activity", "cell_type": "HepG2"}``. Validation uses the annotated base
    class, however, which rejects those extra selector fields with ``ValidationError``.
    Restoring a concrete term requires the tool's narrowed model, as in
    :class:`TermGradient`. There is no registry here to choose a term class automatically.

    Attributes:
        term (SerializeAsAny[DifferentiableTerm]): The quantity this describes, as it was
            requested, serialized with its own selectors rather than the base's fields.
        term_loss (float): The goal's loss at that term's value, before weighting.
        dloss_dvalue (float): Chain-rule coefficient the raw gradient was scaled by.
        weight (float): Weight applied to this term's loss and gradient contribution.
        contribution_l2_norm (float): Euclidean norm of this term's weighted gradient contribution.
    """

    term: SerializeAsAny[DifferentiableTerm] = Field(
        description="The quantity this describes, as it was requested, with its own selectors."
    )
    term_loss: float = Field(description="The goal's loss at that term's value, before weighting.")
    dloss_dvalue: float = Field(description="Chain-rule coefficient the raw gradient was scaled by.")
    weight: float = Field(description="Weight applied to this term's loss and gradient contribution.")
    contribution_l2_norm: float = Field(description="Norm of this term's weighted gradient contribution.")


class ObjectiveResult(BaseModel):
    """The scalar loss and gradient of a composed objective.

    The pair satisfies ``objective_loss = sum(w_i * f_i(v_i))`` and
    ``gradient = sum(w_i * f_i'(v_i) * grad(v_i))``, so the gradient is exactly the derivative of
    the loss reported beside it.

    Attributes:
        objective_loss (float): Weighted sum of every term's loss.
        gradient (list[list[float]]): Gradient of ``objective_loss`` with respect to the
            differentiable input, in the same shape and column order as the term gradients it was
            built from.
        contributions (list[TermContribution]): Per-term diagnostics, in term order.
    """

    objective_loss: float = Field(description="Weighted sum of every term's loss.")
    gradient: list[list[float]] = Field(description="Gradient of objective_loss w.r.t. the input.")
    contributions: list[TermContribution] = Field(description="Per-term diagnostics, in term order.")


def resolve_goal(gradient: TermGradient, goal: Goal | GoalDirection | None) -> Goal:
    """Decide which goal applies to a term.

    An explicit goal always wins. Otherwise a term whose advisory direction states which way is
    better supplies one; a term with no inherent direction does not, and says so rather than
    guessing.

    Args:
        gradient (TermGradient): The answered term to resolve a goal for.
        goal (Goal | GoalDirection | None): Explicit goal or direction, or ``None`` to infer.

    Returns:
        Goal: The goal to apply.

    Raises:
        ValueError: If the supplied goal is neither a ``Goal`` nor a recognized direction, or if
            no goal was given and the term has no advisory direction to fall back on.
    """
    if isinstance(goal, Goal):
        return goal
    if goal is not None:
        # An explicit goal that is not one must fail, rather than quietly becoming the inferred one
        if goal == "minimize":
            return MinimizeGoal()
        if goal == "maximize":
            return MaximizeGoal()
        raise ValueError(
            f"{goal!r} is not a goal: pass 'minimize', 'maximize', or a Goal such as AtLeastGoal, "
            "AtMostGoal, RangeGoal, SigmoidGoal or ClippedLinearGoal."
        )

    # Metadata states the direction; the term class is the default, and the only clue without it.
    direction = (
        gradient.metadata.suggested_optimization_direction
        if gradient.metadata is not None
        else type(gradient.term).SUGGESTED_OPTIMIZATION_DIRECTION
    )
    if direction == "lower":
        return MinimizeGoal()
    if direction == "higher":
        return MaximizeGoal()
    raise ValueError(
        f"term {gradient.term!r} has suggested_optimization_direction={direction!r}, so it needs an explicit "
        "goal. Supply one of 'minimize', 'maximize', or a Goal such as AtLeastGoal, AtMostGoal, RangeGoal, "
        "SigmoidGoal or ClippedLinearGoal."
    )


def _collect_gradients(inputs: ComposableGradients) -> tuple[list[TermGradient], bool]:
    """Flatten whatever was handed in into the answered terms it contains.

    A result contributes all of its terms, in its own order, and several results contribute
    theirs in the order the results were given. A whole :class:`GradientOutput` is deliberately
    not accepted: its results are different candidate sequences, which do not share an input and
    so do not belong in one objective.

    Args:
        inputs (ComposableGradients): One answered term or evaluated item, or a sequence mixing
            the two.

    Returns:
        tuple[list[TermGradient], bool]: The collected gradients, and whether any of them came
            out of a result rather than being named individually.

    Raises:
        TypeError: If something other than a term gradient or an evaluated item was supplied.
    """
    items = [inputs] if isinstance(inputs, TermGradient | GradientResult) else list(inputs)
    gradients: list[TermGradient] = []
    from_results = False
    for item in items:
        if isinstance(item, GradientResult):
            from_results = True
            gradients.extend(item.terms)
        elif isinstance(item, TermGradient):
            gradients.append(item)
        else:
            raise TypeError(f"compose a TermGradient or a GradientResult, got {item!r}")
    return gradients, from_results


def _match_key(key: TermKey, terms: Sequence[DifferentiableTerm], *, label: str) -> int:
    """Find the one term a goal or weight key names.

    A term key matches the whole specification, selectors included. A string key matches a term's
    ``id``, which names a *kind* of quantity, so it only works where one term of that kind was
    asked for: two activity terms differing by cell type are both ``"activity"``, and naming them
    that way would be a guess about which one was meant.

    Args:
        key (TermKey): The term itself, or its ``id``.
        terms (Sequence[DifferentiableTerm]): The collected terms, in composition order.
        label (str): What to call the argument in an error, ``"goals"`` or ``"weights"``.

    Returns:
        int: Index of the matching term.

    Raises:
        ValueError: If the key matches no collected term, or matches more than one.
        TypeError: If the key is neither a term nor a string.
    """
    if isinstance(key, DifferentiableTerm):
        matches = [index for index, term in enumerate(terms) if term == key]
        if not matches:
            raise ValueError(f"{label} key {key!r} matches none of the collected terms: {list(terms)}")
        return matches[0]
    if isinstance(key, str):
        matches = [index for index, term in enumerate(terms) if term.id == key]
        if not matches:
            available = sorted({term.id for term in terms})
            raise ValueError(f"{label} key {key!r} matches no collected term; the ids present are {available}")
        if len(matches) > 1:
            raise ValueError(
                f"{label} key {key!r} matches {len(matches)} terms, so it does not say which one is meant. "
                "An id names a kind of quantity; key these by the term itself to tell them apart."
            )
        return matches[0]
    raise TypeError(f"{label} keys must be a term or a term id, got {key!r}")


def _per_term(
    supplied: GoalsArgument | WeightsArgument | None,
    terms: Sequence[DifferentiableTerm],
    *,
    label: str,
    default: object,
    require_mapping: bool,
) -> list[object]:
    """Line a goals or weights argument up with the terms it applies to.

    Two spellings are accepted. A mapping is keyed by a term or its id, and anything it leaves
    out takes the default, which is how a caller states a goal for one quantity without restating
    the rest. A sequence runs parallel to the collected terms, so it must be exactly as long as
    they are, with ``None`` in any slot that should take the default.

    A sequence is only meaningful when the caller named the terms themselves. A result does
    return its terms in the order they were requested, but reading a sequence against that order
    means knowing what a container holds and in what arrangement, which a caller should not have
    to reason about to state a goal, so the sequence form is refused there.

    Args:
        supplied (GoalsArgument | WeightsArgument | None): What the caller passed, or ``None``.
        terms (Sequence[DifferentiableTerm]): The collected terms, in composition order.
        label (str): What to call the argument in an error, ``"goals"`` or ``"weights"``.
        default (object): What a term the caller did not mention receives.
        require_mapping (bool): Whether a result was collected, making the sequence form invalid.

    Returns:
        list[object]: One entry per term, in composition order.

    Raises:
        ValueError: If a sequence is given for collected results, is the wrong length, or a
            mapping names an unknown term, an ambiguous id, or one term twice.
    """
    if supplied is None:
        return [default] * len(terms)
    if isinstance(supplied, Mapping):
        assigned: dict[int, object] = {}
        for key, value in supplied.items():
            index = _match_key(key, terms, label=label)
            if index in assigned:
                raise ValueError(f"two {label} keys name the same term {terms[index]!r}")
            assigned[index] = value
        return [default if assigned.get(index) is None else assigned[index] for index in range(len(terms))]
    if require_mapping:
        raise ValueError(
            f"{label} must be a mapping when a GradientResult is composed: a sequence would have to be "
            "read against the arrangement of terms inside it. Key them by term or by term id instead."
        )
    entries = list(supplied)
    if len(entries) != len(terms):
        raise ValueError(
            f"{label} has {len(entries)} entries for {len(terms)} terms; a sequence must line up with "
            "the terms one for one, or be given as a mapping keyed by term."
        )
    return [default if entry is None else entry for entry in entries]


def _reject_incompatible(gradients: Sequence[TermGradient], *, strict: bool) -> None:
    """Refuse to add gradients that are not derivatives with respect to the same thing.

    Matching dimensions prove nothing: two ``(60, 4)`` matrices can describe different loci, in
    different column orders, in different coordinate spaces, at different points of a design run.
    :meth:`GradientMetadata.incompatibility_with` compares each of those, and a gradient that
    reports no metadata cannot be checked at all, so it is refused rather than assumed to fit.

    Which tool produced a gradient is not among the checks. Two models differentiating the same
    relaxed input at the same point compose, and one model's gradients from two different steps
    do not.

    Args:
        gradients (Sequence[TermGradient]): The collected gradients.
        strict (bool): Whether gradients taken at different points are also incompatible.

    Raises:
        ValueError: If any gradient lacks metadata, or disagrees with the first.
    """
    if len(gradients) < 2:
        return
    unchecked = [answer.term for answer in gradients if answer.metadata is None]
    if unchecked:
        raise ValueError(
            f"terms {unchecked} report no gradient metadata, so there is no way to tell whether they "
            "are derivatives with respect to the same input. Combining gradients requires it."
        )
    described = [(answer.term, answer.metadata) for answer in gradients if answer.metadata is not None]
    (reference_term, reference), *rest = described
    for term, metadata in rest:
        reason = reference.incompatibility_with(metadata, strict=strict)
        if reason:
            raise ValueError(f"cannot combine {reference_term!r} with {term!r}: {reason}")


def compose_gradients(
    result: GradientResult,
    goals: GoalsArgument | None = None,
    weights: WeightsArgument | None = None,
    *,
    strict: bool = True,
) -> ObjectiveResult:
    """Combine one evaluated item's gradients into a single minimizable objective.

    Thin wrapper over :func:`compose_terms` for the common case of composing everything one item
    returned. Because a result is composed rather than individual terms, goals and weights must
    be keyed rather than positional. See that function for the rules and the errors raised.

    Args:
        result (GradientResult): One evaluated item whose terms are to be composed.
        goals (GoalsArgument | None): Goal per term, keyed by term or term id.
        weights (WeightsArgument | None): Non-negative weight per term, keyed the same way.
        strict (bool): Whether gradients taken at different points are incompatible.

    Returns:
        ObjectiveResult: The composed loss, its gradient, and per-term diagnostics.
    """
    return compose_terms(result, goals=goals, weights=weights, strict=strict)


def compose_terms(
    inputs: ComposableGradients,
    goals: GoalsArgument | None = None,
    weights: WeightsArgument | None = None,
    *,
    strict: bool = True,
) -> ObjectiveResult:
    """Combine gradient terms into a single minimizable objective.

    Each term's raw gradient is scaled by its goal's chain-rule coefficient, then weighted and
    summed, so the returned gradient is exactly the gradient of the returned loss. Step against it
    to improve the objective.

    What can be composed is whatever refers to the same differentiable input: individual answered
    terms, a whole evaluated item, several items, or a mixture. Two models that differentiated the
    same relaxed state at the same point compose with each other; the check is on the gradients'
    own metadata, not on which tool produced them. A whole :class:`GradientOutput` is not
    accepted, because its results are separate candidate sequences.

    Every collected term takes part. Omitting a goal asks for the term's advisory direction rather
    than excluding it, and omitting a weight asks for 1.0.

    Nothing here rescales a gradient independently of the loss reported with it. Balancing terms
    by gradient magnitude produces a descent direction that is not the gradient of any reported
    scalar, and it would undo the bounded contribution that :class:`SigmoidGoal` and
    :class:`ClippedLinearGoal` exist to provide. Reconcile differing scales inside the loss with
    :class:`ClippedLinearGoal` instead, where the chain rule carries the change through.

    Args:
        inputs (ComposableGradients): One answered term or evaluated item, or a sequence mixing
            the two.
        goals (GoalsArgument | None): Goal per term, keyed by term or by term id, or a sequence
            running parallel to the terms. The sequence form is only available when every input
            was an individual term, so that stating a goal never requires knowing how a result
            arranges what it holds. A term left out, or given ``None``, follows its advisory
            direction.
        weights (WeightsArgument | None): Non-negative weight per term, in the same two forms,
            defaulting to 1.0.
        strict (bool): Whether gradients taken at different points of a design run are
            incompatible. Leave this on unless reusing an earlier step's gradient deliberately.

    Returns:
        ObjectiveResult: The composed loss, its gradient, and per-term diagnostics.

    Raises:
        ValueError: If there are no terms, a term was asked for twice, a goal or weight names an
            unknown or ambiguous term or does not line up with them, a weight is negative or
            non-finite, a term is missing its gradient or its metadata, the gradients are not
            derivatives with respect to the same input, or any coefficient, loss, weighted loss or
            running total is not finite.
        TypeError: If something other than a term gradient or an evaluated item was supplied.
    """
    gradients, from_results = _collect_gradients(inputs)
    if not gradients:
        raise ValueError("no gradient terms to compose; none were supplied")

    # The requested quantities, which goals and weights are matched against
    terms = [answer.term for answer in gradients]

    # Reject a quantity asked for twice, which would make a goal or weight ambiguous
    duplicates = [term for index, term in enumerate(terms) if term in terms[:index]]
    if duplicates:
        raise ValueError(f"the same term appears more than once in one result: {duplicates}")

    resolved_goals = _per_term(goals, terms, label="goals", default=None, require_mapping=from_results)
    resolved_weights = _per_term(weights, terms, label="weights", default=1.0, require_mapping=from_results)

    # Reject forward-only terms, which have a value but nothing to compose
    missing = [answer.term for answer in gradients if answer.gradient is None]
    if missing:
        raise ValueError(
            f"terms {missing} carry no gradient, so they cannot be composed. "
            "Re-run with gradients enabled, or compose only terms that have them."
        )

    # Require one shared (L, V), since summing gradients across coordinate systems is meaningless
    shapes = {(len(a.gradient), len(a.gradient[0]) if a.gradient else 0) for a in gradients}  # type: ignore[arg-type]
    if len(shapes) > 1:
        raise ValueError(f"term gradients disagree in shape: {sorted(shapes)}; they must share one coordinate system")

    # Check that the gradients refer to the same differentiated input.
    _reject_incompatible(gradients, strict=strict)

    # Initialize the running objective and its gradient (L, V)
    objective_loss = 0.0
    combined = np.zeros(shapes.pop(), dtype=np.float64)
    contributions: list[TermContribution] = []

    for answer, requested, supplied_weight in zip(gradients, resolved_goals, resolved_weights, strict=True):
        # Resolve what this term is being asked for, and how much it counts
        goal = resolve_goal(answer, requested)  # type: ignore[arg-type]
        weight = float(supplied_weight)  # type: ignore[arg-type]
        if weight < 0.0 or not math.isfinite(weight):
            raise ValueError(f"weight for term {answer.term!r} must be finite and non-negative, got {weight}")

        # Evaluate the goal's loss and derivative at the raw scalar value.
        term_loss = goal.loss(answer.term_value)
        coefficient = goal.dloss_dvalue(answer.term_value)
        if not math.isfinite(term_loss) or not math.isfinite(coefficient):
            raise ValueError(
                f"{type(goal).__name__} on term {answer.term!r} produced a non-finite loss ({term_loss}) "
                f"or coefficient ({coefficient}) at value {answer.term_value}"
            )

        # Apply the chain rule, scaling the raw gradient by d(loss)/d(value) (L, V) -> (L, V)
        contribution = coefficient * np.asarray(answer.gradient, dtype=np.float64)

        # Scale this term's contribution by its weight (L, V) -> (L, V)
        contribution = weight * contribution
        if not np.all(np.isfinite(contribution)):
            raise ValueError(f"term {answer.term!r} produced a non-finite gradient contribution")

        # Finite factors can still multiply or sum to infinity while each one looks reasonable.
        weighted_loss = weight * term_loss
        objective_loss += weighted_loss
        if not math.isfinite(weighted_loss) or not math.isfinite(objective_loss):
            raise ValueError(
                f"term {answer.term!r} drove the objective loss non-finite: weight {weight} times "
                f"loss {term_loss} reached {weighted_loss}, running total {objective_loss}."
            )

        # Accumulate into the running objective (L, V) += (L, V)
        combined += contribution

        # Record what this term contributed, for diagnosing which one drove the step
        contributions.append(
            TermContribution(
                term=answer.term,
                term_loss=term_loss,
                dloss_dvalue=coefficient,
                weight=weight,
                contribution_l2_norm=float(np.linalg.norm(contribution.ravel())),
            )
        )

    # Return the composed objective and the gradient to descend against (L, V)
    return ObjectiveResult(objective_loss=objective_loss, gradient=combined.tolist(), contributions=contributions)


__all__ = [
    "ComposableGradients",
    "GoalsArgument",
    "ObjectiveResult",
    "TermContribution",
    "TermKey",
    "WeightsArgument",
    "compose_gradients",
    "compose_terms",
    "resolve_goal",
]
