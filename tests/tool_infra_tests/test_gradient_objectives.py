"""tests/tool_infra_tests/test_gradient_objectives.py.

Unit tests for composing raw gradient terms into one minimizable objective.

Covers goal resolution from a term's advisory direction, the chain rule, weighting, the two ways
goals line up with terms, each rejection path, and the invariant that the composed gradient
differentiates the composed loss. Synthetic gradients throughout: no models, no environments, no
GPU.
"""

from __future__ import annotations

import json
import math
from typing import ClassVar, Literal

import pytest
from pydantic import Field, ValidationError

from proto_tools.utils.gradient.goals import (
    AtMostGoal,
    Goal,
    MaximizeGoal,
    MinimizeGoal,
    SigmoidGoal,
)
from proto_tools.utils.gradient.models import (
    DifferentiableTerm,
    GradientMetadata,
    GradientResult,
    TermGradient,
)
from proto_tools.utils.gradient.objectives import ObjectiveResult, compose_terms, resolve_goal
from proto_tools.utils.tool_io import Directionality


class PlainTerm(DifferentiableTerm):
    """A term with no inherent direction.

    Attributes:
        id (Literal["plain"]): Fixed identifier for this kind of quantity.
        name (str): Selector naming the quantity.
    """

    id: Literal["plain"] = Field(
        default="plain",
        title="Term ID",
        description="Identifies this quantity as a directionless stand-in.",
    )
    name: str = Field(title="Name", description="Selector naming the quantity.")


class HigherIsBetterTerm(PlainTerm):
    """A term whose quantity is conventionally better when larger.

    Attributes:
        id (Literal["higher"]): Fixed identifier for this kind of quantity.
    """

    #: Larger values are the usual intent.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"

    id: Literal["higher"] = Field(  # type: ignore[assignment]
        default="higher",
        title="Term ID",
        description="Identifies this quantity as a better-when-larger stand-in.",
    )


class LowerIsBetterTerm(PlainTerm):
    """A term whose quantity is conventionally better when smaller.

    Attributes:
        id (Literal["lower"]): Fixed identifier for this kind of quantity.
    """

    #: Smaller values are the usual intent.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"

    id: Literal["lower"] = Field(  # type: ignore[assignment]
        default="lower",
        title="Term ID",
        description="Identifies this quantity as a better-when-smaller stand-in.",
    )


#: Columns the synthetic relaxed state is written in, sliced to whatever width a test uses.
VOCAB = ["A", "C", "G", "T"]


def relaxed_state(rows: int, columns: int) -> list[list[float]]:
    """Return the uniform relaxed state every synthetic term is differentiated at.

    Terms that share a shape share this state, so they also share a fingerprint and compose.
    Metadata describes the *input*, never the gradient, which is what makes two different
    quantities over one sequence compatible with each other.
    """
    return [[1.0 / columns] * columns for _ in range(rows)]


class ActivityTerm(DifferentiableTerm):
    """A parameterized quantity standing in for cell-line activity.

    Attributes:
        id (Literal["activity"]): Fixed identifier for this kind of quantity.
        cell_type (str): Cell line whose predicted activity is differentiated.
    """

    #: Larger values are the usual intent.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"

    id: Literal["activity"] = Field(
        default="activity",
        title="Term ID",
        description="Identifies this quantity as a cell-line activity prediction.",
    )
    cell_type: str = Field(title="Cell Type", description="Cell line whose activity is differentiated.")


class ConfidenceTerm(DifferentiableTerm):
    """A parameterless quantity standing in for a second, unrelated model.

    Attributes:
        id (Literal["confidence"]): Fixed identifier for this kind of quantity.
    """

    #: Larger values are the usual intent.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"

    id: Literal["confidence"] = Field(
        default="confidence",
        title="Term ID",
        description="Identifies this quantity as a whole-prediction confidence score.",
    )


def answer(
    value: float,
    gradient: list[list[float]] | None = None,
    name: str = "a",
    kind: type[DifferentiableTerm] = PlainTerm,
    metadata: GradientMetadata | None = None,
    term: DifferentiableTerm | None = None,
) -> TermGradient:
    """Build an answered term with a synthetic gradient over the shared relaxed state."""
    rows = len(gradient) if gradient else 1
    columns = len(gradient[0]) if gradient and gradient[0] else 1
    return TermGradient(
        term=term or kind(name=name),
        term_value=value,
        gradient=gradient,
        metadata=metadata
        or GradientMetadata.from_input(
            relaxed_state(rows, columns),
            vocab=VOCAB[:columns],
            temperature=1.0,
            description="synthetic quantity",
        ),
    )


# ── Goal resolution ───────────────────────────────────────────────────────────


def test_explicit_goal_beats_the_advisory_direction() -> None:
    """A caller's goal wins over whatever the quantity suggests."""
    assert isinstance(resolve_goal(answer(1.0, kind=HigherIsBetterTerm), MinimizeGoal()), MinimizeGoal)


def test_advisory_direction_supplies_a_goal_when_none_is_given() -> None:
    """A quantity that states which way is better does not need an explicit goal."""
    assert isinstance(resolve_goal(answer(1.0, kind=HigherIsBetterTerm), None), MaximizeGoal)
    assert isinstance(resolve_goal(answer(1.0, kind=LowerIsBetterTerm), None), MinimizeGoal)


def test_a_direction_shorthand_resolves_without_naming_a_goal_class() -> None:
    """``"minimize"`` and ``"maximize"`` stand in for the two parameterless goals."""
    assert isinstance(resolve_goal(answer(1.0), "minimize"), MinimizeGoal)
    assert isinstance(resolve_goal(answer(1.0), "maximize"), MaximizeGoal)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [(HigherIsBetterTerm, MaximizeGoal), (LowerIsBetterTerm, MinimizeGoal)],
    ids=["higher", "lower"],
)
def test_the_term_class_supplies_the_direction_when_metadata_is_absent(
    kind: type[PlainTerm], expected: type[Goal]
) -> None:
    """Composing one gradient without metadata is supported, so inference cannot depend on it."""
    bare = TermGradient(term=kind(name="a"), term_value=1.0, gradient=[[1.0]])

    assert bare.metadata is None
    assert isinstance(resolve_goal(bare, None), expected)


def test_a_directionless_term_without_metadata_still_demands_a_goal() -> None:
    """Falling back to the class does not invent a direction the class does not state."""
    bare = TermGradient(term=PlainTerm(name="a"), term_value=1.0, gradient=[[1.0]])

    with pytest.raises(ValueError, match="needs an explicit goal"):
        resolve_goal(bare, None)


def test_directionless_quantity_demands_an_explicit_goal() -> None:
    """A quantity with no inherent direction refuses to guess, and says what would fix it."""
    with pytest.raises(ValueError, match="needs an explicit goal"):
        resolve_goal(answer(1.0), None)


# ── Composition ───────────────────────────────────────────────────────────────


def test_gradient_is_the_chain_rule_applied_to_the_raw_gradient() -> None:
    """The composed gradient is d(loss)/d(value) times the term's own gradient."""
    objective = compose_terms([answer(3.0, [[1.0, -2.0]])], goals=[MinimizeGoal()])

    assert objective.objective_loss == pytest.approx(3.0)
    assert objective.gradient == [[1.0, -2.0]]

    maximized = compose_terms([answer(3.0, [[1.0, -2.0]])], goals=[MaximizeGoal()])
    assert maximized.objective_loss == pytest.approx(-3.0)
    assert maximized.gradient == [[-1.0, 2.0]]


def test_terms_sum_and_weights_scale_their_contributions() -> None:
    """Two terms add, and a weight multiplies both the loss and the gradient it contributes."""
    answers = [answer(2.0, [[1.0, 0.0]], name="a"), answer(5.0, [[0.0, 1.0]], name="b")]
    goals = [MinimizeGoal(), MinimizeGoal()]

    plain = compose_terms(answers, goals=goals)
    assert plain.objective_loss == pytest.approx(7.0)
    assert plain.gradient == [[1.0, 1.0]]

    weighted = compose_terms(answers, goals=goals, weights=[3.0, 1.0])
    assert weighted.objective_loss == pytest.approx(2.0 * 3.0 + 5.0)
    assert weighted.gradient == [[3.0, 1.0]]


def test_goals_and_weights_may_be_keyed_by_the_term_they_apply_to() -> None:
    """The mapping form says which quantity it means, rather than relying on position."""
    answers = [answer(2.0, [[1.0, 0.0]], name="a"), answer(5.0, [[0.0, 1.0]], name="b")]
    first, second = (a.term for a in answers)

    keyed = compose_terms(
        answers,
        goals={first: MinimizeGoal(), second: MinimizeGoal()},
        weights={first: 3.0},
    )
    positional = compose_terms(answers, goals=[MinimizeGoal(), MinimizeGoal()], weights=[3.0, 1.0])

    assert keyed.objective_loss == pytest.approx(positional.objective_loss)
    assert keyed.gradient == positional.gradient


def test_a_none_entry_in_a_sequence_takes_the_default() -> None:
    """The positional spelling of an omitted goal and an omitted weight."""
    answers = [
        answer(2.0, [[1.0, 0.0]], name="a"),
        answer(5.0, [[0.0, 1.0]], name="b", kind=LowerIsBetterTerm),
    ]

    objective = compose_terms(answers, goals=[MaximizeGoal(), None], weights=[2.0, None])

    assert [c.dloss_dvalue for c in objective.contributions] == [-1.0, 1.0], "explicit, then advisory"
    assert [c.weight for c in objective.contributions] == [2.0, 1.0]


def test_composed_gradient_is_the_gradient_of_the_composed_loss() -> None:
    """The invariant: ``ObjectiveResult.gradient`` differentiates ``ObjectiveResult.loss``.

    Each term's own gradient is the local linear model of its value, so perturbing the input by
    ``eps * u`` moves value ``i`` by ``eps * <g_i, u>``. Recomposing from the moved values gives a
    finite difference of the composed loss, which must equal the directional derivative of the
    composed gradient along ``u``. Any step that rescales a gradient without rescaling its loss
    breaks this, which is why gradient-magnitude balancing does not live here.
    """
    gradients = [[[1.0, -2.0], [0.5, 0.0]], [[0.0, 3.0], [-1.0, 2.0]]]
    values = [2.0, 5.0]
    goals = [SigmoidGoal(direction="maximize", center=2.0, scale=1.5), AtMostGoal(threshold=4.0)]
    weights = [2.0, 0.5]
    names = ["a", "b"]

    def compose_at(offsets: list[float]):
        """Compose with each value moved by its own offset."""
        moved = [answer(values[i] + offsets[i], gradients[i], name=names[i]) for i in range(2)]
        return compose_terms(moved, goals=goals, weights=weights)

    baseline = compose_at([0.0, 0.0])

    # Walk a direction with a different sign in every entry, so no component cancels by symmetry
    direction = [[0.7, -0.3], [0.2, 0.9]]
    epsilon = 1e-6

    # First order, each value moves by the inner product of its gradient with the direction
    deltas = [
        sum(g * d for row, drow in zip(grad, direction, strict=True) for g, d in zip(row, drow, strict=True))
        for grad in gradients
    ]
    forward = compose_at([epsilon * delta for delta in deltas])
    backward = compose_at([-epsilon * delta for delta in deltas])
    numeric = (forward.objective_loss - backward.objective_loss) / (2 * epsilon)

    analytic = sum(
        g * d for row, drow in zip(baseline.gradient, direction, strict=True) for g, d in zip(row, drow, strict=True)
    )
    assert numeric == pytest.approx(analytic, rel=1e-5, abs=1e-9)


def test_satisfied_constraint_contributes_nothing() -> None:
    """A satisfied constraint adds no loss and no gradient, leaving the active term to steer."""
    answers = [answer(1.0, [[5.0, 5.0]], name="satisfied"), answer(4.0, [[0.0, 2.0]], name="active")]

    objective = compose_terms(answers, goals=[AtMostGoal(threshold=2.0), MinimizeGoal()])

    assert objective.contributions[0].dloss_dvalue == 0.0
    assert objective.contributions[0].contribution_l2_norm == 0.0
    assert objective.gradient[0] == pytest.approx([0.0, 2.0])


def test_contributions_report_what_composition_did_with_each_term() -> None:
    """Diagnostics name the quantity and what came of it, not a restatement of the request."""
    objective = compose_terms([answer(3.0, [[1.0, -2.0]], name="a")], goals=[MaximizeGoal()], weights=[2.0])
    (contribution,) = objective.contributions

    assert contribution.term == PlainTerm(name="a"), "which quantity, as it was asked for"
    assert contribution.term_loss == pytest.approx(-3.0), "the goal's loss, before any weighting"
    assert contribution.dloss_dvalue == pytest.approx(-1.0)
    assert contribution.weight == pytest.approx(2.0)
    assert contribution.contribution_l2_norm == pytest.approx(2.0 * math.sqrt(5.0))


def test_the_three_quantities_stay_distinct_under_a_weight() -> None:
    """The three quantities never collapse into each other.

    ``term_value`` is what the model said, ``term_loss`` is what the goal made of it, and the
    objective is the weighted sum of those losses.
    """
    answers = [answer(2.0, [[1.0, 0.0]], name="a"), answer(5.0, [[0.0, 1.0]], name="b")]

    objective = compose_terms(answers, goals=[MaximizeGoal(), MinimizeGoal()], weights=[3.0, 0.5])
    first, second = objective.contributions

    assert [a.term_value for a in answers] == [2.0, 5.0], "raw, and read from the gradients"
    assert (first.term_loss, second.term_loss) == (-2.0, 5.0), "unweighted, as the goal made them"
    assert objective.objective_loss == pytest.approx(3.0 * -2.0 + 0.5 * 5.0)
    assert objective.gradient == [[-3.0, 0.5]], "the same weights carried through the chain rule"

    # The contributions carry no copy of the raw value; it is already on the gradient they describe
    assert not hasattr(first, "term_value")
    assert [c.term for c in objective.contributions] == [a.term for a in answers], "matched by term"


def test_a_contribution_serializes_the_concrete_term_with_its_selectors() -> None:
    """A base-typed field would publish only the id, losing which cell line was asked about."""
    objective = compose_terms(
        [answer(3.0, [[1.0, -2.0]], term=ActivityTerm(cell_type="HepG2"))],
        goals=[MaximizeGoal()],
    )
    payload = objective.model_dump()

    assert set(payload) == {"objective_loss", "gradient", "contributions"}
    assert payload["contributions"][0]["term"] == {"id": "activity", "cell_type": "HepG2"}
    assert set(payload["contributions"][0]) == {
        "term",
        "term_loss",
        "dloss_dvalue",
        "weight",
        "contribution_l2_norm",
    }
    assert json.loads(objective.model_dump_json())["contributions"][0]["term"]["cell_type"] == "HepG2"


def test_reading_a_dumped_contribution_back_fails_rather_than_dropping_selectors() -> None:
    """What ``SerializeAsAny`` does and does not buy.

    It is a serialization hint, not a polymorphic loader: validation still builds the declared
    base class, which forbids unknown fields. A dumped selector therefore comes back as a loud
    error rather than quietly disappearing. Restoring concrete terms needs a narrowed model of
    the tool's own, as ``TermGradient`` has; nothing here keeps a registry to guess from.
    """
    objective = compose_terms(
        [answer(3.0, [[1.0, -2.0]], term=ActivityTerm(cell_type="HepG2"))],
        goals=[MaximizeGoal()],
    )

    with pytest.raises(ValidationError, match="cell_type"):
        ObjectiveResult.model_validate_json(objective.model_dump_json())

    # A term carrying no selectors has nothing to lose, so it round trips unchanged
    plain = compose_terms([answer(0.8, [[1.0]], term=ConfidenceTerm())], goals=[MaximizeGoal()])
    restored = ObjectiveResult.model_validate_json(plain.model_dump_json())
    assert restored.objective_loss == pytest.approx(-0.8)
    assert restored.contributions[0].term.id == "confidence"


# ── What can be composed ──────────────────────────────────────────────────────


def test_a_single_term_gradient_needs_no_list() -> None:
    """One quantity is a perfectly good objective, and should not have to be wrapped."""
    objective = compose_terms(answer(3.0, [[1.0, -2.0]]), goals=[MinimizeGoal()])

    assert objective.objective_loss == pytest.approx(3.0)
    assert objective.gradient == [[1.0, -2.0]]


def test_a_result_contributes_every_term_it_carries() -> None:
    """Composing an evaluated item means composing what it returned, without naming each term."""
    result = GradientResult(
        loss=0.0,
        terms=[
            answer(2.0, [[1.0, 0.0]], name="a", kind=LowerIsBetterTerm),
            answer(5.0, [[0.0, 1.0]], name="b", kind=LowerIsBetterTerm),
        ],
    )

    objective = compose_terms(result)

    assert [c.term.name for c in objective.contributions] == ["a", "b"]
    assert objective.objective_loss == pytest.approx(7.0), "both advisory directions resolved to minimize"


def test_several_results_flatten_in_the_order_they_were_given() -> None:
    """Each result keeps its internal order, and the results keep theirs."""
    first = GradientResult(loss=0.0, terms=[answer(1.0, [[1.0, 0.0]], name="a", kind=LowerIsBetterTerm)])
    second = GradientResult(
        loss=0.0,
        terms=[
            answer(2.0, [[0.0, 1.0]], name="b", kind=LowerIsBetterTerm),
            answer(3.0, [[1.0, 1.0]], name="c", kind=LowerIsBetterTerm),
        ],
    )

    objective = compose_terms([second, first])

    assert [c.term.name for c in objective.contributions] == ["b", "c", "a"]


def test_terms_and_results_may_be_mixed() -> None:
    """A quantity named on its own composes beside everything another item returned."""
    loose = answer(1.0, [[1.0, 0.0]], name="loose", kind=LowerIsBetterTerm)
    result = GradientResult(loss=0.0, terms=[answer(2.0, [[0.0, 1.0]], name="held", kind=LowerIsBetterTerm)])

    objective = compose_terms([loose, result])

    assert [c.term.name for c in objective.contributions] == ["loose", "held"]
    assert objective.objective_loss == pytest.approx(3.0)


def test_two_models_compose_end_to_end_through_string_keys() -> None:
    """The cross-model case in full: two results, two term classes, goals named by id.

    Each model differentiated the same relaxed state, so their gradients are compatible however
    different the quantities are. The term left out of both dictionaries takes the defaults.
    """
    one_model = GradientResult(
        loss=0.0,
        terms=[answer(2.0, [[1.0, 0.0]], term=ActivityTerm(cell_type="HepG2"))],
    )
    another_model = GradientResult(
        loss=0.0,
        terms=[
            answer(0.8, [[0.0, 1.0]], term=ConfidenceTerm()),
            answer(4.0, [[1.0, 1.0]], name="extra", kind=LowerIsBetterTerm),
        ],
    )

    objective = compose_terms(
        [one_model, another_model],
        goals={"activity": MaximizeGoal(), "confidence": MinimizeGoal()},
        weights={"activity": 2.0, "confidence": 0.5},
    )

    assert [type(c.term).__name__ for c in objective.contributions] == [
        "ActivityTerm",
        "ConfidenceTerm",
        "LowerIsBetterTerm",
    ], "results flatten in the order given, each keeping its own order"

    activity, confidence, extra = objective.contributions
    assert (activity.term_loss, activity.weight) == (-2.0, 2.0)
    assert (confidence.term_loss, confidence.weight) == (0.8, 0.5)
    assert (extra.term_loss, extra.weight) == (4.0, 1.0), "unnamed: advisory direction at weight 1.0"

    assert objective.objective_loss == pytest.approx(2.0 * -2.0 + 0.5 * 0.8 + 1.0 * 4.0)
    #  2*(-1)*[[1,0]]  +  0.5*(1)*[[0,1]]  +  1*(1)*[[1,1]]
    assert objective.gradient[0] == pytest.approx([-1.0, 1.5])


# ── Naming terms ──────────────────────────────────────────────────────────────


def test_a_goal_may_be_keyed_by_a_term_id() -> None:
    """The short spelling, for a quantity only asked for once."""
    answers = [answer(2.0, [[1.0, 0.0]], name="a"), answer(0.8, [[0.0, 1.0]], term=ConfidenceTerm())]

    objective = compose_terms(answers, goals={"plain": MinimizeGoal(), "confidence": MaximizeGoal()})

    assert [c.dloss_dvalue for c in objective.contributions] == [1.0, -1.0]


def test_an_id_shared_by_two_terms_is_ambiguous() -> None:
    """Two quantities of one kind differ by their selectors, so the kind does not say which."""
    answers = [answer(2.0, [[1.0, 0.0]], name="a"), answer(5.0, [[0.0, 1.0]], name="b")]

    with pytest.raises(ValueError, match=r"goals key 'plain' matches 2 terms"):
        compose_terms(answers, goals={"plain": MinimizeGoal()})

    # Keying by the terms themselves is what tells them apart
    objective = compose_terms(answers, goals={answers[0].term: MinimizeGoal(), answers[1].term: MaximizeGoal()})
    assert [c.dloss_dvalue for c in objective.contributions] == [1.0, -1.0]


def test_an_id_naming_no_collected_term_is_rejected() -> None:
    """A typo lists the kinds that are actually present."""
    with pytest.raises(ValueError, match=r"goals key 'plddt' matches no collected term.*\['plain'\]"):
        compose_terms([answer(1.0, [[1.0]])], goals={"plddt": MinimizeGoal()})


def test_two_keys_naming_one_term_are_rejected() -> None:
    """An id and the term itself can name the same quantity; supplying both is a contradiction."""
    only = answer(0.8, [[1.0]], term=ConfidenceTerm())

    with pytest.raises(ValueError, match="two goals keys name the same term"):
        compose_terms([only], goals={"confidence": MinimizeGoal(), ConfidenceTerm(): MaximizeGoal()})


def test_a_key_that_is_neither_a_term_nor_an_id_is_a_type_error() -> None:
    """Keys name quantities; anything else is a programming mistake."""
    with pytest.raises(TypeError, match="goals keys must be a term or a term id"):
        compose_terms([answer(1.0, [[1.0]])], goals={42: MinimizeGoal()})  # type: ignore[dict-item]


def test_positional_goals_and_weights_are_refused_for_a_result() -> None:
    """The order of terms inside a result is not part of its contract, so it cannot be relied on."""
    result = GradientResult(loss=0.0, terms=[answer(1.0, [[1.0]], kind=LowerIsBetterTerm)])

    with pytest.raises(ValueError, match="goals must be a mapping when a GradientResult is composed"):
        compose_terms(result, goals=[MinimizeGoal()])
    with pytest.raises(ValueError, match="weights must be a mapping when a GradientResult is composed"):
        compose_terms(result, weights=[2.0])

    # Mixing a loose term in still counts as composing a result
    with pytest.raises(ValueError, match="goals must be a mapping"):
        compose_terms([answer(1.0, [[1.0]], name="loose", kind=LowerIsBetterTerm), result], goals=[None, None])


# ── Defaults ──────────────────────────────────────────────────────────────────


def test_every_default_applies_when_nothing_is_supplied() -> None:
    """No goals and no weights means advisory directions at weight 1.0, not an empty objective."""
    answers = [
        answer(2.0, [[1.0, 0.0]], name="a", kind=HigherIsBetterTerm),
        answer(5.0, [[0.0, 1.0]], name="b", kind=LowerIsBetterTerm),
    ]

    objective = compose_terms(answers)

    assert [c.weight for c in objective.contributions] == [1.0, 1.0]
    assert objective.objective_loss == pytest.approx(-2.0 + 5.0)
    assert objective.gradient == [[-1.0, 1.0]]


def test_a_mapping_leaves_unnamed_terms_on_their_defaults() -> None:
    """Naming one quantity is not a statement about the others, and never excludes them."""
    answers = [
        answer(2.0, [[1.0, 0.0]], name="a", kind=HigherIsBetterTerm),
        answer(5.0, [[0.0, 1.0]], name="b", kind=LowerIsBetterTerm),
    ]

    objective = compose_terms(answers, goals={answers[0].term: MinimizeGoal()}, weights={answers[1].term: 4.0})

    assert [c.dloss_dvalue for c in objective.contributions] == [1.0, 1.0], "explicit minimize, then advisory"
    assert [c.weight for c in objective.contributions] == [1.0, 4.0]
    assert len(objective.contributions) == 2, "an unnamed term still takes part"


def test_a_misspelled_direction_is_rejected_rather_than_inferred() -> None:
    """An explicit goal that is not one must fail, not quietly become the advisory default."""
    with pytest.raises(ValueError, match="'maximise' is not a goal"):
        compose_terms([answer(1.0, [[1.0]], kind=HigherIsBetterTerm)], goals=["maximise"])
    with pytest.raises(ValueError, match="'Minimize' is not a goal"):
        compose_terms([answer(1.0, [[1.0]], kind=HigherIsBetterTerm)], goals={"higher": "Minimize"})


# ── Coordinate compatibility ──────────────────────────────────────────────────


def _at(matrix: list[list[float]], **fields: object) -> GradientMetadata:
    """Describe a gradient taken over an explicit relaxed state."""
    defaults: dict[str, object] = {"vocab": VOCAB[: len(matrix[0])], "temperature": 1.0}
    return GradientMetadata.from_input(matrix, **(defaults | fields))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("label", "other", "reason"),
    [
        ("vocabulary", _at(relaxed_state(1, 2), vocab=["C", "A"]), "vocabularies differ"),
        ("positions", _at(relaxed_state(1, 2), positions=[7]), "different input positions"),
        ("coordinate space", _at(relaxed_state(1, 2), temperature=None), "softmax Jacobian"),
        ("input state", _at([[0.9, 0.1]]), "different points"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_gradients_that_are_not_about_the_same_thing_are_refused(
    label: str, other: GradientMetadata, reason: str
) -> None:
    """Matching dimensions prove nothing; each axis of the metadata is compared."""
    answers = [
        answer(1.0, [[1.0, 0.0]], name="a"),
        answer(2.0, [[0.0, 1.0]], name="b", metadata=other),
    ]

    with pytest.raises(ValueError, match=rf"cannot combine .*{reason}"):
        compose_terms(answers, goals=[MinimizeGoal(), MinimizeGoal()])


def test_a_gradient_without_metadata_cannot_be_combined() -> None:
    """Missing metadata is unverifiable, not presumed compatible."""
    unchecked = TermGradient(term=PlainTerm(name="b"), term_value=2.0, gradient=[[0.0, 1.0]])
    answers = [answer(1.0, [[1.0, 0.0]], name="a"), unchecked]

    with pytest.raises(ValueError, match="report no gradient metadata"):
        compose_terms(answers, goals=[MinimizeGoal(), MinimizeGoal()])


def test_one_gradient_alone_needs_no_metadata() -> None:
    """There is nothing to be compatible with, so a lone term composes as it always did."""
    alone = TermGradient(term=PlainTerm(name="a"), term_value=3.0, gradient=[[1.0, -2.0]])

    assert compose_terms([alone], goals=[MinimizeGoal()]).objective_loss == pytest.approx(3.0)


def test_an_earlier_step_may_be_reused_deliberately() -> None:
    """``strict`` is the opt-out for the one incompatibility that is an approximation."""
    answers = [
        answer(1.0, [[1.0, 0.0]], name="a"),
        answer(2.0, [[0.0, 1.0]], name="b", metadata=_at([[0.9, 0.1]])),
    ]
    goals = [MinimizeGoal(), MinimizeGoal()]

    with pytest.raises(ValueError, match="different points"):
        compose_terms(answers, goals=goals)
    assert compose_terms(answers, goals=goals, strict=False).gradient == [[1.0, 1.0]]


# ── Rejections ────────────────────────────────────────────────────────────────


def test_composition_requires_terms() -> None:
    """Composing nothing is a mistake, not an empty objective."""
    with pytest.raises(ValueError, match="no gradient terms"):
        compose_terms([])


def test_the_same_term_twice_is_rejected() -> None:
    """Two identical requests would make a goal or weight keyed by term ambiguous."""
    with pytest.raises(ValueError, match="the same term appears more than once"):
        compose_terms([answer(1.0, [[1.0]]), answer(2.0, [[1.0]])], goals=[MinimizeGoal(), MinimizeGoal()])


def test_unknown_goal_and_weight_keys_are_rejected() -> None:
    """A mapping naming a quantity this result does not carry fails loudly."""
    absent = PlainTerm(name="typo")
    with pytest.raises(ValueError, match=r"goals key .* matches none of the collected terms"):
        compose_terms([answer(1.0, [[1.0]])], goals={absent: MinimizeGoal()})
    with pytest.raises(ValueError, match=r"weights key .* matches none of the collected terms"):
        compose_terms([answer(1.0, [[1.0]])], goals=[MinimizeGoal()], weights={absent: 2.0})


def test_a_sequence_must_line_up_with_the_terms() -> None:
    """A short or long sequence is a silent misalignment waiting to happen."""
    answers = [answer(1.0, [[1.0]], name="a"), answer(2.0, [[1.0]], name="b")]
    with pytest.raises(ValueError, match="goals has 1 entries for 2 terms"):
        compose_terms(answers, goals=[MinimizeGoal()])
    with pytest.raises(ValueError, match="weights has 3 entries for 2 terms"):
        compose_terms(answers, goals=[MinimizeGoal(), MinimizeGoal()], weights=[1.0, 1.0, 1.0])


def test_negative_weight_is_rejected() -> None:
    """A negative weight would silently invert a goal the caller already stated."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        compose_terms([answer(1.0, [[1.0]])], goals=[MinimizeGoal()], weights=[-1.0])


def test_missing_gradient_is_rejected() -> None:
    """Forward-only terms have values but nothing to compose."""
    with pytest.raises(ValueError, match="carry no gradient"):
        compose_terms([answer(1.0)], goals=[MinimizeGoal()])


def test_an_objective_loss_that_overflows_is_rejected() -> None:
    """Every factor can be finite while the product is not, which the per-value checks miss."""
    with pytest.raises(ValueError, match="drove the objective loss non-finite"):
        compose_terms([answer(1e308, [[1.0]])], goals=[MinimizeGoal()], weights=[1e10])


def test_a_running_total_that_overflows_is_rejected() -> None:
    """Two finite weighted losses can still sum past what a float can hold."""
    answers = [answer(1e308, [[1.0]], name="a"), answer(1e308, [[1.0]], name="b")]

    with pytest.raises(ValueError, match="drove the objective loss non-finite"):
        compose_terms(answers, goals=[MinimizeGoal(), MinimizeGoal()])


def test_mismatched_gradient_shapes_are_rejected() -> None:
    """Terms must share a coordinate system; equal dimensions are not assumed."""
    answers = [answer(1.0, [[1.0, 2.0]], name="a"), answer(1.0, [[1.0, 2.0], [3.0, 4.0]], name="b")]
    with pytest.raises(ValueError, match="disagree in shape"):
        compose_terms(answers, goals=[MinimizeGoal(), MinimizeGoal()])


# ── Result-object methods ─────────────────────────────────────────────────────


def test_result_methods_delegate_to_the_shared_implementation() -> None:
    """``compose`` and ``apply_goal`` are discoverable entry points, not a second implementation."""
    result = GradientResult(loss=0.0, terms=[answer(3.0, [[1.0, -2.0]], kind=LowerIsBetterTerm)])

    composed = result.compose()
    assert composed.objective_loss == pytest.approx(3.0)
    assert composed.gradient == [[1.0, -2.0]]

    single = result.terms[0].apply_goal(MaximizeGoal())
    assert single.objective_loss == pytest.approx(-3.0)
    assert single.gradient == [[-1.0, 2.0]]
