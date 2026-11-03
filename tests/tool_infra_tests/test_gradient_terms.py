"""tests/tool_infra_tests/test_gradient_terms.py.

Unit tests for the shared differentiable-term, gradient and metadata models.

Covers term declaration and immutability, the fixed identifier that makes a tool's terms a
discriminated union, the raw-derivative semantics of a returned gradient, automatic L2 norms, the
narrowing that keeps tool-specific selectors alive through serialization, term discovery and
resolution. Pure model validation: no tool execution, no environments, no GPU.
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import Field, ValidationError

from proto_tools.utils.base_config import ConfigField
from proto_tools.utils.gradient.metadata import GradientMetadata, gradient_l2_norm
from proto_tools.utils.gradient.models import GradientConfig, GradientOutput, GradientResult
from proto_tools.utils.gradient.terms import (
    DifferentiableTerm,
    GroupedTermCatalog,
    TermCatalog,
    TermGradient,
    _is_term_class,
    _term_id,
    _validate_term_catalog,
    resolve_terms,
)
from proto_tools.utils.tool_io import Directionality


class CellActivityTerm(DifferentiableTerm):
    """A single-selector term, standing in for cell-line activity selection.

    Attributes:
        id (Literal["activity"]): Fixed identifier for this kind of quantity.
        cell_type (str): Cell line whose predicted activity is differentiated.
    """

    #: Higher predicted activity is the usual intent for this stand-in quantity.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"

    id: Literal["activity"] = Field(
        default="activity",
        title="Term ID",
        description="Identifies this quantity as a cell-line activity prediction.",
    )
    cell_type: str = Field(title="Cell Type", description="Cell line whose activity is differentiated.")


class TrackActivityTerm(DifferentiableTerm):
    """A multi-selector term, standing in for track-and-bin selection.

    Attributes:
        id (Literal["track_activity"]): Fixed identifier for this kind of quantity.
        track (int): Output track index.
        bin_start (int): First output bin, 1-indexed and inclusive.
        bin_end (int): Last output bin, 1-indexed and inclusive.
    """

    id: Literal["track_activity"] = Field(
        default="track_activity",
        title="Term ID",
        description="Identifies this quantity as a binned output-track activity.",
    )
    track: int = Field(title="Track", description="Output track index.")
    bin_start: int = Field(title="Bin Start", description="First output bin, 1-indexed inclusive.")
    bin_end: int = Field(title="Bin End", description="Last output bin, 1-indexed inclusive.")


class ConfidenceTerm(DifferentiableTerm):
    """A term that takes no selectors, standing in for a whole-prediction confidence metric."""

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "higher"

    id: Literal["plddt"] = Field(
        default="plddt",
        title="Term ID",
        description="Identifies this quantity as a whole-prediction confidence score.",
    )


class ErrorTerm(DifferentiableTerm):
    """A term that takes no selectors and is better when smaller, in physical units."""

    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "lower"

    id: Literal["pae"] = Field(
        default="pae",
        title="Term ID",
        description="Identifies this quantity as a whole-prediction error estimate.",
    )


class CellTermGradient(TermGradient):
    """A gradient narrowed to cell-selection terms, so their selectors survive serialization.

    Attributes:
        term (CellActivityTerm): The cell-line quantity this gradient answers.
    """

    term: CellActivityTerm = Field(title="Term", description="Cell-line quantity this gradient answers.")


class CellGradientResult(GradientResult):
    """A per-item result narrowed to cell-selection gradients.

    Attributes:
        terms (list[CellTermGradient]): Cell-line gradients for this item, in the order requested.
    """

    terms: list[CellTermGradient] = Field(
        default_factory=list,
        title="Terms",
        description="Cell-line gradients for this item, in the order requested.",
    )


class CellGradientOutput(GradientOutput):
    """An output narrowed to cell-selection results.

    Attributes:
        results (list[CellGradientResult]): One narrowed result per input item.
    """

    results: list[CellGradientResult] = Field(
        title="Results",
        description="One narrowed result per input item, in input order.",
    )


#: What a tool offering both parameterless quantities declares, so each reads back as its own class.
PredictionTerm = Annotated[ConfidenceTerm | ErrorTerm, Field(discriminator="id")]


class PredictionTermGradient(TermGradient):
    """A gradient narrowed to a discriminated union of the two parameterless quantities.

    Attributes:
        term (PredictionTerm): Whichever confidence or error quantity this gradient answers.
    """

    term: PredictionTerm = Field(title="Term", description="Confidence or error quantity answered.")


class PredictionGradientResult(GradientResult):
    """A per-item result narrowed to the parameterless prediction quantities.

    Attributes:
        terms (list[PredictionTermGradient]): Gradients for this item, in the order requested.
    """

    terms: list[PredictionTermGradient] = Field(
        default_factory=list,
        title="Terms",
        description="Prediction-quantity gradients for this item, in the order requested.",
    )


class PredictionGradientOutput(GradientOutput):
    """An output narrowed to the parameterless prediction results.

    Attributes:
        results (list[PredictionGradientResult]): One narrowed result per input item.
    """

    results: list[PredictionGradientResult] = Field(
        title="Results",
        description="One narrowed result per input item, in input order.",
    )


class FixedCatalogConfig(GradientConfig):
    """A config whose differentiable quantities do not depend on any other setting."""

    TERMS: ClassVar[TermCatalog] = {"plddt": ConfidenceTerm, "pae": ErrorTerm}


class GroupedCatalogConfig(GradientConfig):
    """A config whose differentiable quantities depend on the species head in use.

    Attributes:
        species (Literal["human", "mouse", "rat"]): Which trained head the tracks belong to.
    """

    TERM_GROUP_FIELD: ClassVar[str | None] = "species"
    TERMS: ClassVar[GroupedTermCatalog] = {
        "human": {"track_activity": TrackActivityTerm},
        "mouse": {"track_activity": TrackActivityTerm},
    }

    species: Literal["human", "mouse", "rat"] = ConfigField(
        default="human",
        title="Species",
        description="Which trained head the output tracks belong to.",
    )


class UntypedGroupConfig(GradientConfig):
    """A config that groups on a field stating no permitted values of its own.

    Attributes:
        construct_type (str): Which construct the model was fitted for.
    """

    TERM_GROUP_FIELD: ClassVar[str | None] = "construct_type"
    TERMS: ClassVar[GroupedTermCatalog] = {
        "utr5": {"activity": CellActivityTerm},
        "utr3": {"activity": CellActivityTerm},
    }

    construct_type: str = ConfigField(
        default="utr5",
        title="Construct Type",
        description="Which construct the model was fitted for.",
    )


def _metadata(
    matrix: list[list[float]] | None = None,
    *,
    temperature: float | None = 1.0,
    positions: list[int] | None = None,
    vocab: tuple[str, ...] | str = ("A", "C", "G", "T"),
    **fields: object,
) -> GradientMetadata:
    """Describe a gradient over a uniform 60 nt state unless told otherwise."""
    return GradientMetadata.from_input(
        matrix if matrix is not None else [[0.25] * 4] * 60,
        vocab=vocab,
        temperature=temperature,
        positions=positions,
        **fields,  # type: ignore[arg-type]
    )


# ── Term declaration ──────────────────────────────────────────────────────────


def test_the_identifier_names_the_kind_not_the_request() -> None:
    """Every instance of a class reports the same ID; the selectors say which quantity it is."""
    hepg2 = CellActivityTerm(cell_type="HepG2")
    jurkat = CellActivityTerm(cell_type="Jurkat")

    assert hepg2.id == jurkat.id == "activity"
    assert hepg2 != jurkat, "the selectors, not the ID, distinguish the two requests"


def test_a_term_cannot_be_built_or_mutated_to_another_kind() -> None:
    """A one-value Literal plus a frozen model pins the kind at both ends."""
    with pytest.raises(ValidationError):
        CellActivityTerm(id="pae", cell_type="K562")
    with pytest.raises(ValidationError):
        CellActivityTerm(cell_type="K562").id = "pae"


def test_terms_reject_unknown_selectors_and_mutation() -> None:
    """Terms are closed and frozen: a typo fails loudly, and a requested term cannot drift."""
    with pytest.raises(ValidationError):
        CellActivityTerm(cell_type="K562", cel_type="typo")
    with pytest.raises(ValidationError):
        CellActivityTerm(cell_type="K562").cell_type = "HepG2"


def test_a_caller_cannot_state_the_optimization_direction() -> None:
    """It is class-level, so selecting a quantity cannot claim which way is better."""
    with pytest.raises(ValidationError):
        CellActivityTerm(cell_type="K562", suggested_optimization_direction="lower")  # type: ignore[call-arg]


# ── Raw derivative semantics ──────────────────────────────────────────────────


def test_a_gradient_reports_the_raw_value_and_derivative_unchanged() -> None:
    """Nothing is folded in: a quantity that is better when smaller still reports what it measured."""
    answer = TermGradient(
        term=ErrorTerm(),
        term_value=4.75,
        gradient=[[0.5, -0.25]],
        metadata=_metadata(vocab=("A", "C"), matrix=[[0.5, 0.5]], units="angstroms"),
    )

    assert answer.term_value == 4.75, "no direction, weight or transform touches the value"
    assert answer.gradient == [[0.5, -0.25]], "and none touches the sign of the derivative"
    assert answer.metadata is not None
    assert answer.metadata.suggested_optimization_direction == "lower", "advisory only"
    assert answer.metadata.units == "angstroms"


def test_the_term_class_supplies_the_advisory_direction() -> None:
    """A tool builds metadata once; the direction comes from whichever term was asked for."""
    plain = _metadata()
    assert plain.suggested_optimization_direction == "context-dependent"

    assert (
        TermGradient(term=ConfidenceTerm(), term_value=0.9, metadata=plain).metadata.suggested_optimization_direction
        == "higher"
    )
    assert (
        TermGradient(term=ErrorTerm(), term_value=0.9, metadata=plain).metadata.suggested_optimization_direction
        == "lower"
    )


def test_a_tool_may_state_a_direction_its_term_class_does_not_declare() -> None:
    """The term's declaration is a default; a tool can set the direction on the metadata itself."""
    answer = TermGradient(
        term=TrackActivityTerm(track=0, bin_start=1, bin_end=8),
        term_value=1.0,
        metadata=_metadata().model_copy(update={"suggested_optimization_direction": "lower"}),
    )
    assert answer.metadata.suggested_optimization_direction == "lower"


# ── Gradient norms ────────────────────────────────────────────────────────────


def test_norm_flattens_rather_than_taking_a_spectral_norm() -> None:
    """The identity matrix separates the two: Frobenius is sqrt(2), spectral is 1."""
    assert gradient_l2_norm([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(math.sqrt(2.0))


def test_norm_is_filled_in_from_the_gradient() -> None:
    """A gradient reports its norm without the tool computing one."""
    answer = TermGradient(term=ConfidenceTerm(), term_value=2.0, gradient=[[3.0, 4.0], [0.0, 0.0]])
    assert answer.grad_l2_norm == pytest.approx(5.0)


def test_norm_stays_none_without_a_gradient() -> None:
    """Forward-only results carry a value and no norm."""
    answer = TermGradient(term=ConfidenceTerm(), term_value=2.0)
    assert answer.gradient is None
    assert answer.grad_l2_norm is None


# ── Serialization ─────────────────────────────────────────────────────────────


def test_a_narrowed_gradient_round_trips_with_its_selectors_intact() -> None:
    """A tool-specific gradient preserves its concrete term through a round trip."""
    original = CellTermGradient(
        term=CellActivityTerm(cell_type="K562"),
        term_value=1.5,
        gradient=[[1.0, 0.0]],
        metadata=_metadata(vocab=("A", "C"), matrix=[[0.5, 0.5]], description="predicted activity"),
    )
    restored = CellTermGradient.model_validate(original.model_dump())

    assert restored == original
    assert restored.term == CellActivityTerm(cell_type="K562")
    assert restored.term.cell_type == "K562"


def test_a_base_typed_term_keeps_its_kind_but_loses_its_selectors() -> None:
    """Pydantic serializes against the declared type: the ID is declared there, the selectors are not."""
    dumped = GradientResult(
        loss=0.0,
        terms=[CellTermGradient(term=CellActivityTerm(cell_type="K562"), term_value=1.0)],
    ).model_dump()

    assert dumped["terms"][0]["term"] == {"id": "activity"}, "the kind survives, the selectors do not"

    restored = CellGradientResult.model_validate(
        {"loss": 0.0, "terms": [{"term": {"cell_type": "K562"}, "term_value": 1.0}]}
    )
    assert restored.terms[0].term.cell_type == "K562", "narrowing is what preserves selectors"


def test_the_full_output_nesting_round_trips() -> None:
    """``GradientOutput -> GradientResult -> TermGradient -> DifferentiableTerm``, narrowed throughout."""
    output = CellGradientOutput(
        vocab=["A", "C", "G", "T"],
        results=[
            CellGradientResult(
                loss=0.0,
                terms=[
                    CellTermGradient(
                        term=CellActivityTerm(cell_type="K562"),
                        term_value=1.0,
                        gradient=[[1.0, 0.0, 0.0, 0.0]],
                        metadata=_metadata(matrix=[[0.25] * 4]),
                    )
                ],
            )
        ],
    )
    restored = CellGradientOutput.model_validate(output.model_dump())

    assert restored == output
    answer = restored.results[0].terms[0]
    assert answer.term.cell_type == "K562"
    assert answer.grad_l2_norm == pytest.approx(1.0)
    assert answer.metadata is not None and answer.metadata.length == 1


def test_requested_order_is_the_returned_order() -> None:
    """Terms come back in the order they were asked for, which is how a caller lines them up."""
    requested = [CellActivityTerm(cell_type=name) for name in ("HepG2", "K562", "Jurkat")]
    result = CellGradientResult(
        loss=0.0,
        terms=[CellTermGradient(term=term, term_value=float(index)) for index, term in enumerate(requested)],
    )

    assert [answer.term for answer in result.terms] == requested
    restored = CellGradientResult.model_validate(result.model_dump())
    assert [answer.term.cell_type for answer in restored.terms] == ["HepG2", "K562", "Jurkat"]


def test_an_unmigrated_result_still_validates() -> None:
    """``terms`` is empty on a tool that still reports one pre-composed objective."""
    unmigrated = GradientResult(loss=0.5, gradient=[[0.1, 0.2]])
    assert unmigrated.terms == []


# ── The ID contract ───────────────────────────────────────────────────────────


def _catalogs() -> list[tuple[str, object]]:
    """Every catalog declared by a ``GradientConfig`` subclass that is loaded, synthetic or real."""
    found: list[tuple[str, object]] = []
    stack = [GradientConfig]
    while stack:
        config = stack.pop()
        stack.extend(config.__subclasses__())
        if config.TERMS:
            found.append((f"{config.__name__}.TERMS", config.TERMS))
    return found


@pytest.mark.parametrize(("owner", "catalog"), _catalogs(), ids=lambda value: value if isinstance(value, str) else "")
def test_every_catalogued_term_publishes_field_metadata(owner: str, catalog: object) -> None:
    """Narrowing ``id`` replaces the base's field, so each term restates its title and description.

    Without that, a term nested in a registered tool's output publishes a generated ``"Id"`` title
    and no description at all, which the tool schema metadata check refuses.
    """
    groups = [catalog] if all(_is_term_class(entry) for entry in catalog.values()) else list(catalog.values())  # type: ignore[union-attr]
    for group in groups:
        for name, term_class in group.items():
            properties = term_class.model_json_schema()["properties"]
            for field, schema in properties.items():
                assert schema.get("title"), f"{owner}[{name!r}] field {field!r} publishes no title"
                assert schema.get("description"), f"{owner}[{name!r}] field {field!r} publishes no description"
            assert properties["id"]["title"] != "Id", (
                f"{owner}[{name!r}] left id with Pydantic's generated title; restate it with "
                "Field(default=..., title=..., description=...)"
            )


def test_a_class_without_a_pinned_identifier_is_refused() -> None:
    """The base's plain ``str`` is not a discriminator, so a subclass has to narrow it."""

    class Unpinned(DifferentiableTerm):
        """A term that forgot to pin its kind."""

    with pytest.raises(TypeError, match="must pin its id to one value"):
        _term_id(Unpinned)


def test_an_identifier_permitting_several_values_is_refused() -> None:
    """Two values is not a discriminator: a dump could not say which class to read back."""

    class Ambiguous(DifferentiableTerm):
        """A term whose kind is not decided.

        Attributes:
            id (Literal["one", "two"]): Two kinds, which is one too many.
        """

        id: Literal["one", "two"] = "one"

    with pytest.raises(TypeError, match="must pin its id to one value"):
        _term_id(Ambiguous)


def test_a_default_disagreeing_with_the_literal_is_refused() -> None:
    """A term must actually report the kind it declares."""

    class Mismatched(DifferentiableTerm):
        """A term whose default contradicts its annotation.

        Attributes:
            id (Literal["stated"]): The kind it claims but does not report.
        """

        id: Literal["stated"] = "unstated"  # type: ignore[assignment]

    with pytest.raises(TypeError, match=r"pins its id to 'stated' but defaults to 'unstated'"):
        _term_id(Mismatched)


def test_a_catalog_key_disagreeing_with_the_class_is_refused_as_it_is_declared() -> None:
    """A tool cannot list a quantity under a name it does not answer to."""
    with pytest.raises(TypeError, match=r"lists ConfidenceTerm under 'pae' but it pins its id to 'plddt'"):

        class MislabelledConfig(GradientConfig):
            """A config whose catalog key contradicts the class it points at."""

            TERMS: ClassVar[TermCatalog] = {"pae": ConfidenceTerm}


def test_a_catalog_of_something_other_than_terms_is_refused() -> None:
    """Only term classes can be catalogued, so a stray value fails where it was written."""
    with pytest.raises(TypeError, match="not a DifferentiableTerm subclass"):

        class NotTermsConfig(GradientConfig):
            """A config cataloguing something that is not a term."""

            TERMS: ClassVar[TermCatalog] = {"plddt": str}  # type: ignore[dict-item]


def test_a_catalog_mixing_groups_with_terms_is_refused() -> None:
    """Half-grouped is neither shape, and discovery could not say which it is."""
    with pytest.raises(TypeError, match="mixes grouped and ungrouped entries"):
        _validate_term_catalog({"human": {"plddt": ConfidenceTerm}, "pae": ErrorTerm}, owner="Mixed.TERMS")


def test_a_grouped_catalog_is_checked_group_by_group() -> None:
    """The contract holds inside each group, and the error says which group broke it."""
    with pytest.raises(TypeError, match=r"TERMS\['mouse'\] lists ConfidenceTerm under 'pae'"):
        _validate_term_catalog(
            {"human": {"plddt": ConfidenceTerm}, "mouse": {"pae": ConfidenceTerm}},
            owner="TERMS",
        )


# ── Discriminated serialization ───────────────────────────────────────────────


def test_two_parameterless_quantities_read_back_as_their_own_classes() -> None:
    """The case the identifier exists for: without it both would dump to the same empty object."""
    output = PredictionGradientOutput(
        vocab=["A", "C", "G", "T"],
        results=[
            PredictionGradientResult(
                loss=0.0,
                terms=[
                    PredictionTermGradient(
                        term=ConfidenceTerm(),
                        term_value=0.87,
                        gradient=[[0.1, 0.0, 0.0, 0.0]],
                        metadata=_metadata(matrix=[[0.25] * 4], description="mean pLDDT"),
                    ),
                    PredictionTermGradient(
                        term=ErrorTerm(),
                        term_value=4.75,
                        gradient=[[0.0, -0.2, 0.0, 0.0]],
                        metadata=_metadata(matrix=[[0.25] * 4], description="mean PAE", units="angstroms"),
                    ),
                ],
            )
        ],
    )

    restored = PredictionGradientOutput.model_validate_json(output.model_dump_json())

    confidence, error = restored.results[0].terms
    assert type(confidence.term) is ConfidenceTerm, "a dumped term must read back as its own class"
    assert type(error.term) is ErrorTerm
    assert (confidence.term_value, error.term_value) == (0.87, 4.75)
    assert confidence.gradient == [[0.1, 0.0, 0.0, 0.0]]
    assert error.gradient == [[0.0, -0.2, 0.0, 0.0]]
    assert confidence.grad_l2_norm == pytest.approx(0.1)
    assert confidence.metadata is not None and confidence.metadata.description == "mean pLDDT"
    assert error.metadata is not None and error.metadata.units == "angstroms"
    assert confidence.metadata.suggested_optimization_direction == "higher"
    assert error.metadata.suggested_optimization_direction == "lower"
    assert restored == output


def test_a_parameterized_term_keeps_its_selectors_through_json() -> None:
    """A selector-carrying quantity survives the same round trip, identifier and all."""
    output = CellGradientOutput(
        vocab=["A", "C", "G", "T"],
        results=[
            CellGradientResult(
                loss=0.0,
                terms=[
                    CellTermGradient(term=CellActivityTerm(cell_type=name), term_value=float(index))
                    for index, name in enumerate(("HepG2", "Jurkat"))
                ],
            )
        ],
    )

    payload = output.model_dump()
    assert payload["results"][0]["terms"][0]["term"] == {"id": "activity", "cell_type": "HepG2"}

    restored = CellGradientOutput.model_validate_json(output.model_dump_json())
    assert [answer.term.cell_type for answer in restored.results[0].terms] == ["HepG2", "Jurkat"]
    assert {answer.term.id for answer in restored.results[0].terms} == {"activity"}
    assert restored == output


# ── Discovery ─────────────────────────────────────────────────────────────────


def test_a_fixed_catalog_is_discoverable_without_building_a_config() -> None:
    """``supported_terms()`` is class-level, so a caller can ask before configuring anything."""
    assert FixedCatalogConfig.supported_terms() == {"plddt": ConfidenceTerm, "pae": ErrorTerm}


def test_a_catalog_names_term_classes_so_selectors_come_from_their_fields() -> None:
    """Parameterized quantities are described by their fields, not enumerated combination by combination."""
    catalog = GroupedCatalogConfig.supported_terms(species="human")
    assert set(catalog["track_activity"].model_fields) == {"id", "track", "bin_start", "bin_end"}


def test_a_grouped_catalog_is_nested_until_it_is_filtered() -> None:
    """Asked without a filter it shows every group; asked with one it flattens to that group."""
    assert GroupedCatalogConfig.supported_terms() == {
        "human": {"track_activity": TrackActivityTerm},
        "mouse": {"track_activity": TrackActivityTerm},
    }
    assert GroupedCatalogConfig.supported_terms(species="human") == {"track_activity": TrackActivityTerm}


def test_a_flat_catalog_takes_no_filters() -> None:
    """Filtering a tool whose quantities do not depend on a setting is a mistake, not a no-op."""
    with pytest.raises(ValueError, match="takes no filters"):
        FixedCatalogConfig.supported_terms(species="human")


def test_discovery_filters_on_the_grouping_field_alone() -> None:
    """Another field's name, or two at once, says the caller expects grouping that does not exist."""
    with pytest.raises(ValueError, match="filters on 'species' alone"):
        GroupedCatalogConfig.supported_terms(construct_type="utr5")
    with pytest.raises(ValueError, match="filters on 'species' alone"):
        GroupedCatalogConfig.supported_terms(species="human", construct_type="utr5")


def test_an_unsupported_group_value_reports_what_the_field_permits() -> None:
    """The config field is the source of truth for which values are legal."""
    with pytest.raises(ValueError, match=r"unknown species 'fly'.*\['human', 'mouse', 'rat'\]"):
        GroupedCatalogConfig.supported_terms(species="fly")


def test_a_permitted_group_with_no_terms_is_reported_as_such() -> None:
    """A legal setting the tool declares nothing for is a different failure from an illegal one."""
    with pytest.raises(ValueError, match="no differentiable terms for species='rat'"):
        GroupedCatalogConfig.supported_terms(species="rat")


def test_discovery_hands_back_a_copy_of_the_catalog() -> None:
    """Editing what discovery returned must not edit the tool's own declaration."""
    FixedCatalogConfig.supported_terms().clear()
    assert FixedCatalogConfig.supported_terms() == {"plddt": ConfidenceTerm, "pae": ErrorTerm}

    grouped = GroupedCatalogConfig.supported_terms()
    assert isinstance(grouped["human"], dict)
    grouped["human"].clear()
    grouped.clear()
    assert GroupedCatalogConfig.supported_terms(species="human") == {"track_activity": TrackActivityTerm}

    GroupedCatalogConfig.supported_terms(species="human").clear()
    assert GroupedCatalogConfig.supported_terms(species="human") == {"track_activity": TrackActivityTerm}


def test_a_group_field_stating_no_values_falls_back_to_the_catalog() -> None:
    """Where the schema permits anything, the declared groups are what can be asked for."""
    assert UntypedGroupConfig.supported_terms(construct_type="utr3") == {"activity": CellActivityTerm}
    with pytest.raises(ValueError, match=r"unknown construct_type 'utr9'.*\['utr3', 'utr5'\]"):
        UntypedGroupConfig.supported_terms(construct_type="utr9")


# ── Resolution ────────────────────────────────────────────────────────────────


def test_terms_resolve_from_names_classes_and_instances_in_order() -> None:
    """Three spellings, increasing in specificity, resolving to instances in the order supplied."""
    instance = CellActivityTerm(cell_type="K562")
    catalog = {"plddt": ConfidenceTerm, "pae": ErrorTerm, "activity": CellActivityTerm}

    resolved = resolve_terms(["pae", ConfidenceTerm, instance], catalog)

    assert resolved == [ErrorTerm(), ConfidenceTerm(), instance]
    assert resolved[2] is instance, "an instance is passed through as given"


def test_an_unknown_name_lists_what_the_tool_supports() -> None:
    """A typo fails against the catalog rather than resolving to nothing."""
    with pytest.raises(ValueError, match=r"unknown term 'plddtt'.*\['pae', 'plddt'\]"):
        resolve_terms(["plddtt"], {"plddt": ConfidenceTerm, "pae": ErrorTerm})


def test_a_class_from_another_tool_is_refused() -> None:
    """A term class is only askable where the tool catalogues it."""
    with pytest.raises(ValueError, match="CellActivityTerm is not one of this tool's terms"):
        resolve_terms([CellActivityTerm], {"plddt": ConfidenceTerm})


def test_an_instance_from_another_tool_is_refused() -> None:
    """Instantiating a quantity does not make another model's term askable of this one."""
    with pytest.raises(ValueError, match="CellActivityTerm is not one of this tool's terms"):
        resolve_terms([CellActivityTerm(cell_type="HepG2")], {"plddt": ConfidenceTerm})


def test_a_grouped_catalog_must_be_filtered_before_resolving() -> None:
    """Resolution works against one setting's terms, so a nested catalog is a caller mistake."""
    with pytest.raises(ValueError, match=r"catalog entries \['human', 'mouse'\] are groups"):
        resolve_terms([ConfidenceTerm], GroupedCatalogConfig.supported_terms())  # type: ignore[arg-type]


def test_a_parameterized_term_must_be_asked_for_as_an_instance() -> None:
    """There is no default track or bin span to guess at, so the error says to pass one."""
    with pytest.raises(
        ValueError, match=r"TrackActivityTerm selects a quantity by \['bin_end', 'bin_start', 'track'\]"
    ):
        resolve_terms(["track_activity"], {"track_activity": TrackActivityTerm})
    with pytest.raises(ValueError, match="rather than by name or class alone"):
        resolve_terms([TrackActivityTerm], {"track_activity": TrackActivityTerm})


def test_a_value_that_is_not_a_term_at_all_is_a_type_error() -> None:
    """Resolution covers three spellings; anything else is a programming mistake."""
    with pytest.raises(TypeError, match="must be a name, a term class or a term instance"):
        resolve_terms([42], {"plddt": ConfidenceTerm})  # type: ignore[list-item]


# ── Metadata ──────────────────────────────────────────────────────────────────


def test_coordinate_space_follows_the_temperature_contract() -> None:
    """A positive temperature differentiates logits; ``None`` differentiates probabilities."""
    assert _metadata(temperature=1.0).coordinate_space == "logits"
    assert _metadata(temperature=None).coordinate_space == "probabilities"


def test_positions_catch_identical_values_at_different_loci() -> None:
    """The case a fingerprint cannot catch: one window's gradient summed onto another's.

    Two windows of the same context, both uniformly initialized, hold byte-identical matrices and
    therefore share a fingerprint. Only the row positions distinguish them, which is why the
    metadata compares both rather than treating the fingerprint as sufficient.
    """
    here = _metadata(positions=list(range(1000, 1060)))
    there = _metadata(positions=list(range(2000, 2060)))

    assert here.input_fingerprint == there.input_fingerprint, "identical values, as a design run starts"
    assert "different input positions" in (here.incompatibility_with(there) or "")


def test_contiguous_rows_compare_equal_to_their_explicit_form() -> None:
    """``None`` means ``1..length``, so the two spellings of the same rows agree."""
    assert _metadata().incompatibility_with(_metadata(positions=list(range(1, 61)))) is None
    assert "different input positions" in (
        _metadata().incompatibility_with(_metadata(positions=list(range(2, 62)))) or ""
    )


def test_vocabulary_order_is_part_of_the_metadata() -> None:
    """Two tools may use the same columns in a different order; the sum would scramble them."""
    reason = _metadata(vocab=("A", "C", "G", "T")).incompatibility_with(_metadata(vocab=("A", "C", "T", "G")))
    assert "vocabularies differ" in (reason or "")


def test_a_different_point_is_strict_only() -> None:
    """Reusing an earlier step's gradient is an approximation, not nonsense, so strict governs it."""
    start = _metadata()
    moved = _metadata([[0.30] * 4] * 60)

    assert "different points" in (start.incompatibility_with(moved) or "")
    assert start.incompatibility_with(moved, strict=False) is None, "an opt-out, for deliberate reuse"


def test_relaxation_is_recorded_but_never_blocks() -> None:
    """A straight-through surrogate still points downhill, so it is reported rather than refused."""
    exact = GradientMetadata.from_input([[0.25] * 4] * 60, vocab="ACGT", temperature=1.0, hard=0.0)
    surrogate = GradientMetadata.from_input([[0.25] * 4] * 60, vocab="ACGT", temperature=1.0, hard=1.0)

    assert surrogate.hard == 1.0
    assert exact.incompatibility_with(surrogate) is None
