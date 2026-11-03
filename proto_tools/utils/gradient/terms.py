"""proto_tools/utils/gradient/terms.py.

The quantities a gradient tool can differentiate, and what comes back for one.

A tool declares its quantities as :class:`DifferentiableTerm` subclasses and catalogues them by
name, so a caller can discover what is on offer and ask for it by name, class or instance.
:class:`TermGradient` is the answer to one such request: the term itself, its raw value, and the
raw derivative of that value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, TypeGuard, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from proto_tools.utils.gradient.metadata import GradientMetadata, gradient_l2_norm
from proto_tools.utils.tool_io import Directionality

if TYPE_CHECKING:
    from proto_tools.utils.gradient.goals import Goal, GoalDirection
    from proto_tools.utils.gradient.objectives import ObjectiveResult


class DifferentiableTerm(BaseModel):
    """One differentiable scalar quantity a tool can expose, and how to ask for it.

    A subclass pins ``id`` to the kind of quantity it is and declares the selectors that identify
    which one — a cell type, a track and a bin span — and nothing else. Optimization intent does
    not belong here: weights, directions to optimize in, loss shaping and normalization are all
    properties of an objective a caller composes, not of the quantity itself.

    A term is frozen and closed, so a selection cannot drift between being asked for and being
    reported, and a mistyped selector fails where it was written rather than quietly selecting
    something else.

    A concrete term declares its identifier as a one-value ``Literal`` defaulting to that value,
    matching the name it is catalogued under::

        class ActivityTerm(DifferentiableTerm):
            id: Literal["activity"] = Field(
                default="activity",
                title="Term ID",
                description="Identifies this quantity as a cell-line activity prediction.",
            )
            cell_type: str = Field(title="Cell Type", description="...")

    That makes a tool's terms a discriminated union, so a serialized term reads back as the class
    it came from rather than as the base — the difference between two parameterless quantities
    being distinguishable and both dumping to the same thing.

    Narrowing the annotation replaces the field, so this base's ``title`` and ``description`` do
    not carry over: a subclass restates them, or its published schema arrives with a generated
    title and no description at all.

    Attributes:
        id (str): Which kind of quantity this is, fixed by the class and shared by every instance
            of it: two activity terms selecting different cell types report the same ``id``. It
            identifies the type, not the request — the concrete class and its selectors are what
            say which quantity was asked for.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    # Which way this quantity is generally better; the gradient copies it into its metadata.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    # Units the quantity is measured in: a property of the quantity, copied into its metadata.
    UNITS: ClassVar[str | None] = None

    # What the quantity measures; copied into metadata the same way, so no caller states it.
    DESCRIPTION: ClassVar[str] = ""

    id: str = Field(
        title="Term ID",
        description="Which kind of quantity this is; fixed by the term class, shared by its instances.",
    )


#: A tool's differentiable quantities, by the name callers ask for them under.
TermCatalog = Mapping[str, type[DifferentiableTerm]]

#: The same, for a tool whose available quantities depend on another setting's value.
GroupedTermCatalog = Mapping[str, TermCatalog]


class TermGradient(BaseModel):
    """One differentiated scalar: the term that was asked for, and what came back for it.

    The invariant this model carries is ``gradient == d(term_value)/d(input)``. No optimization
    direction, weight, loss transform or normalization is folded in, so a caller applies those
    afterwards to as many gradients as they like without re-running the model.

    The term is held rather than inherited, which keeps the request and the answer separable: the
    quantity that comes back is the same object that was asked for, selectors included. Pydantic
    serializes against the declared type, so a tool whose terms carry selectors narrows this field
    to its own concrete term class — see :class:`GradientResult` for the matching narrowing.

    Attributes:
        term (DifferentiableTerm): The exact quantity that was requested, selectors included.
        term_value (float): The raw scalar the model produced for that quantity.
        gradient (list[list[float]] | None): Raw derivative of ``term_value`` with respect to the
            relaxed input, an ``(L, V)`` matrix matching its shape and column order, or ``None``
            when gradients were not requested.
        grad_l2_norm (float | None): Euclidean norm of the flattened ``gradient``, filled in
            automatically, or ``None`` when there is no gradient.
        metadata (GradientMetadata | None): Everything describing this gradient — what it
            measures, its units, its advisory direction, and the quantities that determine
            whether it can be combined with another. ``None`` on a tool that does not report them.
    """

    # An answer is built once and then read; only the derived fields below fill themselves in.
    model_config = ConfigDict(extra="forbid", validate_assignment=False, frozen=False)

    term: DifferentiableTerm = Field(
        title="Term",
        description="The differentiable quantity this gradient answers, selectors included.",
    )
    term_value: float = Field(
        title="Term Value",
        description="Raw scalar the model produced for this quantity.",
    )
    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Raw derivative of term_value w.r.t. the input; matches its shape and column order.",
    )
    grad_l2_norm: float | None = Field(
        default=None,
        title="Gradient L2 Norm",
        description="Euclidean norm of the flattened gradient; computed automatically when a gradient is present.",
    )
    metadata: GradientMetadata | None = Field(
        default=None,
        title="Gradient Metadata",
        description="Quantities determining whether this gradient can be combined with another.",
    )

    @model_validator(mode="after")
    def fill_derived_fields(self) -> Self:
        """Fill the gradient norm, and take the direction, units and description from the term."""
        if self.metadata is not None and self.metadata.suggested_optimization_direction == "context-dependent":
            # The term class declares the direction; metadata is where consumers read it.
            direction = type(self.term).SUGGESTED_OPTIMIZATION_DIRECTION
            if direction != "context-dependent":
                self.metadata = self.metadata.model_copy(update={"suggested_optimization_direction": direction})
        if self.metadata is not None and self.metadata.units is None and type(self.term).UNITS is not None:
            # Same reasoning as the direction: the quantity owns its units, not the caller.
            self.metadata = self.metadata.model_copy(update={"units": type(self.term).UNITS})
        if self.metadata is not None and not self.metadata.description and type(self.term).DESCRIPTION:
            # Same reasoning again: what the quantity measures is fixed by the quantity.
            self.metadata = self.metadata.model_copy(update={"description": type(self.term).DESCRIPTION})
        if self.gradient is not None and self.grad_l2_norm is None:
            self.grad_l2_norm = gradient_l2_norm(self.gradient)
        return self

    def apply_goal(self, goal: Goal | GoalDirection | None = None) -> ObjectiveResult:
        """Turn this one raw gradient into a minimizable objective.

        Args:
            goal (Goal | GoalDirection | None): What to ask of this quantity. Omit it to follow
                the advisory direction, which requires the quantity to have one.

        Returns:
            ObjectiveResult: The goal's loss at this term's value as ``objective_loss``, with the
                matching gradient.
        """
        # Imported here because composition builds on this module's models.
        from proto_tools.utils.gradient.objectives import compose_terms

        return compose_terms([self], goals=[goal])


def _is_term_class(value: object) -> TypeGuard[type[DifferentiableTerm]]:
    """Return whether a requested quantity was given as a term class rather than a name or instance.

    Args:
        value (object): One entry of a term request.

    Returns:
        TypeGuard[type[DifferentiableTerm]]: Whether the entry is a term class.
    """
    return isinstance(value, type) and issubclass(value, DifferentiableTerm)


def _term_id(term_class: type[DifferentiableTerm]) -> str:
    """Return the single value a term class pins its ``id`` to.

    Args:
        term_class (type[DifferentiableTerm]): The concrete term class to read.

    Returns:
        str: The identifier every instance of that class reports.

    Raises:
        TypeError: If ``id`` is absent, is not a ``Literal`` of exactly one string, or defaults to
            something other than that string.
    """
    info = term_class.model_fields.get("id")
    options = get_args(info.annotation) if info is not None else ()
    identifier = options[0] if len(options) == 1 else None
    if info is None or get_origin(info.annotation) is not Literal or not isinstance(identifier, str):
        raise TypeError(f'{term_class.__name__} must pin its id to one value, as id: Literal["<name>"] = "<name>"')
    if info.default != identifier:
        raise TypeError(f"{term_class.__name__} pins its id to {identifier!r} but defaults to {info.default!r}")
    return identifier


def _validate_term_catalog(catalog: TermCatalog | GroupedTermCatalog, *, owner: str) -> None:
    """Check that a catalog lists term classes whose pinned IDs match the names they are listed under.

    The name a quantity is catalogued under is the same string its class pins ``id`` to, which is
    what lets a caller ask for it by name and a serialized term read back as the right class. Two
    classes cannot collide on an identifier within one group, because each is checked against its
    own key; the same identifier appearing in different groups is normal, since a grouped catalog
    offers the same quantities for different settings.

    Args:
        catalog (TermCatalog | GroupedTermCatalog): The declared catalog, flat or grouped.
        owner (str): What to name in an error, such as ``"BorzoiGradientConfig.TERMS"``.

    Raises:
        TypeError: If the catalog mixes groups with terms, lists something that is not a term
            class, or lists a class whose pinned ID disagrees with its key.
    """
    groups = [name for name, entry in catalog.items() if isinstance(entry, Mapping)]
    if groups and len(groups) != len(catalog):
        loose = sorted(set(catalog) - set(groups))
        raise TypeError(f"{owner} mixes grouped and ungrouped entries; {loose} are not groups")
    if groups:
        for name in groups:
            _validate_term_catalog(cast(GroupedTermCatalog, catalog)[name], owner=f"{owner}[{name!r}]")
        return
    for name, entry in catalog.items():
        if not _is_term_class(entry):
            raise TypeError(f"{owner}[{name!r}] is {entry!r}, not a DifferentiableTerm subclass")
        identifier = _term_id(entry)
        if identifier != name:
            raise TypeError(f"{owner} lists {entry.__name__} under {name!r} but it pins its id to {identifier!r}")


def resolve_terms(
    values: Sequence[str | type[DifferentiableTerm] | DifferentiableTerm],
    catalog: TermCatalog,
) -> list[DifferentiableTerm]:
    """Resolve requested quantities to term instances against a tool's catalog.

    Three spellings are accepted, in increasing specificity: the name a quantity is catalogued
    under, the term class itself, and an instance. The first two only work for a quantity that
    takes no selectors — anything parameterized has to be instantiated, because there is no
    default cell type or track to guess at.

    Args:
        values (Sequence[str | type[DifferentiableTerm] | DifferentiableTerm]): Names, classes or
            instances, in the order they were asked for.
        catalog (TermCatalog): Flat ``name -> term class`` mapping for the tool, from
            :meth:`GradientConfig.supported_terms`.

    Returns:
        list[DifferentiableTerm]: One instance per request, in the order supplied.

    Every spelling is checked against the catalog, instances included: instantiating a term does
    not make another model's quantity askable of this one.

    Raises:
        ValueError: If the catalog is still grouped rather than filtered down to one setting, if a
            request names a quantity the catalog does not hold, or if a name or class names a
            quantity whose selectors have no defaults and so cannot be instantiated bare.
        TypeError: If a value is neither a name, a term class nor a term instance.
    """
    groups = sorted(name for name, entry in catalog.items() if not _is_term_class(entry))
    if groups:
        raise ValueError(
            f"catalog entries {groups} are groups rather than terms; filter it down with "
            "supported_terms(<setting>=...) before resolving"
        )
    supported = set(catalog.values())
    resolved: list[DifferentiableTerm] = []
    for value in values:
        if isinstance(value, DifferentiableTerm):
            if type(value) not in supported:
                raise ValueError(f"{type(value).__name__} is not one of this tool's terms: {sorted(catalog)}")
            resolved.append(value)
            continue
        if isinstance(value, str):
            if value not in catalog:
                raise ValueError(f"unknown term {value!r}; this tool supports {sorted(catalog)}")
            term_class: type[DifferentiableTerm] = catalog[value]
        elif _is_term_class(value):
            if value not in supported:
                raise ValueError(f"{value.__name__} is not one of this tool's terms: {sorted(catalog)}")
            term_class = value
        else:
            raise TypeError(f"a term must be a name, a term class or a term instance, got {value!r}")
        try:
            # A catalogued class pins its own id, so only selectors can be missing here.
            resolved.append(term_class())  # type: ignore[call-arg]
        except ValidationError as error:
            selectors = sorted(name for name, info in term_class.model_fields.items() if info.is_required())
            if not selectors:
                raise
            raise ValueError(
                f"{term_class.__name__} selects a quantity by {selectors}, so ask for it as "
                f"{term_class.__name__}({selectors[0]}=...) rather than by name or class alone"
            ) from error
    return resolved
