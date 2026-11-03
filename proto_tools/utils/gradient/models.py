"""proto_tools/utils/gradient/models.py.

Shared input and output contracts for relaxed-sequence gradient tools.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, get_args

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.utils.base_config import BaseConfig, ConfigField
from proto_tools.utils.gradient.metadata import GradientMetadata
from proto_tools.utils.gradient.terms import (
    DifferentiableTerm,
    GroupedTermCatalog,
    TermCatalog,
    TermGradient,
    _is_term_class,
    _validate_term_catalog,
)
from proto_tools.utils.sequence_logits import (
    NucleotideSequenceLogits,
    ProteinSequenceLogits,
    SequenceLogits,
    iter_sequence_logits,
)
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField

if TYPE_CHECKING:
    from proto_tools.utils.gradient.objectives import GoalsArgument, ObjectiveResult, WeightsArgument


def coerce_to_item_list(value: Any) -> Any:
    """Wrap one supplied gradient item in a list, leaving a list of items alone.

    A caller evaluating a single candidate should not have to type a list, and a caller
    evaluating many should not have to wrap each one. One matrix is recognized by its rows being
    numbers; a list whose first element is itself a matrix, a model or a mapping is already a
    collection of items. Ambiguity is not guessed at: anything else is handed to Pydantic as-is
    so it reports the real type error.

    Args:
        value (Any): Whatever was supplied for the ``inputs`` field.

    Returns:
        Any: A list of items, or the value unchanged when it is not recognizable.

    Raises:
        ValueError: If an empty list was supplied; there is nothing to evaluate.
    """
    if isinstance(value, SequenceLogits | BaseModel | Mapping):
        return [value]
    if isinstance(value, list):
        if not value:
            raise ValueError("inputs must contain at least one item")
        first = value[0]
        if isinstance(first, list | tuple) and first and isinstance(first[0], int | float):
            return [value]  # one (L, V) matrix supplied directly
    return value


class RelaxationConfig(BaseConfig):
    """Blend coefficients controlling how a relaxed sequence reaches the model.

    These are scalars, not tensors: they weight the terms of
    :func:`standalone_helpers.relaxation.relaxed_sequence`. ``temperature`` lives on the Input
    because it describes the supplied numbers; relaxation is a property of the run.

    Both coefficients are applied the way ColabDesign applies them, one after the other, at
    https://github.com/sokrypton/ColabDesign/blob/5ab4efaba2321a6c3c314b82d2fff8e0241f5c2d/colabdesign/shared/model.py#L205-L206
    (the revision ``alphafold2/standalone/requirements.txt`` pins)::

        seq["pseudo"] = opt["soft"] * seq["soft"] + (1 - opt["soft"]) * seq["input"]
        seq["pseudo"] = opt["hard"] * seq["hard"] + (1 - opt["hard"]) * seq["pseudo"]

    Attributes:
        hard (float): Straight-through coefficient; 1 forwards a hard one-hot to the model.

            At ``0.0`` the model sees the relaxed blend itself. At ``1.0`` the forward pass sees
            a discrete one-hot while the backward pass still flows through the probabilities, so
            the gradient describes a sequence the model scored exactly as it would score a real
            one. Values in between interpolate between the two.

            This is the setting to reach for when the goal is to evaluate a *discrete* sequence;
            lowering ``soft`` is not.
        soft (float): Blend from the raw input matrix (0) to softmax probabilities (1).

            It is recommended to leave ``soft`` at ``1.0`` in almost all use cases. That feeds
            softmax-normalized probabilities to the model — a differentiable relaxation of a
            valid sequence representation. Values below ``1.0`` blend those probabilities with
            the raw input logits, which can produce inputs outside the model's expected
            sequence representation and alter gradient behaviour.

            The setting is left exposed for research purposes and for parity with the
            ColabDesign implementation linked above.
    """

    soft: float = ConfigField(
        default=1.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="Soft Mixing",
        description="Blend raw input (0) toward softmax probabilities (1); 1.0 recommended for almost all uses.",
    )
    hard: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="Hard Mixing",
        description="Straight-through coefficient; 1 forwards a hard one-hot to the model.",
    )


class GradientInput(BaseToolInput):
    """Shared contract for one relaxed-sequence gradient evaluation step.

    Subclasses declare the relaxed state they consume — a vocabulary-specific
    :class:`~proto_tools.utils.sequence_logits.SequenceLogits` field, or a composite model
    exposing its own ``iter_sequence_logits()`` — and this base owns the temperature semantics
    and the gradient contract they share.

    ``temperature`` fixes both how that state is read and which space the returned gradient
    lives in:

    * A positive float (``1.0`` by default) reads it as unnormalized logits. The backend
      applies ``softmax(logits / temperature)``, so the gradient is logit-space.
    * ``None`` reads it as probability distributions, consumed directly. Every active row is
      validated, and the gradient is probability-space.

    The field is named ``logits`` in both modes; each tool's own documentation states what it
    means there.

    Attributes:
        temperature (float | None): Softmax temperature applied to the relaxed state, or
            ``None`` to consume that state as a probability distribution.
    """

    temperature: float | None = InputField(
        default=1.0,
        title="Temperature",
        description="Softmax temperature for softmax(logits / T); None consumes probabilities directly.",
        gt=0.0,
        allow_inf_nan=False,
    )

    def sequence_logits(self) -> dict[str, SequenceLogits]:
        """Return every relaxed matrix on this input, keyed by its dotted field path.

        Returns:
            dict[str, SequenceLogits]: Matrices found on this input, in field order.
        """
        return dict(iter_sequence_logits(self))

    def allowed_tokens(
        self,
        path: str,  # noqa: ARG002 — unrestricted by default; overrides key the mask off the path
        logits: SequenceLogits,  # noqa: ARG002 — unrestricted by default; overrides size the mask from it
    ) -> list[list[bool]] | None:
        """Return the mask of permitted columns for one matrix, or ``None`` for no restriction.

        A row that is entirely ``False`` marks fixed context: not a design variable, so it must
        hold zeros in both temperature modes. Override this in tools whose permitted vocabulary
        varies by position, such as the mixed protein/DNA models.

        Args:
            path (str): Field path of the matrix, as keyed by :meth:`sequence_logits`.
            logits (SequenceLogits): The matrix the mask applies to.

        Returns:
            list[list[bool]] | None: Per-position permitted columns, or ``None`` when every
                column is permitted at every position.
        """
        return None

    @classmethod
    def relaxed_field(cls) -> str:
        """Return the name of the field holding this tool's relaxed state.

        Returns:
            str: ``"inputs"`` for a tool that takes a list of items, else ``"logits"``.
        """
        return "inputs" if "inputs" in cls.model_fields else "logits"

    @classmethod
    def relaxed_state_type(cls, field: str | None = None) -> type[SequenceLogits]:
        """Return the :class:`SequenceLogits` subclass a field holds.

        Args:
            field (str | None): Name of the relaxed-state field, or ``None`` to use this
                class's own relaxed field.

        Returns:
            type[SequenceLogits]: The vocabulary-specific matrix type declared for that field.

        Raises:
            TypeError: If the field is absent or is not a single ``SequenceLogits``.
        """
        field = field or cls.relaxed_field()
        info = cls.model_fields.get(field)
        annotation = info.annotation if info is not None else None
        if isinstance(annotation, type) and issubclass(annotation, SequenceLogits):
            return annotation
        # A list of items: the element type carries the vocabulary.
        (element,) = get_args(annotation) or (None,)
        if isinstance(element, type) and issubclass(element, SequenceLogits):
            return element
        raise TypeError(f"{cls.__name__} has no single {field!r} matrix, so it needs its own from_probabilities()")

    @classmethod
    def candidate_type(cls) -> type | None:
        """Return the per-candidate type a list-shaped input holds, or ``None``.

        Returns:
            type | None: The element type of ``inputs``, or ``None`` for an input that declares a
                single ``logits`` matrix instead.
        """
        if cls.relaxed_field() != "inputs":
            return None
        (element,) = get_args(cls.model_fields["inputs"].annotation) or (None,)
        return element if isinstance(element, type) else None

    @classmethod
    def candidate_specs(cls, probabilities: Any) -> list[dict[str, Any]]:
        """Normalize the factory's first argument into one spec per candidate.

        One ``(L, V)`` matrix describes a single candidate. A list describes several, each entry
        either a bare matrix or a mapping holding that candidate's ``probabilities`` alongside its
        own context, which keeps matrices and context together instead of in parallel arrays.

        Args:
            probabilities (Any): One matrix, or a list of matrices or candidate mappings.

        Returns:
            list[dict[str, Any]]: One mapping per candidate, each carrying ``probabilities``.

        Raises:
            ValueError: If an empty list was supplied, or a mapping omits ``probabilities``.
        """
        if isinstance(probabilities, Mapping):
            probabilities = [probabilities]
        if not isinstance(probabilities, Sequence) or isinstance(probabilities, str | bytes):
            raise ValueError(f"probabilities must be a matrix or a list of candidates, got {type(probabilities)}")
        if not probabilities:
            raise ValueError("probabilities must contain at least one candidate")
        first = probabilities[0]
        if isinstance(first, Mapping):
            specs = [dict(spec) for spec in probabilities]
            for index, spec in enumerate(specs):
                if "probabilities" not in spec:
                    raise ValueError(f"candidate {index} must supply its own 'probabilities'")
            return specs
        # Depth tells one matrix from a list of them: a matrix's rows hold numbers.
        if isinstance(first, Sequence) and first and isinstance(first[0], Sequence):
            return [{"probabilities": matrix} for matrix in probabilities]
        return [{"probabilities": probabilities}]

    @classmethod
    def build_candidate(
        cls, probabilities: Sequence[Sequence[float]], *, temperature: float = 1.0, **fields: Any
    ) -> Any:
        """Build one candidate from its probabilities and its own context.

        The base implementation covers an input whose candidates are bare matrices. A tool whose
        candidates bundle context overrides this, converts the matrix, and packs the bundle.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, V)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's own context fields, if it takes any.

        Returns:
            Any: The candidate, ready to place in ``inputs``.

        Raises:
            TypeError: If context fields are supplied for a candidate that is a bare matrix.
        """
        if fields:
            raise TypeError(f"{cls.__name__} candidates take no context fields, got {sorted(fields)}")
        return cls.relaxed_state_type().from_probabilities(probabilities, temperature=temperature)

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Any,
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> Self:
        """Build an input whose logits reproduce ``probabilities`` under its own temperature.

        This is the recommended way to start from a distribution: the temperature is given once
        and used both to convert the logits and to read them back, so the two cannot disagree.
        Converting by hand through
        :meth:`~proto_tools.utils.sequence_logits.SequenceLogits.from_probabilities` and then
        passing a different ``temperature`` to the constructor silently reshapes the
        distribution.

        One matrix builds one candidate. A list builds several, each entry either a bare matrix or
        a mapping carrying that candidate's ``probabilities`` with its own context; every matrix is
        converted at the one shared ``temperature``. Context passed by keyword applies to every
        candidate, and a mapping's own entry overrides it.

        Only needed to optimize in logit space. To have a backend consume probabilities as they
        are, pass them straight to the constructor with ``temperature=None``.

        Args:
            probabilities (Any): One ``(L, V)`` matrix of strictly positive rows, or a list of
                matrices or candidate mappings.
            temperature (float): Positive, finite temperature stored on the input.
            fields (Any): The tool's remaining input fields, and any candidate context shared by
                every candidate, such as a structure or template.

        Returns:
            Self: Input satisfying ``softmax(logits / temperature) == probabilities``.

        Raises:
            ValueError: If the temperature is not positive and finite, or several candidates are
                supplied to a tool that holds a single matrix.
        """
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
        specs = cls.candidate_specs(probabilities)
        field = cls.relaxed_field()
        if field != "inputs":
            if len(specs) > 1:
                raise ValueError(f"{cls.__name__} holds a single matrix, so it takes one candidate")
            matrix = cls.relaxed_state_type().from_probabilities(specs[0]["probabilities"], temperature=temperature)
            return cls(**{field: matrix, "temperature": temperature, **fields})

        # Context names belong to the candidate; everything else is an input-level field.
        candidate = cls.candidate_type()
        candidate_fields = getattr(candidate, "model_fields", {})
        shared = {name: fields.pop(name) for name in list(fields) if name in candidate_fields}
        candidates = [
            cls.build_candidate(
                spec.pop("probabilities"),
                temperature=temperature,
                **{**shared, **spec},
            )
            for spec in specs
        ]
        # Only a subclass declares 'inputs'; this branch runs for exactly those.
        return cls(inputs=candidates, temperature=temperature, **fields)  # type: ignore[call-arg]

    @model_validator(mode="after")
    def validate_relaxed_state(self) -> Self:
        """Enforce fixed-context zeros always, and the probability contract when temperature is None."""
        for path, matrix in self.sequence_logits().items():
            allowed = self.allowed_tokens(path, matrix)
            if allowed is not None:
                matrix.validate_fixed_rows(allowed=allowed, label=path)
            if self.temperature is None:
                matrix.validate_probabilities(allowed=allowed, label=path)
        return self


class ProteinGradientInput(GradientInput):
    """Gradient input over one relaxed protein chain.

    Attributes:
        temperature (float | None): Inherited softmax temperature; see :class:`GradientInput`.
        logits (ProteinSequenceLogits): Relaxed ``(L, 20)`` state in canonical amino-acid order
            ``ACDEFGHIKLMNPQRSTVWY``, shared by the returned gradient.
    """

    logits: ProteinSequenceLogits = InputField(
        title="Logits",
        description="Relaxed sequence state with shape (L, 20) in canonical amino-acid order.",
        examples=[[[0.0] * 20, [0.0] * 20]],
    )


class BatchedNucleotideGradientInput(GradientInput):
    """Gradient input over a batch of equal-length relaxed DNA sequences.

    Attributes:
        temperature (float | None): Inherited softmax temperature; see :class:`GradientInput`.
        inputs (list[NucleotideSequenceLogits]): One relaxed ``(L, 4)`` state per candidate in
            ``A,C,G,T`` order. The backend evaluates the batch in one vectorized pass, so
            every item must share one length.
    """

    inputs: list[NucleotideSequenceLogits] = InputField(
        title="Relaxed Sequences",
        description="One relaxed (L, 4) DNA state per candidate in A,C,G,T order",
        min_length=1,
        examples=[[[[0.0] * 4] * 50]],
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one relaxed state as well as a list of them."""
        return coerce_to_item_list(value)

    @model_validator(mode="after")
    def validate_batch_lengths(self) -> Self:
        """Require every batch item to share one sequence length."""
        lengths = sorted({len(item) for item in self.inputs})
        if len(lengths) > 1:
            raise ValueError(f"all batched logits must have the same sequence length; got {lengths}")
        return self


class GradientConfig(RelaxationConfig):
    """Shared configuration for gradient tools, carrying the quantities they can differentiate.

    A tool declares ``TERMS``, and callers discover what it offers through
    :meth:`supported_terms` without having to build a config first. Where the available
    quantities depend on another setting — the species a model was trained on, the construct a
    predictor was fitted for — the catalog is grouped by that setting's value and
    ``TERM_GROUP_FIELD`` names the field that selects a group.

    Attributes:
        soft (float): Inherited relaxation blend; see :class:`RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; see :class:`RelaxationConfig`.
    """

    # What this tool can differentiate, keyed by name or by the setting the choice depends on.
    TERMS: ClassVar[TermCatalog | GroupedTermCatalog] = {}

    # The config field whose value selects a group of a grouped TERMS; None when the catalog is flat.
    TERM_GROUP_FIELD: ClassVar[str | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Check a tool's catalog as it is declared, rather than when someone first asks for it."""
        super().__pydantic_init_subclass__(**kwargs)
        if cls.TERMS:
            _validate_term_catalog(cls.TERMS, owner=f"{cls.__name__}.TERMS")

    @classmethod
    def supported_terms(cls, **group: str) -> TermCatalog | GroupedTermCatalog:
        """Return the quantities this tool can differentiate.

        Each value is the term class itself, which carries its own selector fields and whatever
        values they permit. That is deliberately a catalog of what can be asked for rather than an
        enumeration of every combination of arguments: a parameterized quantity is described by
        its fields, which a caller reads from the class.

        Args:
            group (str): For a tool whose quantities depend on a setting, that setting's value,
                such as ``species="human"``. Omit it to see every group.

        Returns:
            TermCatalog | GroupedTermCatalog: ``name -> term class`` for a flat catalog, and for
                one group of a grouped catalog. ``setting value -> name -> term class`` for a
                grouped catalog asked without a filter. Always a fresh mapping, so editing what
                discovery hands back cannot edit the tool's own declaration.

        Raises:
            ValueError: If a filter is given to a tool whose catalog is not grouped, names a field
                other than the grouping one, names a value that field does not permit, or names a
                group for which no terms are declared.
        """
        if not group:
            if all(_is_term_class(entry) for entry in cls.TERMS.values()):
                return dict(cast(TermCatalog, cls.TERMS))
            return {name: dict(terms) for name, terms in cast(GroupedTermCatalog, cls.TERMS).items()}
        if cls.TERM_GROUP_FIELD is None:
            raise ValueError(f"{cls.__name__}.supported_terms() takes no filters; its catalog is not grouped")
        if len(group) > 1 or cls.TERM_GROUP_FIELD not in group:
            raise ValueError(
                f"{cls.__name__}.supported_terms() filters on {cls.TERM_GROUP_FIELD!r} alone, got {sorted(group)}"
            )
        value = group[cls.TERM_GROUP_FIELD]
        permitted = cls._group_values()
        if value not in permitted:
            raise ValueError(f"unknown {cls.TERM_GROUP_FIELD} {value!r}; {cls.__name__} supports {sorted(permitted)}")
        terms = cast(GroupedTermCatalog, cls.TERMS).get(value)
        if terms is None:
            raise ValueError(f"{cls.__name__} declares no differentiable terms for {cls.TERM_GROUP_FIELD}={value!r}")
        return dict(terms)

    @classmethod
    def _group_values(cls) -> tuple[str, ...]:
        """Return the values the grouping field permits, from its own schema where it states them.

        The config field is the source of truth for which values are legal; the catalog only says
        which of them have terms. A field that permits anything falls back to the catalog's keys.
        """
        info = cls.model_fields.get(cls.TERM_GROUP_FIELD or "")
        options = get_args(info.annotation) if info is not None else ()
        return tuple(str(option) for option in options) or tuple(cls.TERMS)


class GradientResult(BaseModel):
    """One evaluated item: the gradients it was asked for, its objective, and its own metrics.

    Everything a gradient tool produces per item lives here, so a batch split across workers
    reassembles into exactly one of these per input. Only the vocabulary stays at the top level,
    because the alphabet is a property of the tool rather than of an item.

    ``terms`` holds the requested quantities in the order they were asked for, which is how a
    caller lines a returned gradient up with what they wanted from it. A tool whose terms carry
    selectors narrows the field to its own ``TermGradient`` subclass, because Pydantic serializes
    against the declared type and a base-typed term would lose them.

    ``loss`` and ``gradient`` are the pre-term surface, populated only by tools that have not
    migrated yet and removed once none remain. Nothing reports both.

    Attributes:
        terms (list[TermGradient]): One entry per differentiated quantity requested for this item,
            in the order requested, each carrying its own raw value and raw gradient. Empty on a
            tool that has not yet migrated to per-term gradients.
        gradient (list[list[float]] | None): Pre-term gradient for this item, an ``(L, V)`` matrix
            matching its input shape and column order. ``None`` on a migrated tool, and in
            forward-only mode.
        loss (float | None): Pre-term objective value for this item; no batch mean or total.
            ``None`` on a migrated tool, which reports its scalar as a term value instead.
        metrics (dict[str, Any]): Metrics for this item alone.
    """

    terms: list[TermGradient] = Field(
        default_factory=list,
        title="Terms",
        description="One gradient per differentiated quantity, in the order requested.",
    )
    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Pre-term gradient w.r.t. this item's input logits; None on a tool reporting terms.",
    )
    loss: float | None = Field(
        default=None,
        title="Loss",
        description="Pre-term objective for this item; None on a tool reporting terms.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        title="Auxiliary Metrics",
        description="Metrics reported for this item alongside its objective value.",
    )

    def compose(
        self,
        goals: GoalsArgument | None = None,
        weights: WeightsArgument | None = None,
        *,
        strict: bool = True,
    ) -> ObjectiveResult:
        """Combine this item's raw gradient terms into one minimizable objective.

        Args:
            goals (GoalsArgument | None): Goal per term, keyed by the term itself or by its id.
                A term left out follows its advisory direction, which requires it to have one.
            weights (WeightsArgument | None): Non-negative weight per term, keyed the same way,
                defaulting to 1.0.
            strict (bool): Whether gradients taken at different points are incompatible.

        Returns:
            ObjectiveResult: The composed ``objective_loss`` and its gradient, with per-term
                diagnostics.
        """
        # Imported here because composition builds on this module's models.
        from proto_tools.utils.gradient.objectives import compose_gradients

        return compose_gradients(self, goals=goals, weights=weights, strict=strict)


class GradientOutput(BaseToolOutput):
    """Gradients of a backend objective with respect to one or more relaxed sequence states.

    Each gradient shares its item's shape and column order. The coordinate space follows the
    input's ``temperature``: logit-space for a positive float, probability-space for ``None``.

    ``results`` holds one entry per input item, in the order the items were supplied, however the
    batch was split across devices or remote chunks.

    Attributes:
        results (list[GradientResult]): One result per input item, in input order.
        vocab (list[str]): Column ordering for both the input logits and the returned gradients.
    """

    results: list[GradientResult] = Field(
        title="Results",
        description="One result per input item, in input order, each carrying its own gradient, loss and metrics",
    )
    vocab: list[str] = Field(
        title="Vocabulary",
        description="Symbols defining the column ordering for both the input logits and the returned gradient.",
    )

    @classmethod
    def result_type(cls) -> type[GradientResult]:
        """Return the per-item result model this output declares.

        Returns:
            type[GradientResult]: Element type of ``results``.

        Raises:
            TypeError: If ``results`` does not declare a ``GradientResult`` element.
        """
        annotation = cls.model_fields["results"].annotation
        (element,) = get_args(annotation) or (None,)
        if isinstance(element, type) and issubclass(element, GradientResult):
            return element
        raise TypeError(f"{cls.__name__}.results does not declare a GradientResult element")

    @classmethod
    def from_single(cls, payload: Mapping[str, Any], *, result_fields: Sequence[str] = ()) -> Self:
        """Build a one-item output from a worker payload shaped for a single evaluation.

        A backend that still evaluates one item per dispatch returns ``gradient``/``loss``/
        ``metrics`` at the top level; this lifts them into a single ``results`` entry and leaves
        the rest of the payload as call-level fields.

        Args:
            payload (Mapping[str, Any]): Worker payload for one evaluated item.
            result_fields (Sequence[str]): Extra per-item keys to move into the result, such as
                ``"structure"`` for a tool that predicts one.

        Returns:
            Self: Output holding exactly one result.
        """
        per_item = (*("gradient", "loss", "metrics"), *result_fields)
        item = {key: payload[key] for key in per_item if key in payload}
        call_level = {key: value for key, value in payload.items() if key not in item}
        return cls(results=[cls.result_type()(**item)], **call_level)

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
        # Suffix-additive so dotted names (e.g. "step_v1.5") aren't truncated.
        base = Path(export_path)
        json_path = base.parent / f"{base.name}.json"
        payload = self.model_dump(include={"results", "vocab"})
        json_path.write_text(json.dumps(payload, indent=2))


def attach_intrinsic_term(
    payload: MutableMapping[str, Any],
    term: DifferentiableTerm,
    *,
    matrices: Sequence[Sequence[Sequence[float]]],
    temperature: float | None,
    positions: Sequence[Sequence[int] | None] | None = None,
    hard: float = 0.0,
    soft: float = 1.0,
) -> None:
    """Give every item in a worker payload the one term its tool differentiates.

    A tool with a single intrinsic objective reports that objective as its item ``loss`` and its
    derivative as the item ``gradient``, which is exactly the term's value and raw gradient. This
    reshapes such a payload in place, before it is validated into an output model, so the tool
    itself needs no per-item loop.

    Both per-item keys are consumed rather than left behind, so a migrated tool carries its
    scalar and its gradient in exactly one place.

    The differentiated matrices are passed in rather than read off an input, because only the
    tool knows what it actually differentiated: a model that concatenates two chains into one
    matrix differentiates that matrix, not either field it was built from.

    Args:
        payload (MutableMapping[str, Any]): Worker payload holding a ``results`` list and the
            ``vocab`` its gradients are columned by.
        term (DifferentiableTerm): The term describing the quantity, shared by every item.
        matrices (Sequence[Sequence[Sequence[float]]]): The relaxed state each item was
            differentiated at, in item order.
        temperature (float | None): The input's temperature, which fixes the coordinate space.
        positions (Sequence[Sequence[int] | None] | None): 1-indexed input position per row for
            each item, where a tool relaxes a window inside a longer context.
        hard (float): Straight-through coefficient in force.
        soft (float): Relaxation blend in force.

    Raises:
        ValueError: If a matrix was not supplied for every item.
    """
    items = payload["results"]
    if len(matrices) != len(items):
        raise ValueError(f"attach_intrinsic_term got {len(matrices)} matrices for {len(items)} items")
    rows = positions if positions is not None else [None] * len(items)
    for item, matrix, item_positions in zip(items, matrices, rows, strict=True):
        # Moved, not copied: the term becomes the single place this item's scalar and gradient live
        item["terms"] = [
            TermGradient(
                term=term,
                term_value=item.pop("loss"),
                gradient=item.pop("gradient"),
                metadata=GradientMetadata.from_input(
                    matrix,
                    vocab=payload["vocab"],
                    temperature=temperature,
                    positions=item_positions,
                    hard=hard,
                    soft=soft,
                ),
            )
        ]


__all__ = [
    "BatchedNucleotideGradientInput",
    "GradientConfig",
    "GradientInput",
    "GradientOutput",
    "GradientResult",
    "ProteinGradientInput",
    "RelaxationConfig",
    "attach_intrinsic_term",
    "coerce_to_item_list",
]
