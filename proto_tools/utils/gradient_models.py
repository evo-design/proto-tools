"""proto_tools/utils/gradient_models.py.

Shared input and output contracts for relaxed-sequence gradient tools.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.utils.base_config import BaseConfig, ConfigField
from proto_tools.utils.sequence_logits import (
    NucleotideSequenceLogits,
    ProteinSequenceLogits,
    SequenceLogits,
    iter_sequence_logits,
)
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField


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

    Attributes:
        soft (float): Blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Straight-through coefficient; 1 forwards a hard one-hot to the model.
    """

    soft: float = ConfigField(
        default=1.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="Soft Mixing",
        description="Blend the raw input matrix (0) toward softmax probabilities (1).",
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
            ``A,C,G,T`` order. A single state may be passed on its own. The backend evaluates the
            batch in one vectorized pass, so every item must share one length.
    """

    inputs: list[NucleotideSequenceLogits] = InputField(
        title="Relaxed Sequences",
        description="One relaxed (L, 4) DNA state per candidate in A,C,G,T order; a single state is accepted alone",
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


class GradientResult(BaseModel):
    """One evaluated item: its gradient, its objective, and its own metrics.

    Everything a gradient tool produces per item lives here, so a batch split across workers
    reassembles into exactly one of these per input. Only the vocabulary stays at the top level,
    because the alphabet is a property of the tool rather than of an item.

    Attributes:
        gradient (list[list[float]] | None): Gradient for this item, an ``(L, V)`` matrix
            matching its input shape and column order, or ``None`` in forward-only mode.
        loss (float): This item's own objective value; no batch mean or total.
        metrics (dict[str, Any]): Metrics for this item alone.
    """

    gradient: list[list[float]] | None = Field(
        default=None,
        title="Gradient",
        description="Gradient of this item's loss w.r.t. its input logits; matches its shape and column order",
    )
    loss: float = Field(
        title="Loss",
        description="This item's objective; lower is better (typically a positive negative log-likelihood)",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        title="Auxiliary Metrics",
        description="Metrics reported for this item alongside its objective value.",
    )


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


__all__ = [
    "BatchedNucleotideGradientInput",
    "RelaxationConfig",
    "GradientInput",
    "GradientOutput",
    "GradientResult",
    "coerce_to_item_list",
    "ProteinGradientInput",
]
