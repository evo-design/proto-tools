"""proto_tools/utils/gradient_models.py.

Shared input and output contracts for relaxed-sequence gradient tools.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from proto_tools.utils.base_config import BaseConfig, ConfigField
from proto_tools.utils.sequence_logits import (
    NucleotideSequenceLogits,
    ProteinSequenceLogits,
    SequenceLogits,
    iter_sequence_logits,
)
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField

GradientValue = list[list[float]] | list[list[list[float]]] | None


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
    def relaxed_state_type(cls, field: str = "logits") -> type[SequenceLogits]:
        """Return the :class:`SequenceLogits` subclass a field holds.

        Args:
            field (str): Name of the relaxed-state field.

        Returns:
            type[SequenceLogits]: The vocabulary-specific matrix type declared for that field.

        Raises:
            TypeError: If the field is absent or is not a single ``SequenceLogits``.
        """
        info = cls.model_fields.get(field)
        annotation = info.annotation if info is not None else None
        if isinstance(annotation, type) and issubclass(annotation, SequenceLogits):
            return annotation
        raise TypeError(f"{cls.__name__} has no single {field!r} matrix, so it needs its own from_probabilities()")

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Sequence[Sequence[float]],
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

        Only needed to optimize in logit space. To have a backend consume probabilities as they
        are, pass them straight to the constructor with ``temperature=None``.

        Args:
            probabilities (Sequence[Sequence[float]]): ``(L, V)`` rows, each a strictly positive
                probability distribution.
            temperature (float): Positive, finite temperature stored on the input.
            fields (Any): The tool's remaining input fields, such as a structure or template.

        Returns:
            Self: Input satisfying ``softmax(logits / temperature) == probabilities``.
        """
        matrix = cls.relaxed_state_type().from_probabilities(probabilities, temperature=temperature)
        # ``logits`` is declared by the subclass, not by GradientInput itself.
        return cls(**{"logits": matrix, "temperature": temperature, **fields})

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
        logits (list[NucleotideSequenceLogits]): Relaxed ``(B, L, 4)`` state in ``A,C,G,T``
            order. Use ``B=1`` for a single design candidate.
    """

    logits: list[NucleotideSequenceLogits] = InputField(
        title="Logits",
        description="Batched relaxed DNA state with shape (B, L, 4) in A,C,G,T order.",
        min_length=1,
        examples=[[[[0.0] * 4] * 50]],
    )

    @classmethod
    def from_probabilities(  # type: ignore[override]  # a batch takes (B, L, V), not (L, V)
        cls,
        probabilities: Sequence[Sequence[Sequence[float]]],
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> Self:
        """Build a batched input whose logits reproduce ``probabilities`` under its temperature.

        Args:
            probabilities (Sequence[Sequence[Sequence[float]]]): ``(B, L, 4)`` rows, each row a
                strictly positive probability distribution.
            temperature (float): Positive, finite temperature stored on the input.
            fields (Any): The tool's remaining input fields.

        Returns:
            Self: Input satisfying ``softmax(logits / temperature) == probabilities``.
        """
        matrices = [
            NucleotideSequenceLogits.from_probabilities(item, temperature=temperature) for item in probabilities
        ]
        return cls(logits=matrices, temperature=temperature, **fields)

    @model_validator(mode="after")
    def validate_batch_lengths(self) -> Self:
        """Require every batch item to share one sequence length."""
        lengths = sorted({len(item) for item in self.logits})
        if len(lengths) > 1:
            raise ValueError(f"all batched logits must have the same sequence length; got {lengths}")
        return self


class GradientResult(BaseModel):
    """One evaluated item: its gradient, its objective, and its own metrics.

    Everything a gradient tool produces per item lives here, so a batch split across workers
    reassembles into exactly one of these per input. Only the vocabulary stays at the top level,
    because the alphabet is a property of the tool rather than of an item.

    Attributes:
        gradient (GradientValue): Gradient for this item, matching its input shape and column
            order, or ``None`` when the tool runs in forward-only mode.
        loss (float): This item's own objective value; no batch mean or total.
        metrics (dict[str, Any]): Metrics for this item alone.
    """

    gradient: GradientValue = Field(
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
    "GradientValue",
    "ProteinGradientInput",
]
