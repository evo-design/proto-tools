"""Puffin transcription-initiation gradient tool."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
    PUFFIN_MIN_INPUT_LENGTH,
    PUFFIN_PADDING,
    TRACK_NAMES,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    DifferentiableTerm,
    GradientConfig,
    GradientInput,
    GradientMetadata,
    GradientOutput,
    GradientResult,
    InputField,
    NucleotideSequenceLogits,
    TermCatalog,
    TermGradient,
    ToolInstance,
    coerce_to_item_list,
    resolve_terms,
)
from proto_tools.utils.tool_io import Directionality, Metrics, MetricSpec

logger = logging.getLogger(__name__)

PuffinTrack = Literal[
    "FANTOM_CAGE+",
    "ENCODE_CAGE+",
    "ENCODE_RAMPAGE+",
    "GRO_CAP+",
    "PRO_CAP+",
    "PRO_CAP-",
    "GRO_CAP-",
    "ENCODE_RAMPAGE-",
    "ENCODE_CAGE-",
    "FANTOM_CAGE-",
]


class PuffinSignalTerm(DifferentiableTerm):
    """One differentiable Puffin quantity: mean initiation signal over a track and position span.

    Whether more signal is wanted depends on whether the track is a target or an off-target,
    and how much it matters relative to other tracks is a weight — both are goals a caller
    applies to the returned value, not properties of the quantity.

    An open-ended span is resolved against the candidate's own output window, so the term that
    comes back names the positions actually averaged rather than the ``None`` that was asked for.

    Attributes:
        id (Literal["track_signal"]): Fixed identifier for this kind of quantity.
        track (PuffinTrack): Output track to optimize, named as in ``TRACK_NAMES``; ``+`` and
            ``-`` select the forward and reverse strand heads.
        position_start (int | None): First output position to average, 1-indexed and inclusive.
            ``None`` starts at the first output position.
        position_end (int | None): Last output position to average, 1-indexed and inclusive.
            ``None`` ends at the last output position.
    """

    #: A track can be something to drive up or something to suppress; only the caller knows.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = "Mean predicted transcription-initiation signal over a Puffin track and position span"

    id: Literal["track_signal"] = Field(
        default="track_signal",
        title="Term ID",
        description="Identifies this quantity as Puffin mean track signal over a position span.",
    )
    track: PuffinTrack = Field(
        default="FANTOM_CAGE+",
        title="Track",
        description="Puffin output track to optimize, named as in TRACK_NAMES (strand suffixed)",
    )
    position_start: int | None = Field(
        default=None,
        ge=1,
        title="Position Start",
        description="First output position to average, 1-indexed inclusive; None starts at the first",
    )
    position_end: int | None = Field(
        default=None,
        ge=1,
        title="Position End",
        description="Last output position to average, 1-indexed inclusive; None ends at the last",
    )

    @model_validator(mode="after")
    def validate_position_span(self) -> Self:
        """Require an ordered span when both ends are given."""
        if (
            self.position_start is not None
            and self.position_end is not None
            and self.position_end < self.position_start
        ):
            raise ValueError(
                f"puffin gradient: position_end ({self.position_end}) precedes position_start ({self.position_start})"
            )
        return self


class PuffinGradientInput(GradientInput):
    """Input for the Puffin gradient tool: one or many relaxed DNA states.

    Puffin scores every base of its input, so the whole relaxed matrix is differentiated — there
    is no fixed context window to pin and no separate designed-position list.

    Attributes:
        inputs (list[NucleotideSequenceLogits]): One relaxed state per candidate, each ``(L, 4)``
            in ``A,C,G,T`` order with ``L`` at least ``PUFFIN_MIN_INPUT_LENGTH``. Puffin trims
            ``PUFFIN_PADDING`` bases from each end, so the scored output span is
            ``L - 2 * PUFFIN_PADDING`` positions.
        temperature (float | None): Softmax temperature, shared by every candidate. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[NucleotideSequenceLogits] = InputField(
        title="Relaxed Sequences",
        description="One relaxed (L, 4) DNA state per candidate in A,C,G,T order",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one relaxed state as well as a list of them."""
        return coerce_to_item_list(value)

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Reject any candidate shorter than one padded output position."""
        for index, matrix in enumerate(self.inputs):
            if len(matrix) < PUFFIN_MIN_INPUT_LENGTH:
                raise ValueError(
                    f"puffin gradient: inputs[{index}] must be at least {PUFFIN_MIN_INPUT_LENGTH} bp "
                    f"(2 * {PUFFIN_PADDING} bp padding + 1 bp output); got {len(matrix)}"
                )
        return self


class PuffinTermGradient(TermGradient):
    """One Puffin track's mean initiation signal and its raw gradient.

    Attributes:
        term (PuffinSignalTerm): The track-and-position quantity this gradient answers.
    """

    term: PuffinSignalTerm = Field(title="Term", description="Track-and-position quantity answered.")


class PuffinGradientResult(GradientResult):
    """One evaluated candidate and every track signal differentiated for it.

    Attributes:
        terms (list[PuffinTermGradient]): One entry per requested track, in the order requested.
    """

    terms: list[PuffinTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Terms",
        description="One gradient per requested track-and-position quantity, in the order requested.",
    )


class PuffinGradientSampleMetrics(Metrics):
    """Per-candidate differentiable Puffin objective metrics.

    Attributes:
        output_length (int): Number of scored output positions for this candidate.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "output_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Number of scored output positions for this candidate.",
            "better_values_are": "context-dependent",
        },
    }

    primary_metric: str | None = Field(
        default="output_length",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class PuffinGradientOutput(GradientOutput):
    """Puffin track-signal output; gradients are optional in forward-only mode.

    Attributes:
        results (list[PuffinGradientResult]): One per input candidate, in input order, each
            holding that candidate's requested track signals and their raw ``(L, 4)`` gradients
            (``None`` in forward-only mode).
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """

    results: list[PuffinGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input candidate, in input order, each carrying its track signals.",
    )


class PuffinGradientConfig(GradientConfig):
    """Configuration for the Puffin transcription-initiation gradient tool.

    Attributes:
        gradient_terms (list[PuffinSignalTerm]): Track-and-position quantities to differentiate.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    TERMS: ClassVar[TermCatalog] = {"track_signal": PuffinSignalTerm}

    gradient_terms: list[PuffinSignalTerm] = ConfigField(
        title="Gradient Terms",
        default_factory=lambda: [PuffinSignalTerm()],
        min_length=1,
        description="Track-and-position quantities to differentiate, by name or as instances",
    )
    compute_gradient: bool = ConfigField(
        title="Compute Gradient",
        default=True,
        description="Run backward pass and return gradient; set False for forward-only scoring",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept a term by name or class, leaving an already-built instance as it is."""
        if not isinstance(value, list):
            return value
        catalog = cls.supported_terms()
        return [
            resolve_terms([entry], catalog)[0]  # type: ignore[arg-type]
            if isinstance(entry, str | type)
            else entry
            for entry in value
        ]

    @model_validator(mode="after")
    def reject_repeated_terms(self) -> Self:
        """The same track and position span twice would be one quantity asked for twice."""
        asked = [(term.track, term.position_start, term.position_end) for term in self.gradient_terms]
        repeated = sorted({item for item in asked if asked.count(item) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same track and position span more than once: {repeated}")
        return self


def example_input() -> PuffinGradientInput:
    """Minimal valid input for testing and examples."""
    return PuffinGradientInput(inputs=[NucleotideSequenceLogits([[0.25] * 4] * PUFFIN_MIN_INPUT_LENGTH)])


@tool(
    key="puffin-gradient",
    label="Puffin Gradient",
    category="sequence_scoring",
    input_class=PuffinGradientInput,
    config_class=PuffinGradientConfig,
    output_class=PuffinGradientOutput,
    description="Compute Puffin transcription-initiation gradients for relaxed DNA sequences",
    uses_gpu=True,
    pin_visible_devices=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=4,
    metrics_class=PuffinGradientSampleMetrics,
)
def run_puffin_gradient(
    inputs: PuffinGradientInput,
    config: PuffinGradientConfig,
    instance: Any = None,
) -> PuffinGradientOutput:
    """Report each requested Puffin track signal and its gradient w.r.t. the relaxed logits."""
    logger.debug("Using local worker for Puffin gradient (%d candidate(s))", len(inputs.inputs))
    result = ToolInstance.dispatch(
        "puffin",
        {
            "operation": "compute_gradient",
            "items": [matrix.root for matrix in inputs.inputs],
            "gradient_terms": [term.model_dump() for term in config.gradient_terms],
            "track_names": list(TRACK_NAMES),
            "temperature": inputs.temperature,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )

    # Puffin differentiates the whole relaxed state, so a gradient covers every input row.
    results = []
    for matrix, item in zip(inputs.inputs, result["results"], strict=True):
        basis = GradientMetadata.from_input(
            matrix.root,
            vocab=result["vocab"],
            temperature=inputs.temperature,
            soft=config.soft,
            hard=config.hard,
        )
        results.append(
            PuffinGradientResult(
                terms=[
                    PuffinTermGradient(
                        term=PuffinSignalTerm(**entry["term"]),
                        term_value=float(entry["value"]),
                        gradient=entry["gradient"],
                        metadata=basis,
                    )
                    for entry in item["terms"]
                ],
                metrics=item["metrics"],
            )
        )
    return PuffinGradientOutput(results=results, vocab=result["vocab"])
