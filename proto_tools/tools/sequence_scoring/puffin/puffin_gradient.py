"""Puffin transcription-initiation gradient tool."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
    PUFFIN_MIN_INPUT_LENGTH,
    PUFFIN_PADDING,
    TRACK_NAMES,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    ConfigField,
    GradientInput,
    GradientOutput,
    InputField,
    NucleotideSequenceLogits,
    RelaxationConfig,
    ToolInstance,
    coerce_to_item_list,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

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
PuffinObjectiveDirection = Literal["max", "min"]


class PuffinGradientLossTerm(BaseModel):
    """One differentiable Puffin transcription-initiation objective term.

    The term's score is the mean predicted signal for ``track`` across output positions
    ``position_start``..``position_end``. ``direction`` fixes its sign so the summed objective is
    always minimized, and ``weight`` scales it before the terms are added together.

    Attributes:
        track (PuffinTrack): Output track to optimize, named as in ``TRACK_NAMES``; ``+`` and
            ``-`` select the forward and reverse strand heads.
        position_start (int | None): First output position to average, 1-indexed and inclusive.
            ``None`` starts at the first output position.
        position_end (int | None): Last output position to average, 1-indexed and inclusive.
            ``None`` ends at the last output position.
        direction (PuffinObjectiveDirection): ``"max"`` minimizes ``-mean_signal``; ``"min"``
            minimizes ``+mean_signal``.
        weight (float): Non-negative scalar applied before terms are summed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

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
    direction: PuffinObjectiveDirection = Field(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') mean signal for this track",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Weight",
        description="Relative weight of this term in the summed objective (must be non-negative)",
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


class PuffinGradientSampleMetrics(Metrics):
    """Per-candidate differentiable Puffin objective metrics.

    Attributes:
        loss_terms (list[dict[str, Any]]): Per-objective-term metadata — track, position span,
            direction, weight, mean signal, and the weighted contribution.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "loss": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Weighted scalar Puffin objective value for one candidate.",
            "better_values_are": "lower",
        },
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
        default="loss",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )
    loss_terms: list[dict[str, Any]] = Field(
        default_factory=list,
        title="Loss Terms",
        description="Per-objective-term signal and weighting metadata for this candidate.",
    )


class PuffinGradientOutput(GradientOutput):
    """Puffin weighted-objective output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input candidate, in input order, each holding
            that candidate's full ``(L, 4)`` gradient (``None`` in forward-only mode), its
            weighted objective value, and its per-term signal metrics.
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """


class PuffinGradientConfig(RelaxationConfig):
    """Configuration for the Puffin transcription-initiation gradient tool.

    Attributes:
        loss_terms (list[PuffinGradientLossTerm]): Objective terms summed into the scalar loss.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities (1).
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    loss_terms: list[PuffinGradientLossTerm] = ConfigField(
        title="Loss Terms",
        default_factory=lambda: [PuffinGradientLossTerm()],
        min_length=1,
        description="Track-and-position objective terms summed into the scalar loss",
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
    """Compute Puffin weighted initiation-signal gradients with respect to relaxed DNA logits."""
    logger.debug("Using local worker for Puffin gradient (%d candidate(s))", len(inputs.inputs))
    result = ToolInstance.dispatch(
        "puffin",
        {
            "operation": "compute_gradient",
            "items": [matrix.root for matrix in inputs.inputs],
            "loss_terms": [term.model_dump() for term in config.loss_terms],
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
    return PuffinGradientOutput(**result)
