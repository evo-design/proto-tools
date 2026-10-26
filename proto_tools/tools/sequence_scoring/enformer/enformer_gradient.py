"""Enformer regulatory-activity gradient tool."""

import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.sequence_scoring.enformer.enformer_prediction import (
    ENFORMER_CONTEXT,
    ENFORMER_OUTPUT,
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
from proto_tools.utils.sequence_logits import SequenceLogits
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

EnformerObjectiveDirection = Literal["max", "min"]


class EnformerGradientLossTerm(BaseModel):
    """One differentiable Enformer track-and-bin objective term.

    The term's score is the mean predicted activity over ``track`` across bins
    ``bin_start``..``bin_end``. ``direction`` fixes its sign so the summed objective is always
    minimized, and ``weight`` scales it before the terms are added together.

    Attributes:
        track (int): Enformer output-track index (0-based, as the prediction tool reports them).
        bin_start (int): First output bin to average, 1-indexed and inclusive.
        bin_end (int): Last output bin to average, 1-indexed and inclusive.
        direction (EnformerObjectiveDirection): ``"max"`` minimizes ``-mean_activity``; ``"min"``
            minimizes ``+mean_activity``.
        weight (float): Non-negative scalar applied before terms are summed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    track: int = Field(
        default=0,
        ge=0,
        title="Track",
        description="Enformer output-track index to optimize (5313 human / 1643 mouse tracks)",
    )
    bin_start: int = Field(
        default=1,
        ge=1,
        le=ENFORMER_OUTPUT,
        title="Bin Start",
        description="First output bin to average, 1-indexed and inclusive",
    )
    bin_end: int = Field(
        default=ENFORMER_OUTPUT,
        ge=1,
        le=ENFORMER_OUTPUT,
        title="Bin End",
        description="Last output bin to average, 1-indexed and inclusive",
    )
    direction: EnformerObjectiveDirection = Field(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') mean activity for this track",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Weight",
        description="Relative weight of this term in the summed objective (must be non-negative)",
    )

    @model_validator(mode="after")
    def validate_bin_span(self) -> Self:
        """Require a non-empty, ordered bin span."""
        if self.bin_end < self.bin_start:
            raise ValueError(f"enformer gradient: bin_end ({self.bin_end}) precedes bin_start ({self.bin_start})")
        return self


class EnformerGradientCandidate(BaseModel):
    """One candidate: relaxed bases at selected positions inside a full Enformer context window.

    Enformer reads a fixed 196,608 bp context, but a design problem usually varies only a small
    part of it. The candidate therefore carries the whole context as a sequence plus the
    positions whose bases are relaxed, so the gradient comes back only for those positions and
    the surrounding context still reaches the model.

    Attributes:
        sequence (str): Full ``196608`` bp context window; the designed positions may hold any
            placeholder base, since ``logits`` replaces them.
        positions (list[int]): Context positions the relaxed rows correspond to, 1-indexed,
            inclusive, strictly increasing, one per row of ``logits``.
        logits (NucleotideSequenceLogits): Relaxed ``(len(positions), 4)`` state in ``A,C,G,T``
            order, aligned to ``positions``.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(
        title="Context Sequence",
        description="Full 196608 bp Enformer context window holding the designed positions",
    )
    positions: list[int] = Field(
        title="Designed Positions",
        description="1-indexed inclusive context positions the relaxed rows map to, strictly increasing",
        min_length=1,
    )
    logits: NucleotideSequenceLogits = Field(
        title="Logits",
        description="Relaxed (len(positions), 4) state in A,C,G,T order, aligned to positions",
    )

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this candidate's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this candidate's matrix.
        """
        yield "logits", self.logits

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        """Require an exact context length and positions that align with the relaxed rows."""
        if len(self.sequence) != ENFORMER_CONTEXT:
            raise ValueError(
                f"enformer gradient: sequence must be exactly {ENFORMER_CONTEXT} bp, got {len(self.sequence)}"
            )
        if len(self.positions) != len(self.logits):
            raise ValueError(
                f"enformer gradient: positions ({len(self.positions)}) and logits ({len(self.logits)}) must align"
            )
        if any(later <= earlier for earlier, later in zip(self.positions, self.positions[1:], strict=False)):
            raise ValueError("enformer gradient: positions must be strictly increasing with no duplicates")
        if self.positions[-1] > ENFORMER_CONTEXT:
            raise ValueError(
                f"enformer gradient: position {self.positions[-1]} exceeds the {ENFORMER_CONTEXT} bp context"
            )
        return self


class EnformerGradientInput(GradientInput):
    """Input for the Enformer gradient tool: one or many relaxed candidates.

    Attributes:
        inputs (list[EnformerGradientCandidate]): One context-and-relaxed-positions bundle per
            candidate.
        temperature (float | None): Softmax temperature, shared by every candidate. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[EnformerGradientCandidate] = InputField(
        title="Relaxed Candidates",
        description="One context window with relaxed positions per candidate",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_items(cls, value: Any) -> Any:
        """Accept one candidate as well as a list of them."""
        return coerce_to_item_list(value)

    @classmethod
    def build_candidate(
        cls,
        probabilities: Any,
        *,
        temperature: float = 1.0,
        **fields: Any,
    ) -> EnformerGradientCandidate:
        """Build one candidate pairing the converted matrix with its context window.

        Args:
            probabilities (Any): ``(len(positions), 4)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``sequence`` and
                ``positions``.

        Returns:
            EnformerGradientCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``sequence`` or ``positions`` is missing.
        """
        missing = {"sequence", "positions"} - set(fields)
        if missing:
            raise ValueError(f"enformer gradient from_probabilities requires {sorted(missing)}")
        matrix = NucleotideSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return EnformerGradientCandidate(logits=matrix, **fields)


class EnformerGradientSampleMetrics(Metrics):
    """Per-candidate differentiable Enformer objective metrics.

    Attributes:
        loss_terms (list[dict[str, Any]]): Per-objective-term metadata — track, bin span,
            direction, weight, mean activity, and the weighted contribution.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "loss": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Weighted scalar Enformer objective value for one candidate.",
            "better_values_are": "lower",
        },
        "designed_positions": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Number of relaxed context positions this candidate carries.",
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
        description="Per-objective-term activity and weighting metadata for this candidate.",
    )


class EnformerGradientOutput(GradientOutput):
    """Enformer weighted-objective output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input candidate, in input order, each holding
            that candidate's gradient w.r.t. its relaxed positions (``None`` in forward-only
            mode), its weighted objective value, and its per-term activity metrics.
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """


class EnformerGradientConfig(RelaxationConfig):
    """Configuration for the Enformer regulatory-activity gradient tool.

    Attributes:
        loss_terms (list[EnformerGradientLossTerm]): Objective terms summed into the scalar loss.
        species (Literal['human', 'mouse']): Species model to use.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient_models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    loss_terms: list[EnformerGradientLossTerm] = ConfigField(
        title="Loss Terms",
        default_factory=lambda: [EnformerGradientLossTerm()],
        min_length=1,
        description="Track-and-bin objective terms summed into the scalar loss",
    )
    species: Literal["human", "mouse"] = ConfigField(
        title="Species",
        default="human",
        description="Species model to use",
        reload_on_change=True,
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


def example_input() -> EnformerGradientInput:
    """Minimal valid input for testing and examples."""
    sequence = "A" * ENFORMER_CONTEXT
    positions = [ENFORMER_CONTEXT // 2, ENFORMER_CONTEXT // 2 + 1]
    return EnformerGradientInput(
        inputs=[
            EnformerGradientCandidate(
                sequence=sequence,
                positions=positions,
                logits=NucleotideSequenceLogits([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            )
        ]
    )


@tool(
    key="enformer-gradient",
    label="Enformer Gradient",
    category="sequence_scoring",
    input_class=EnformerGradientInput,
    config_class=EnformerGradientConfig,
    output_class=EnformerGradientOutput,
    description="Compute Enformer track-and-bin activity gradients for relaxed DNA positions",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
    metrics_class=EnformerGradientSampleMetrics,
)
def run_enformer_gradient(
    inputs: EnformerGradientInput,
    config: EnformerGradientConfig,
    instance: Any = None,
) -> EnformerGradientOutput:
    """Compute Enformer weighted track-activity gradients with respect to relaxed DNA logits."""
    logger.debug("Using local worker for Enformer gradient (%d candidate(s))", len(inputs.inputs))
    result = ToolInstance.dispatch(
        "enformer",
        {
            "operation": "compute_gradient",
            "items": [
                {
                    "sequence": candidate.sequence,
                    "positions": candidate.positions,
                    "logits": candidate.logits.root,
                }
                for candidate in inputs.inputs
            ],
            "loss_terms": [term.model_dump() for term in config.loss_terms],
            "temperature": inputs.temperature,
            "soft": config.soft,
            "hard": config.hard,
            "compute_gradient": config.compute_gradient,
            "species": config.species,
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    return EnformerGradientOutput(**result)
