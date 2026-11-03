"""SpliceTransformer splice-channel gradient tool."""

import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.rna_splicing.splice_transformer.splice_transformer import (
    CONTEXT_LENGTH,
    TARGET_LENGTH,
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

# The 18 output channels, in model order: 3 splice classes then 15 tissues.
SpliceTransformerChannel = Literal[
    "NEITHER",
    "ACCEPTOR",
    "DONOR",
    "ADIPOSE_TISSUE",
    "BLOOD",
    "BLOOD_VESSEL",
    "BRAIN",
    "COLON",
    "HEART",
    "KIDNEY",
    "LIVER",
    "LUNG",
    "MUSCLE",
    "NERVE",
    "SMALL_INTESTINE",
    "SKIN",
    "SPLEEN",
    "STOMACH",
]
SPLICE_CHANNEL_INDEX: dict[str, int] = {
    name: index
    for index, name in enumerate(
        (
            "NEITHER",
            "ACCEPTOR",
            "DONOR",
            "ADIPOSE_TISSUE",
            "BLOOD",
            "BLOOD_VESSEL",
            "BRAIN",
            "COLON",
            "HEART",
            "KIDNEY",
            "LIVER",
            "LUNG",
            "MUSCLE",
            "NERVE",
            "SMALL_INTESTINE",
            "SKIN",
            "SPLEEN",
            "STOMACH",
        )
    )
}
SpliceTransformerObjectiveDirection = Literal["max", "min"]


class SpliceTransformerGradientLossTerm(BaseModel):
    """One differentiable SpliceTransformer channel-and-position objective term.

    The term's score is the mean predicted probability for ``channel`` across target positions
    ``position_start``..``position_end``. ``direction`` fixes its sign so the summed objective is
    always minimized, and ``weight`` scales it before the terms are added together.

    Attributes:
        channel (SpliceTransformerChannel): Output channel to optimize — a splice class
            (``NEITHER``/``ACCEPTOR``/``DONOR``, softmaxed across the three) or a tissue
            (sigmoid-activated).
        position_start (int | None): First target position to average, 1-indexed and inclusive.
            ``None`` starts at the first target position.
        position_end (int | None): Last target position to average, 1-indexed and inclusive.
            ``None`` ends at the last target position.
        direction (SpliceTransformerObjectiveDirection): ``"max"`` minimizes ``-mean_probability``;
            ``"min"`` minimizes ``+mean_probability``.
        weight (float): Non-negative scalar applied before terms are summed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    channel: SpliceTransformerChannel = Field(
        default="ACCEPTOR",
        title="Channel",
        description="Output channel to optimize: a splice class or a tissue-specific channel",
    )
    position_start: int | None = Field(
        default=None,
        ge=1,
        le=TARGET_LENGTH,
        title="Position Start",
        description="First target position to average, 1-indexed inclusive; None starts at the first",
    )
    position_end: int | None = Field(
        default=None,
        ge=1,
        le=TARGET_LENGTH,
        title="Position End",
        description="Last target position to average, 1-indexed inclusive; None ends at the last",
    )
    direction: SpliceTransformerObjectiveDirection = Field(
        default="max",
        title="Direction",
        description="Whether to maximize ('max') or minimize ('min') mean probability for this channel",
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
                f"splice-transformer gradient: position_end ({self.position_end}) precedes "
                f"position_start ({self.position_start})"
            )
        return self


class SpliceTransformerGradientCandidate(BaseModel):
    """One candidate: a relaxed target region between its two fixed flanking contexts.

    SpliceTransformer scores a 1,000 bp target region in the presence of 4,000 bp of flanking
    context on each side. The contexts belong to the candidate, so a batch may score different
    loci, and only the target region is relaxed and differentiated.

    Attributes:
        left_context (str): 5' flanking context, exactly ``CONTEXT_LENGTH`` bp.
        right_context (str): 3' flanking context, exactly ``CONTEXT_LENGTH`` bp.
        logits (NucleotideSequenceLogits): Relaxed ``(TARGET_LENGTH, 4)`` target state in
            ``A,C,G,T`` order.
    """

    model_config = ConfigDict(extra="forbid")

    left_context: str = Field(
        title="Left Context",
        description="5' flanking context; exactly 4000 bp",
    )
    right_context: str = Field(
        title="Right Context",
        description="3' flanking context; exactly 4000 bp",
    )
    logits: NucleotideSequenceLogits = Field(
        title="Logits",
        description="Relaxed (1000, 4) target-region state in A,C,G,T order",
    )

    def iter_sequence_logits(self) -> Iterator[tuple[str, SequenceLogits]]:
        """Expose this candidate's relaxed state to the shared validation hooks.

        Yields:
            tuple[str, SequenceLogits]: The field name and this candidate's matrix.
        """
        yield "logits", self.logits

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Require the published context and target lengths the checkpoint was trained on."""
        if len(self.left_context) != CONTEXT_LENGTH:
            raise ValueError(
                f"splice-transformer gradient: left_context must be exactly {CONTEXT_LENGTH} bp, "
                f"got {len(self.left_context)}"
            )
        if len(self.right_context) != CONTEXT_LENGTH:
            raise ValueError(
                f"splice-transformer gradient: right_context must be exactly {CONTEXT_LENGTH} bp, "
                f"got {len(self.right_context)}"
            )
        if len(self.logits) != TARGET_LENGTH:
            raise ValueError(
                f"splice-transformer gradient: logits must cover exactly {TARGET_LENGTH} target "
                f"positions, got {len(self.logits)}"
            )
        return self


class SpliceTransformerGradientInput(GradientInput):
    """Input for the SpliceTransformer gradient tool: one or many relaxed candidates.

    Attributes:
        inputs (list[SpliceTransformerGradientCandidate]): One target-and-contexts bundle per
            candidate.
        temperature (float | None): Softmax temperature, shared by every candidate. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[SpliceTransformerGradientCandidate] = InputField(
        title="Relaxed Candidates",
        description="One relaxed target region with its flanking contexts per candidate",
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
    ) -> SpliceTransformerGradientCandidate:
        """Build one candidate pairing the converted target matrix with its contexts.

        Args:
            probabilities (Any): ``(TARGET_LENGTH, 4)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``left_context`` and
                ``right_context``.

        Returns:
            SpliceTransformerGradientCandidate: Candidate reproducing ``probabilities``.

        Raises:
            ValueError: If either flanking context is missing.
        """
        missing = {"left_context", "right_context"} - set(fields)
        if missing:
            raise ValueError(f"splice-transformer gradient from_probabilities requires {sorted(missing)}")
        matrix = NucleotideSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return SpliceTransformerGradientCandidate(logits=matrix, **fields)


class SpliceTransformerGradientSampleMetrics(Metrics):
    """Per-candidate differentiable SpliceTransformer objective metrics.

    Attributes:
        loss_terms (list[dict[str, Any]]): Per-objective-term metadata — channel, position span,
            direction, weight, mean probability, and the weighted contribution.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "loss": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Weighted scalar SpliceTransformer objective value for one candidate.",
            "better_values_are": "lower",
        },
        "target_length": {
            "availability": "always",
            "type": "int",
            "min": 1,
            "max": None,
            "description": "Number of scored target positions for this candidate.",
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
        description="Per-objective-term probability and weighting metadata for this candidate.",
    )


class SpliceTransformerGradientOutput(GradientOutput):
    """SpliceTransformer weighted-objective output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input candidate, in input order, each holding
            that candidate's ``(TARGET_LENGTH, 4)`` gradient (``None`` in forward-only mode), its
            weighted objective value, and its per-term probability metrics.
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """


class SpliceTransformerGradientConfig(RelaxationConfig):
    """Configuration for the SpliceTransformer splice-channel gradient tool.

    Attributes:
        loss_terms (list[SpliceTransformerGradientLossTerm]): Objective terms summed into the
            scalar loss.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    loss_terms: list[SpliceTransformerGradientLossTerm] = ConfigField(
        title="Loss Terms",
        default_factory=lambda: [SpliceTransformerGradientLossTerm()],
        min_length=1,
        description="Channel-and-position objective terms summed into the scalar loss",
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


def example_input() -> SpliceTransformerGradientInput:
    """Minimal valid input for testing and examples."""
    flank = ("ACGT" * (CONTEXT_LENGTH // 4 + 1))[:CONTEXT_LENGTH]
    return SpliceTransformerGradientInput(
        inputs=[
            SpliceTransformerGradientCandidate(
                left_context=flank,
                right_context=flank,
                logits=NucleotideSequenceLogits([[0.25] * 4] * TARGET_LENGTH),
            )
        ]
    )


@tool(
    key="splice-transformer-gradient",
    label="SpliceTransformer Gradient",
    category="rna_splicing",
    input_class=SpliceTransformerGradientInput,
    config_class=SpliceTransformerGradientConfig,
    output_class=SpliceTransformerGradientOutput,
    description="Compute SpliceTransformer splice-channel gradients for a relaxed target region",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=4,
    metrics_class=SpliceTransformerGradientSampleMetrics,
)
def run_splice_transformer_gradient(
    inputs: SpliceTransformerGradientInput,
    config: SpliceTransformerGradientConfig,
    instance: Any = None,
) -> SpliceTransformerGradientOutput:
    """Compute SpliceTransformer weighted channel gradients with respect to relaxed target logits."""
    logger.debug("Using local worker for SpliceTransformer gradient (%d candidate(s))", len(inputs.inputs))
    result = ToolInstance.dispatch(
        "splice_transformer",
        {
            "operation": "compute_gradient",
            "items": [
                {
                    "left_context": candidate.left_context,
                    "right_context": candidate.right_context,
                    "logits": candidate.logits.root,
                }
                for candidate in inputs.inputs
            ],
            "loss_terms": [
                {**term.model_dump(), "channel_index": SPLICE_CHANNEL_INDEX[term.channel]} for term in config.loss_terms
            ],
            "context_length": CONTEXT_LENGTH,
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
    return SpliceTransformerGradientOutput(**result)
