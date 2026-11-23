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
from proto_tools.utils.sequence_logits import SequenceLogits
from proto_tools.utils.tool_io import Directionality, Metrics, MetricSpec

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


class SpliceTransformerProbabilityTerm(DifferentiableTerm):
    """Mean predicted probability for one SpliceTransformer channel.

    Whether more probability is wanted depends on whether the channel is a target or an
    off-target, and how much it matters relative to other channels is a weight — both are goals
    a caller applies to the returned value, not properties of the quantity.

    An open-ended span is resolved against the target window, so the term that comes back names
    the positions actually averaged rather than the ``None`` that was asked for.

    Attributes:
        id (Literal["channel_probability"]): Fixed identifier for this kind of quantity.
        channel (SpliceTransformerChannel): Output channel to optimize — a splice class
            (``NEITHER``/``ACCEPTOR``/``DONOR``, softmaxed across the three) or a tissue
            (sigmoid-activated).
        position_start (int | None): First target position to average, 1-indexed and inclusive.
            ``None`` starts at the first target position.
        position_end (int | None): Last target position to average, 1-indexed and inclusive.
            ``None`` ends at the last target position.
    """

    #: A channel can be something to drive up or something to suppress; only the caller knows.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    #: What this quantity measures, copied into each gradient's metadata.
    DESCRIPTION: ClassVar[str] = (
        "Mean predicted SpliceTransformer probability for a channel over a target position span"
    )

    id: Literal["channel_probability"] = Field(
        default="channel_probability",
        title="Term ID",
        description="Identifies this quantity as a SpliceTransformer mean channel probability.",
    )
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
    """One relaxed target region between its two fixed flanking contexts.

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
    """One or more relaxed candidates for the SpliceTransformer gradient tool.

    Attributes:
        inputs (list[SpliceTransformerGradientCandidate]): One target-and-contexts bundle per
            candidate.
        temperature (float | None): Softmax temperature, shared by every candidate. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients. ``None`` reads them as probabilities and returns
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
            fields (Any): This candidate's remaining fields, which must include ``left_context`` and
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


class SpliceTransformerTermGradient(TermGradient):
    """One SpliceTransformer channel's mean probability and its raw gradient.

    Attributes:
        term (SpliceTransformerProbabilityTerm): The channel-and-position quantity answered.
    """

    term: SpliceTransformerProbabilityTerm = Field(title="Term", description="Channel-and-position quantity answered.")


class SpliceTransformerGradientResult(GradientResult):
    """One evaluated candidate and every channel probability differentiated for it.

    Attributes:
        terms (list[SpliceTransformerTermGradient]): One entry per requested channel, in order.
    """

    terms: list[SpliceTransformerTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Terms",
        description="One gradient per requested channel-and-position quantity, in the order requested.",
    )


class SpliceTransformerGradientSampleMetrics(Metrics):
    """Per-candidate differentiable SpliceTransformer objective metrics.

    Attributes:
        target_length (int): Number of scored target positions for this candidate.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
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
        default="target_length",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class SpliceTransformerGradientOutput(GradientOutput):
    """SpliceTransformer channel probabilities and requested gradients. Gradients are omitted in forward-only mode.

    Attributes:
        results (list[SpliceTransformerGradientResult]): One per input candidate, in input order,
            each holding that candidate's requested channel probabilities and their raw
            ``(TARGET_LENGTH, 4)`` gradients (``None`` in forward-only mode).
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """

    results: list[SpliceTransformerGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input candidate, in input order, each carrying its channel probabilities.",
    )


class SpliceTransformerGradientConfig(GradientConfig):
    """Configuration for the SpliceTransformer splice-channel gradient tool.

    Attributes:
        gradient_terms (list[SpliceTransformerProbabilityTerm]): Channel-and-position quantities
            to differentiate.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases. Exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient. A value of 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    TERMS: ClassVar[TermCatalog] = {"channel_probability": SpliceTransformerProbabilityTerm}

    gradient_terms: list[SpliceTransformerProbabilityTerm] = ConfigField(
        title="Gradient Terms",
        default_factory=lambda: [SpliceTransformerProbabilityTerm()],
        min_length=1,
        description="Channel-and-position quantities to differentiate, by name or as instances",
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
        """The same channel and position span twice would be one quantity asked for twice."""
        asked = [(term.channel, term.position_start, term.position_end) for term in self.gradient_terms]
        repeated = sorted({item for item in asked if asked.count(item) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same channel and position span more than once: {repeated}")
        return self


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
    """Report each requested SpliceTransformer channel probability and its gradient."""
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
            "gradient_terms": [
                {**term.model_dump(), "channel_index": SPLICE_CHANNEL_INDEX[term.channel]}
                for term in config.gradient_terms
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

    # Only the target window is relaxed, so a gradient covers the candidate's own rows.
    results = []
    for candidate, item in zip(inputs.inputs, result["results"], strict=True):
        basis = GradientMetadata.from_input(
            candidate.logits.root,
            vocab=result["vocab"],
            temperature=inputs.temperature,
            soft=config.soft,
            hard=config.hard,
        )
        results.append(
            SpliceTransformerGradientResult(
                terms=[
                    SpliceTransformerTermGradient(
                        term=SpliceTransformerProbabilityTerm(**entry["term"]),
                        term_value=float(entry["value"]),
                        gradient=entry["gradient"],
                        metadata=basis,
                    )
                    for entry in item["terms"]
                ],
                metrics=item["metrics"],
            )
        )
    return SpliceTransformerGradientOutput(results=results, vocab=result["vocab"])
