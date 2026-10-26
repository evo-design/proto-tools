"""Borzoi regulatory-activity gradient tool."""

import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from proto_tools.tools.sequence_scoring.borzoi.borzoi_prediction import (
    BORZOI_CONTEXT,
    BORZOI_OUTPUT,
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

BorzoiObjectiveDirection = Literal["max", "min"]


class BorzoiGradientLossTerm(BaseModel):
    """One differentiable Borzoi track-and-bin objective term.

    The term's score is the mean predicted activity over ``track`` across bins
    ``bin_start``..``bin_end``. ``direction`` fixes its sign so the summed objective is always
    minimized, and ``weight`` scales it before the terms are added together.

    Attributes:
        track (int): Borzoi output-track index (0-based, as the prediction tool reports them).
        bin_start (int): First output bin to average, 1-indexed and inclusive.
        bin_end (int): Last output bin to average, 1-indexed and inclusive.
        direction (BorzoiObjectiveDirection): ``"max"`` minimizes ``-mean_activity``; ``"min"``
            minimizes ``+mean_activity``.
        weight (float): Non-negative scalar applied before terms are summed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    track: int = Field(
        default=0,
        ge=0,
        title="Track",
        description="Borzoi output-track index to optimize (7611 human / 2608 mouse tracks)",
    )
    bin_start: int = Field(
        default=1,
        ge=1,
        le=BORZOI_OUTPUT,
        title="Bin Start",
        description="First output bin to average, 1-indexed and inclusive",
    )
    bin_end: int = Field(
        default=BORZOI_OUTPUT,
        ge=1,
        le=BORZOI_OUTPUT,
        title="Bin End",
        description="Last output bin to average, 1-indexed and inclusive",
    )
    direction: BorzoiObjectiveDirection = Field(
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
            raise ValueError(f"borzoi gradient: bin_end ({self.bin_end}) precedes bin_start ({self.bin_start})")
        return self


class BorzoiGradientCandidate(BaseModel):
    """One candidate: relaxed bases at selected positions inside a full Borzoi context window.

    Borzoi reads a fixed 524,288 bp context, but a design problem usually varies only a small
    part of it. The candidate therefore carries the whole context as a sequence plus the
    positions whose bases are relaxed, so the gradient comes back only for those positions and
    the surrounding context still reaches the model.

    Attributes:
        sequence (str): Full ``524288`` bp context window; the designed positions may hold any
            placeholder base, since ``logits`` replaces them.
        positions (list[int]): Context positions the relaxed rows correspond to, 1-indexed,
            inclusive, strictly increasing, one per row of ``logits``.
        logits (NucleotideSequenceLogits): Relaxed ``(len(positions), 4)`` state in ``A,C,G,T``
            order, aligned to ``positions``.
    """

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(
        title="Context Sequence",
        description="Full 524288 bp Borzoi context window holding the designed positions",
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
        if len(self.sequence) != BORZOI_CONTEXT:
            raise ValueError(f"borzoi gradient: sequence must be exactly {BORZOI_CONTEXT} bp, got {len(self.sequence)}")
        if len(self.positions) != len(self.logits):
            raise ValueError(
                f"borzoi gradient: positions ({len(self.positions)}) and logits ({len(self.logits)}) must align"
            )
        if any(later <= earlier for earlier, later in zip(self.positions, self.positions[1:], strict=False)):
            raise ValueError("borzoi gradient: positions must be strictly increasing with no duplicates")
        if self.positions[-1] > BORZOI_CONTEXT:
            raise ValueError(f"borzoi gradient: position {self.positions[-1]} exceeds the {BORZOI_CONTEXT} bp context")
        return self


class BorzoiGradientInput(GradientInput):
    """Input for the Borzoi gradient tool: one or many relaxed candidates.

    Attributes:
        inputs (list[BorzoiGradientCandidate]): One context-and-relaxed-positions bundle per
            candidate.
        temperature (float | None): Softmax temperature, shared by every candidate. The default
            ``1.0`` reads each state as unnormalized and applies ``softmax(logits / T)``,
            returning logit-space gradients; ``None`` reads them as probabilities and returns
            probability-space gradients.
    """

    inputs: list[BorzoiGradientCandidate] = InputField(
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
    ) -> BorzoiGradientCandidate:
        """Build one candidate pairing the converted matrix with its context window.

        Args:
            probabilities (Any): ``(len(positions), 4)`` strictly positive rows.
            temperature (float): Positive, finite temperature the matrix is converted at.
            fields (Any): This candidate's remaining fields; must include ``sequence`` and
                ``positions``.

        Returns:
            BorzoiGradientCandidate: Candidate whose logits reproduce ``probabilities``.

        Raises:
            ValueError: If ``sequence`` or ``positions`` is missing.
        """
        missing = {"sequence", "positions"} - set(fields)
        if missing:
            raise ValueError(f"borzoi gradient from_probabilities requires {sorted(missing)}")
        matrix = NucleotideSequenceLogits.from_probabilities(probabilities, temperature=temperature)
        return BorzoiGradientCandidate(logits=matrix, **fields)


class BorzoiGradientSampleMetrics(Metrics):
    """Per-candidate differentiable Borzoi objective metrics.

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
            "description": "Weighted scalar Borzoi objective value for one candidate.",
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


class BorzoiGradientOutput(GradientOutput):
    """Borzoi weighted-objective output; gradient is optional in forward-only mode.

    Attributes:
        results (list[GradientResult]): One per input candidate, in input order, each holding
            that candidate's gradient w.r.t. its relaxed positions (``None`` in forward-only
            mode), its weighted objective value, and its per-term activity metrics.
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """


class BorzoiGradientConfig(RelaxationConfig):
    """Configuration for the Borzoi regulatory-activity gradient tool.

    Attributes:
        loss_terms (list[BorzoiGradientLossTerm]): Objective terms summed into the scalar loss.
        species (Literal['human', 'mouse']): Species model to use.
        replicate (Literal['0', '1', '2', '3']): Replicate ID to run.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1); keep the default 1.0 unless deliberately experimenting. See
            :class:`~proto_tools.utils.gradient_models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    loss_terms: list[BorzoiGradientLossTerm] = ConfigField(
        title="Loss Terms",
        default_factory=lambda: [BorzoiGradientLossTerm()],
        min_length=1,
        description="Track-and-bin objective terms summed into the scalar loss",
    )
    species: Literal["human", "mouse"] = ConfigField(
        title="Species",
        default="human",
        description="Species model to use",
        reload_on_change=True,
    )
    replicate: Literal["0", "1", "2", "3"] = ConfigField(
        title="Replicate",
        default="0",
        description="Replicate ID to run",
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


def example_input() -> BorzoiGradientInput:
    """Minimal valid input for testing and examples."""
    sequence = "A" * BORZOI_CONTEXT
    positions = [BORZOI_CONTEXT // 2, BORZOI_CONTEXT // 2 + 1]
    return BorzoiGradientInput(
        inputs=[
            BorzoiGradientCandidate(
                sequence=sequence,
                positions=positions,
                logits=NucleotideSequenceLogits([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            )
        ]
    )


@tool(
    key="borzoi-gradient",
    label="Borzoi Gradient",
    category="sequence_scoring",
    input_class=BorzoiGradientInput,
    config_class=BorzoiGradientConfig,
    output_class=BorzoiGradientOutput,
    description="Compute Borzoi track-and-bin activity gradients for relaxed DNA positions",
    uses_gpu=True,
    example_input=example_input,
    cacheable=False,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=1,
    metrics_class=BorzoiGradientSampleMetrics,
)
def run_borzoi_gradient(
    inputs: BorzoiGradientInput,
    config: BorzoiGradientConfig,
    instance: Any = None,
) -> BorzoiGradientOutput:
    """Compute Borzoi weighted track-activity gradients with respect to relaxed DNA logits."""
    use_flash_attn = config.species == "human"
    if use_flash_attn and not config.device.startswith("cuda"):
        raise ValueError("Must run on GPU to use FlashAttention with Borzoi")

    logger.debug("Using local worker for Borzoi gradient (%d candidate(s))", len(inputs.inputs))
    result = ToolInstance.dispatch(
        "borzoi",
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
            "replicate": config.replicate,
            "use_flash_attn": use_flash_attn,
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        },
        instance=instance,
        config=config,
    )
    return BorzoiGradientOutput(**result)
