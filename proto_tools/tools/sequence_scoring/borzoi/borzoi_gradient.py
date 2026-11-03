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


class BorzoiActivityTerm(DifferentiableTerm):
    """One differentiable Borzoi quantity: mean predicted activity over a track and bin span.

    Whether more activity is wanted depends on whether the track is a target or an off-target,
    and how much it matters relative to other tracks is a weight — both are goals a caller
    applies to the returned value, not properties of the quantity.

    Attributes:
        id (Literal["track_activity"]): Fixed identifier for this kind of quantity.
        track (int): Borzoi output-track index (0-based, as the prediction tool reports them).
        bin_start (int): First output bin to average, 1-indexed and inclusive.
        bin_end (int): Last output bin to average, 1-indexed and inclusive.
    """

    #: A track can be something to drive up or something to suppress; only the caller knows.
    SUGGESTED_OPTIMIZATION_DIRECTION: ClassVar[Directionality] = "context-dependent"

    id: Literal["track_activity"] = Field(
        default="track_activity",
        title="Term ID",
        description="Identifies this quantity as Borzoi mean track activity over a bin span.",
    )
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

    @model_validator(mode="after")
    def validate_bin_span(self) -> Self:
        """Require a non-empty, ordered bin span."""
        if self.bin_end < self.bin_start:
            raise ValueError(f"borzoi gradient: bin_end ({self.bin_end}) precedes bin_start ({self.bin_start})")
        return self


class BorzoiTermGradient(TermGradient):
    """One Borzoi track's mean activity and its raw gradient.

    Attributes:
        term (BorzoiActivityTerm): The track-and-bin quantity this gradient answers.
    """

    term: BorzoiActivityTerm = Field(title="Term", description="Track-and-bin quantity answered.")


class BorzoiGradientResult(GradientResult):
    """One evaluated candidate and every track activity differentiated for it.

    Attributes:
        terms (list[BorzoiTermGradient]): One entry per requested track, in the order requested.
    """

    terms: list[BorzoiTermGradient] = Field(  # type: ignore[assignment]  # narrows the element type
        default_factory=list,
        title="Terms",
        description="One gradient per requested track-and-bin quantity, in the order requested.",
    )


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
        designed_positions (int): Number of relaxed context positions this candidate carries.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
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
        default="designed_positions",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )


class BorzoiGradientOutput(GradientOutput):
    """Borzoi track-activity output; gradients are optional in forward-only mode.

    Attributes:
        results (list[BorzoiGradientResult]): One per input candidate, in input order, each
            holding that candidate's requested track activities and their raw gradients w.r.t.
            its relaxed positions (``None`` in forward-only mode).
        vocab (list[str]): Nucleotide column ordering for the input logits.
    """

    results: list[BorzoiGradientResult] = Field(  # type: ignore[assignment]  # narrows the element type
        title="Results",
        description="One result per input candidate, in input order, each carrying its track activities.",
    )


class BorzoiGradientConfig(GradientConfig):
    """Configuration for the Borzoi regulatory-activity gradient tool.

    Attributes:
        gradient_terms (list[BorzoiActivityTerm]): Track-and-bin quantities to differentiate,
            given as the catalog name (``"track_activity"``) or as instances naming a track and
            bin span. Each comes back with its own raw activity and gradient.
        species (Literal['human', 'mouse']): Species model to use.
        replicate (Literal['0', '1', '2', '3']): Replicate ID to run.
        soft (float): Inherited blend from the raw input matrix (0) to softmax probabilities
            (1). Recommended to leave at 1.0 in almost all use cases; exposed for research
            and ColabDesign parity. See
            :class:`~proto_tools.utils.gradient.models.RelaxationConfig`.
        hard (float): Inherited straight-through coefficient; 1 forwards a hard one-hot.
        compute_gradient (bool): Run backward pass and return gradient. Set ``False`` for
            forward-only objective scoring.
        device (str): Device to run the model on.
    """

    TERMS: ClassVar[TermCatalog] = {"track_activity": BorzoiActivityTerm}

    gradient_terms: list[BorzoiActivityTerm] = ConfigField(
        title="Gradient Terms",
        default_factory=lambda: [BorzoiActivityTerm()],
        min_length=1,
        description="Track-and-bin quantities to differentiate, by name or as instances",
    )

    @field_validator("gradient_terms", mode="before")
    @classmethod
    def resolve_gradient_terms(cls, value: Any) -> Any:
        """Accept the catalog name and the term class alongside instances.

        A mapping is what a round-tripped config looks like, and Pydantic reads those back on
        its own.
        """
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
        """The same track and bin span twice would be one quantity asked for twice."""
        asked = [(term.track, term.bin_start, term.bin_end) for term in self.gradient_terms]
        repeated = sorted({item for item in asked if asked.count(item) > 1})
        if repeated:
            raise ValueError(f"gradient_terms names the same track and bin span more than once: {repeated}")
        return self

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
    """Compute Borzoi track-activity gradients with respect to relaxed DNA logits."""
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
            "gradient_terms": [term.model_dump() for term in config.gradient_terms],
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

    # Every track's gradient is w.r.t. the candidate's relaxed rows inside the context window.
    results = []
    for candidate, item in zip(inputs.inputs, result["results"], strict=True):
        basis = GradientMetadata.from_input(
            candidate.logits.root,
            vocab=result["vocab"],
            temperature=inputs.temperature,
            description="Mean predicted regulatory activity over a Borzoi track and bin span",
            positions=candidate.positions,
            soft=config.soft,
            hard=config.hard,
        )
        results.append(
            BorzoiGradientResult(
                terms=[
                    BorzoiTermGradient(
                        term=BorzoiActivityTerm(**entry["term"]),
                        term_value=float(entry["value"]),
                        gradient=entry["gradient"],
                        metadata=basis,
                    )
                    for entry in item["terms"]
                ],
                metrics=item["metrics"],
            )
        )
    return BorzoiGradientOutput(results=results, vocab=result["vocab"])
