"""IgLM antibody span re-design (infilling) tool."""

import logging
from typing import Any

from pydantic import Field, field_validator, model_validator

from proto_tools.tools.causal_models.iglm.shared_data_models import (
    IgLMConditioningInput,
    IgLMGenerationConfig,
    IgLMGenerationOutput,
    IgLMSamples,
    IgLMSpan,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import InputField, ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class IgLMInfillInput(IgLMConditioningInput):
    """Input for IgLM span re-design.

    Attributes:
        sequences (list[str]): Parent antibody sequence(s) to re-design.
        spans (list[IgLMSpan]): Residue span replaced in each parent sequence
            (1-indexed, inclusive).
        chains (list[IgLMChain]): Chain type conditioning each sequence.
        species (list[IgLMSpecies]): Species-of-origin conditioning each sequence.
    """

    sequences: list[str] = InputField(
        title="Sequences",
        description="Parent antibody sequence(s) whose span is re-designed",
        min_length=1,
        examples=["EVQLVESGGGLVQPGGSLRLSCAAS"],
    )
    spans: list[IgLMSpan] = InputField(
        title="Spans",
        description="Residue span re-designed in each parent sequence (1-indexed, inclusive)",
    )
    """One span per sequence, deliberately not broadcastable.

    ``iglm-score`` broadcasts a span because mis-locating a scored window produces a
    number you can inspect; here the span selects residues to overwrite, so applying one
    parent's coordinates to a differently numbered parent silently re-designs the wrong
    region and returns designs that look fine.
    """

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_sequences(cls, v: Any) -> Any:
        """Convert a single sequence to a one-element list."""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("spans", mode="before")
    @classmethod
    def normalize_spans(cls, v: Any) -> Any:
        """Convert a single span to a one-element list."""
        if isinstance(v, IgLMSpan | dict):
            return [v]
        return v

    @model_validator(mode="after")
    def validate_spans_against_sequences(self) -> "IgLMInfillInput":
        """Pair one in-bounds span with every sequence, then broadcast the conditioning."""
        if len(self.spans) != len(self.sequences):
            raise ValueError(
                f"iglm-infill: got {len(self.spans)} spans for {len(self.sequences)} sequences; "
                f"supply one span per sequence."
            )
        for i, (sequence, span) in enumerate(zip(self.sequences, self.spans, strict=True)):
            if span.end > len(sequence):
                raise ValueError(
                    f"iglm-infill: span {span.start}-{span.end} runs past the end of sequence {i} "
                    f"(length {len(sequence)})."
                )
        return self.broadcast_parallel_fields("sequences")  # type: ignore[return-value]


class IgLMInfilledSequences(IgLMSamples):
    """The re-designed sequences IgLM generated for one parent sequence.

    Attributes:
        sequences (list[str]): Full re-designed sequences, in generation order.
        infilled_residues (list[str]): Residues IgLM wrote into the span, aligned with
            ``sequences``. Lengths vary; IgLM chooses how long the new span is.
        spans (list[IgLMSpan]): Where those residues sit in each returned sequence,
            ready to hand straight back to ``iglm-score``.
    """

    infilled_residues: list[str] = Field(
        title="Infilled Residues",
        description="Residues written into the span, aligned with 'sequences'",
    )
    spans: list[IgLMSpan] = Field(
        title="Spans",
        description="Where the written residues sit in each sequence (1-indexed, inclusive)",
    )

    @model_validator(mode="after")
    def validate_designs_are_aligned(self) -> "IgLMInfilledSequences":
        """Keep the three per-design lists the same length, including after ``model_validate``.

        Chunk merging, cache round-trips, and worker payloads all rebuild this model from a
        dict, where a mismatch would silently pair a design with another design's span.
        """
        lengths = {len(self.sequences), len(self.infilled_residues), len(self.spans)}
        if len(lengths) != 1:
            raise ValueError(
                f"iglm-infill: got {len(self.sequences)} sequences, {len(self.infilled_residues)} "
                f"infilled_residues and {len(self.spans)} spans; the three must be aligned per design."
            )
        return self

    def fasta_headers(self, input_index: int) -> list[str]:
        """Name the re-designed span in each header so the default export is self-describing."""
        return [
            f"input_{input_index}_seq_{j} span={span.start}-{span.end} residues={residues}"
            for j, (span, residues) in enumerate(zip(self.spans, self.infilled_residues, strict=True))
        ]


class IgLMInfillOutput(IgLMGenerationOutput):
    """Output from IgLM span re-design.

    Attributes:
        results (list[IgLMInfilledSequences]): One bundle per parent sequence, in
            input order, holding the re-designed sequences, the residues IgLM wrote,
            and where each new span sits.
    """

    results: list[IgLMInfilledSequences] = Field(  # type: ignore[assignment]
        title="Results",
        description="Re-designed sequences bundled per parent sequence, in input order",
    )


# Config:
class IgLMInfillConfig(IgLMGenerationConfig):
    """Configuration for IgLM span re-design.

    Attributes:
        model_checkpoint (IGLM_MODEL_CHECKPOINTS): IgLM weights variant.
        num_sequences_per_input (int): Distinct sequences to generate per parent sequence.
        temperature (float): Softmax temperature; lower is more conservative.
        top_p (float): Nucleus sampling cutoff over per-position token probabilities.
        max_length (int): Token budget per generation, including conditioning tokens
            and the masked parent sequence.
        batch_size (int): Candidate sequences per forward pass.
        device (str): Device to run the model on.
    """


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> IgLMInfillInput:
    """Minimal valid input for testing and examples."""
    # 1JPT heavy chain; residues 99-106 are its CDR H3 (DTAAYFDY).
    return IgLMInfillInput(
        sequences=[
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATIS"
            "ADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
        ],
        spans=[IgLMSpan(start=99, end=106)],
        chains=["heavy"],
        species=["human"],
    )


@tool(
    key="iglm-infill",
    label="IgLM Span Re-design",
    category="causal_models",
    input_class=IgLMInfillInput,
    config_class=IgLMInfillConfig,
    output_class=IgLMInfillOutput,
    description="Re-design a span of an antibody sequence with IgLM, conditioned on the rest of the chain",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences", "spans", "chains", "species"],
    iterable_output_field="results",
    max_chunk_size=8,
    cacheable=True,
    stochastic=True,
)
def run_iglm_infill(
    inputs: IgLMInfillInput,
    config: IgLMInfillConfig,
    instance: Any = None,
) -> IgLMInfillOutput:
    """Re-design a span of an antibody sequence with IgLM.

    The span is replaced by a mask token and IgLM writes a new segment
    conditioned on the surrounding residues plus the chain and species tokens —
    the infilling objective it was trained on, and the usual way to diversify a
    CDR loop while keeping its framework intact. The new segment is not
    constrained to the original length.

    Args:
        inputs (IgLMInfillInput): Parent sequences, the span to re-design in each
            (1-indexed, inclusive), and the per-sequence chain and species conditioning.
        config (IgLMInfillConfig): Checkpoint and sampling configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        IgLMInfillOutput: One ``IgLMInfilledSequences`` bundle per parent sequence,
            in input order, holding the full re-designed sequences, the residues
            written into the span, and where each new span sits.

    Examples:
        >>> # Re-design CDR H3 (residues 99-106) of the 1JPT heavy chain
        >>> parent = (
        ...     "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATIS"
        ...     "ADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
        ... )
        >>> inputs = IgLMInfillInput(sequences=[parent], spans=[IgLMSpan(start=99, end=106)])
        >>> result = run_iglm_infill(inputs, IgLMInfillConfig(num_sequences_per_input=100))
        >>> print(result.results[0].infilled_residues[:5])
        >>> # Rank the designs by the likelihood of what IgLM wrote
        >>> designs = result.results[0]
        >>> scored = run_iglm_score(
        ...     IgLMScoringInput(sequences=designs.sequences, spans=designs.spans), IgLMScoringConfig()
        ... )

    Note:
        - Sequences within a bundle are distinct; malformed generations are discarded,
          mirroring upstream's rejection sampling.
        - Coordinates are 1-indexed and inclusive, so upstream's ``(98, 106)`` is
          ``IgLMSpan(start=99, end=106)``.
    """
    logger.debug("Using local venv for IgLM infilling: %s", config.model_checkpoint)
    result = ToolInstance.dispatch(
        "iglm",
        {
            "operation": "infill",
            "sequences": inputs.sequences,
            # Upstream masks with a 0-indexed, end-exclusive range.
            "infill_ranges": [[span.start - 1, span.end] for span in inputs.spans],
            "chain_tokens": inputs.chain_tokens(),
            "species_tokens": inputs.species_tokens(),
            "model_checkpoint": config.model_checkpoint,
            "num_sequences_per_input": config.num_sequences_per_input,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    return IgLMInfillOutput(
        metadata={
            "model_checkpoint": config.model_checkpoint,
            "num_sequences_per_input": config.num_sequences_per_input,
            "temperature": config.temperature,
        },
        results=[
            IgLMInfilledSequences(
                sequences=sequences,
                infilled_residues=residues,
                # IgLM picks the new span's length, so each design ends somewhere different.
                spans=[
                    IgLMSpan(start=parent_span.start, end=parent_span.start + len(written) - 1) for written in residues
                ],
            )
            for parent_span, sequences, residues in zip(
                inputs.spans, result["sequences"], result["infilled_residues"], strict=True
            )
        ],
    )
