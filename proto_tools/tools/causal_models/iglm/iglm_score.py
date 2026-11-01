"""IgLM antibody sequence likelihood scoring tool."""

import logging
from typing import Any

from pydantic import field_validator, model_validator

from proto_tools.tools.causal_models.iglm.shared_data_models import (
    IGLM_MODEL_CHECKPOINTS,
    IgLMConditioningInput,
    IgLMSpan,
)
from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringConfig,
    CausalModelScoringInput,
    CausalModelScoringMetrics,
    CausalModelScoringOutput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, InputField, ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class IgLMScoringInput(CausalModelScoringInput, IgLMConditioningInput):
    """Input for IgLM antibody sequence scoring.

    Attributes:
        sequences (list[str]): Antibody sequence(s) to score.
        spans (list[IgLMSpan | None] | None): Optional span to score conditionally in
            each sequence (1-indexed, inclusive); ``None`` scores the whole sequence.
        chains (list[IgLMChain]): Chain type conditioning each sequence.
        species (list[IgLMSpecies]): Species-of-origin conditioning each sequence.
    """

    spans: list[IgLMSpan | None] | None = InputField(
        default=None,
        broadcastable=True,
        title="Spans",
        description="Span to score conditionally per sequence (1-indexed, inclusive); None scores all",
    )

    @field_validator("spans", mode="before")
    @classmethod
    def normalize_spans(cls, v: Any) -> Any:
        """Convert a single span to a one-element list."""
        if isinstance(v, IgLMSpan | dict):
            return [v]
        return v

    @model_validator(mode="after")
    def validate_spans_against_sequences(self) -> "IgLMScoringInput":
        """Broadcast the parallel fields, then keep every span inside its sequence."""
        broadcast: IgLMScoringInput = self.broadcast_parallel_fields("sequences")  # type: ignore[assignment]
        if broadcast.spans is not None:
            for i, (sequence, span) in enumerate(zip(broadcast.sequences, broadcast.spans, strict=True)):
                if span is not None and span.end > len(sequence):
                    raise ValueError(
                        f"iglm-score: span {span.start}-{span.end} runs past the end of sequence {i} "
                        f"(length {len(sequence)})."
                    )
        return broadcast


# Output:
IgLMScoringOutput = CausalModelScoringOutput


# Config:
class IgLMScoringConfig(CausalModelScoringConfig):
    """Configuration for IgLM antibody sequence scoring.

    Attributes:
        model_checkpoint (IGLM_MODEL_CHECKPOINTS): IgLM weights variant.
        batch_size (int): Sequences per forward pass.
        return_logits (bool): Include per-position logits over IgLM's 33-token vocabulary.
        device (str): Device to run the model on.
    """

    model_checkpoint: IGLM_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="IgLM",
        description="IgLM weights variant; 'IgLM' is 13M parameters, 'IgLM-S' is 1.4M",
        reload_on_change=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> IgLMScoringInput:
    """Minimal valid input for testing and examples."""
    return IgLMScoringInput(
        sequences=[
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATIS"
            "ADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
        ],
        chains=["heavy"],
        species=["human"],
    )


@tool(
    key="iglm-score",
    label="IgLM Scoring",
    category="causal_models",
    input_class=IgLMScoringInput,
    config_class=IgLMScoringConfig,
    output_class=IgLMScoringOutput,
    metrics_class=CausalModelScoringMetrics,
    description="Score antibody sequences with IgLM, conditioned on chain and species",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequences", "spans", "chains", "species"],
    iterable_output_field="scores",
    max_chunk_size=32,
    cacheable=True,
)
def run_iglm_score(
    inputs: IgLMScoringInput,
    config: IgLMScoringConfig,
    instance: Any = None,
) -> IgLMScoringOutput:
    """Score antibody sequences with IgLM.

    Computes the autoregressive likelihood of each sequence given its chain and
    species tokens. Supplying a span switches to IgLM's infilling likelihood: the
    span is scored given the rest of the chain, which is how designed CDR loops are
    ranked against each other.

    Args:
        inputs (IgLMScoringInput): Sequences to score, their per-sequence chain and
            species conditioning, and optional spans to score conditionally
            (1-indexed, inclusive, as everywhere in proto-tools).
        config (IgLMScoringConfig): Checkpoint, batching, and logits configuration.
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        IgLMScoringOutput: A ``CausalModelScoringMetrics`` per input sequence with:

            - ``log_likelihood``: Sum of per-token log-likelihoods over the scored region
            - ``avg_log_likelihood``: Mean per-token log-likelihood; matches the value
              upstream's ``IgLM.log_likelihood`` returns
            - ``perplexity``: exp(-``avg_log_likelihood``)
            - ``logits``: Per-position logits (scored_len, 33) when ``return_logits=True``,
              where ``scored_len`` is the number of residues scored plus one for the
              terminal token (so a whole chain gives ``len(sequence) + 1`` rows and a
              span gives ``span_length + 1``)
            - ``vocab``: IgLM's 33-token vocabulary, giving the column order of ``logits``

    Examples:
        >>> # Perplexity of a human heavy chain
        >>> inputs = IgLMScoringInput(sequences=[parent], chains=["heavy"], species=["human"])
        >>> result = run_iglm_score(inputs, IgLMScoringConfig())
        >>> print(result.scores[0]["perplexity"])

        >>> # Rank designed CDR H3 loops by their infilling likelihood
        >>> inputs = IgLMScoringInput(sequences=designs, spans=[IgLMSpan(start=99, end=106)])
        >>> result = run_iglm_score(inputs, IgLMScoringConfig())

        >>> # Score a heavy and a light chain in one call
        >>> inputs = IgLMScoringInput(sequences=[vh, vl], chains=["heavy", "light"])
        >>> result = run_iglm_score(inputs, IgLMScoringConfig())

    Note:
        - Lower perplexity means IgLM finds the sequence more antibody-like under the
          supplied conditioning; scores from different conditioning are not comparable.
        - Following upstream, the scored region ends with IgLM's terminal token, so a
          sequence contributes one more scored token than it has residues.
        - Span coordinates are 1-indexed and inclusive, matching ``iglm-infill``: the
          spans an infill run reports go straight back in without conversion.
    """
    logger.debug("Using local venv for IgLM scoring: %s", config.model_checkpoint)
    spans = inputs.spans or [None] * len(inputs.sequences)
    result = ToolInstance.dispatch(
        "iglm",
        {
            "operation": "score",
            "sequences": inputs.sequences,
            # Upstream masks with a 0-indexed, end-exclusive range.
            "infill_ranges": [None if span is None else [span.start - 1, span.end] for span in spans],
            "chain_tokens": inputs.chain_tokens(),
            "species_tokens": inputs.species_tokens(),
            "model_checkpoint": config.model_checkpoint,
            "batch_size": config.batch_size,
            "return_logits": config.return_logits,
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
        },
        instance=instance,
        config=config,
    )

    logits = result["logits"]
    scores = [
        CausalModelScoringMetrics(
            **metrics,
            logits=logits[i] if logits is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return IgLMScoringOutput(
        # Scores from different checkpoints sit on different scales, so record which produced these.
        metadata={"model_checkpoint": config.model_checkpoint},
        scores=scores,
    )
