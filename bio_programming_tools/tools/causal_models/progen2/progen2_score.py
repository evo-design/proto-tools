"""ProGen2 scoring tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator

from bio_programming_tools.tools.causal_models.shared_data_models import (
    CausalModelScoringOutput,
    SequenceScores,
)
from bio_programming_tools.utils.env_manager import EnvManager
from bio_programming_tools.utils.tool_io import BaseToolInput
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField, use_cloud_gpu

from .standalone.inference import PROGEN2_MODEL_CHECKPOINTS

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input: ProGen2ScoringInput
class ProGen2ScoringInput(BaseToolInput):
    """Input for ProGen2 protein sequence scoring.

    Attributes:
        sequences: Protein sequences to score. The start token '1' will be
            automatically prepended if not present.
    """

    sequences: List[str] = Field(description="Protein sequences to score")

    @field_validator("sequences", mode="before")
    @classmethod
    def validate_sequences(cls, v):
        """Coerce a single string to a list and validate non-empty."""
        if isinstance(v, str):
            v = [v]
        if not v:
            raise ValueError("sequences must not be empty")
        return v
# Output:
ProGen2ScoringOutput = CausalModelScoringOutput

# Config:
class ProGen2ScoringConfig(BaseConfig):
    """Configuration for ProGen2 protein sequence scoring.

    Computes autoregressive likelihood by computing P(x_t | x_{<t}) for each
    position and summing the log probabilities. Uses batched processing.

    Attributes:
        model_checkpoint (str): ProGen2 model checkpoint to use. Options include
            ``"progen2-small"`` (151M), ``"progen2-medium"`` (754M),
            ``"progen2-oas"`` (754M, antibody-specific), ``"progen2-large"`` (2B),
            ``"progen2-BFD90"`` (2B), ``"progen2-xlarge"`` (6B).
            Default: ``"progen2-large"``.

        local_path (Optional[str]): Optional path to local model weights directory.
            If provided, loads model from local filesystem instead of downloading
            from HuggingFace. Default: ``None``.

        device (str): Device to run the model on. Options include ``"cuda"``,
            ``"cpu"``, or specific GPU devices like ``"cuda:0"``.
            Default: ``"cuda"``.

        batch_size (Optional[int]): Number of sequences to process per batch.
            If None, processes all sequences at once. Lower values reduce memory
            usage but may be slower. Default: ``None``.

        verbose (bool): Whether to print status messages during scoring.
            Default: ``False``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        - ProGen2 uses autoregressive scoring: P(sequence) = prod_t P(x_t | x_{<t})
        - Metrics only count amino acid tokens, not special tokens (start, end, pad)
        - The vocab includes 30 tokens (special tokens + amino acids)
    """

    model_checkpoint: PROGEN2_MODEL_CHECKPOINTS = ConfigField(
        title="Model Checkpoint",
        default="progen2-large",
        description="ProGen2 model checkpoint to use",
    )
    local_path: Optional[str] = ConfigField(
        title="Local Model Path",
        default=None,
        description="Path to local model weights",
        hidden=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
    )
    batch_size: Optional[int] = ConfigField(
        title="Batch Size",
        default=None,
        ge=1,
        description="Max number of samples on the GPU at once",
        advanced=True,
    )
    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages",
        hidden=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Disable to save memory.",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="progen2-score",
    label="ProGen2 Scoring",
    input=ProGen2ScoringInput,
    config=ProGen2ScoringConfig,
    output=ProGen2ScoringOutput,
    description="Score protein sequences using ProGen2 language model",
)
def run_progen2_score(
    inputs: ProGen2ScoringInput, config: ProGen2ScoringConfig
) -> ProGen2ScoringOutput:
    """Score protein sequences using ProGen2 autoregressive language model.

    Computes the likelihood of protein sequences using ProGen2's autoregressive
    modeling. For each position t, computes log P(x_t | x_{<t}) and sums
    these to get the total log-likelihood.

    Args:
        inputs (ProGen2ScoringInput): Validated input containing protein sequences
            to score.
        config (ProGen2ScoringConfig): Scoring configuration specifying model,
            batch size, and whether to return logits.

    Returns:
        ProGen2ScoringOutput: Contains SequenceScores for each input sequence with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``
            - ``logits``: Per-position logits tensor (seq_len, vocab_size=30) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 30 tokens (special + amino acids) if
              ``return_logits=True``, otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only, no logits)
        >>> inputs = ProGen2ScoringInput(sequences=["MVLSPADKTN", "MKTLLILAVVAA"])
        >>> config = ProGen2ScoringConfig(model_checkpoint="progen2-large")
        >>> result = run_progen2_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
        >>>
        >>> # Scoring with logits for downstream analysis
        >>> config = ProGen2ScoringConfig(return_logits=True)
        >>> result = run_progen2_score(inputs, config)
        >>> print(f"Vocab: {result.scores[0].vocab}")
        >>>
        >>> # Using antibody-specific model
        >>> config = ProGen2ScoringConfig(model_checkpoint="progen2-oas")
        >>> result = run_progen2_score(inputs, config)

    Note:
        - Lower perplexity indicates higher model confidence in the sequence
        - The start token '1' is automatically prepended if not present
        - Set ``return_logits=False`` (default) to save memory when only metrics
          are needed
    """
    if use_cloud_gpu():
        logger.debug(f"Using the cloud runtime for ProGen2 scoring: {config.model_checkpoint}")
        import _gpu_runtime

        ProGen2Service = _gpu_runtime.Cls.from_name("bio-programming", "ProGen2Service")
        result = ProGen2Service().score.remote(
            model_checkpoint=config.model_checkpoint,
            sequences=inputs.sequences,
            verbose=config.verbose,
            batch_size=config.batch_size,
            return_logits=config.return_logits,
        )
    else:
        logger.debug(f"Using local venv for ProGen2 scoring: {config.model_checkpoint}")
        venv_manager = EnvManager("progen2")
        script_path = Path(__file__).parent / "standalone" / "inference.py"
        result = venv_manager.call_standalone_script_in_venv(
            script_path=script_path,
            input_dict={
                "operation": "score",
                "sequences": inputs.sequences,
                "model_checkpoint": config.model_checkpoint,
                "local_path": config.local_path,
                "device": config.device,
                "verbose": config.verbose,
                "batch_size": config.batch_size,
                "return_logits": config.return_logits,
            },
            device=config.device,
            verbose=config.verbose,
        )

    logits = result.get("logits")

    scores = [
        SequenceScores(
            metrics=metrics,
            logits=logits[i] if logits is not None else None,
            vocab=result["vocab"],
        )
        for i, metrics in enumerate(result["metrics"])
    ]

    return ProGen2ScoringOutput(scores=scores)
