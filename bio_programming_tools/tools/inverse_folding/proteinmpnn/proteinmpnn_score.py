"""ProteinMPNN scoring tool."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import Field
from tqdm import tqdm

from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingScoringOutput,
    SequenceScores,
    SequenceStructurePair,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_cache import tool_cache_iterable
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolInput

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input:
class ProteinMPNNScoringInput(BaseToolInput):
    """Input for ProteinMPNN scoring.

    Attributes:
        sequence_structure_pairs (List[SequenceStructurePair]): List of sequence-structure pairs to score.
            Each pair contains a sequence and a structure to score the sequence against.
    """

    sequence_structure_pairs: List[SequenceStructurePair] = Field(
        description="List of sequence-structure pairs to score"
    )

# Output:
ProteinMPNNScoringOutput = InverseFoldingScoringOutput

# Config:
class ProteinMPNNScoringConfig(BaseConfig):
    """Configuration for ProteinMPNN structure-conditioned scoring.

    Scores protein sequences based on how well they fit a given 3D structure
    using ProteinMPNN's structure-conditioned language model.

    Attributes:
        fixed_positions (Optional[Dict[str, List[int]]]): Dictionary mapping chain
            IDs to fixed positions in the sequence. If None, no positions will be fixed.
            In scoring, fixed positions will not be utilized in perplexity calculation.
            NOTE: Positions should match positions in the structure (generally 1-indexed).

        seed (int): Random seed to use for scoring. Default: 42.

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution). Default: ``"cuda"``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

    Note:
        - ProteinMPNN uses AlphaFold alphabet ordering (21 tokens including X)
        - Vocab order: ARNDCQEGHILKMFPSTWYVX
        - Logits are structure-conditioned: P(aa_i | structure, context)
    """

    fixed_positions: Optional[Dict[str, List[int]]] = ConfigField(
        title="Fixed Positions",
        default=None,
        description="Dictionary mapping chain IDs to fixed positions in the sequence. If None, no positions will be fixed",
        examples={"A": [1, 2, 3, 4, 5, 20, 21, 22], "B": [10, 11, 12, 13, 14, 15, 20, 21, 22]},
    )

    seed: int = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed to use for scoring",
        examples=[42, 123, 456],
        hidden=True,
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on. Options include 'cuda' (NVIDIA GPU), 'cpu' (CPU execution)",
        hidden=True,
        examples=["cuda", "cpu"],
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
    key="proteinmpnn-score",
    label="ProteinMPNN Scoring",
    category="inverse_folding",
    input=ProteinMPNNScoringInput,
    config=ProteinMPNNScoringConfig,
    output=ProteinMPNNScoringOutput,
    description="Score protein sequences using ProteinMPNN",
    uses_gpu=True,
)
@tool_cache_iterable(
    input_iterable_field="sequence_structure_pairs",
    output_iterable_field="scores",
    tool_name="proteinmpnn-score",
)
def run_proteinmpnn_score(
    inputs: ProteinMPNNScoringInput,
    config: ProteinMPNNScoringConfig,
    instance=None,
) -> ProteinMPNNScoringOutput:
    """Score protein sequences using ProteinMPNN structure-conditioned model.

    Computes the likelihood of protein sequences given a 3D structure using
    ProteinMPNN's structure-conditioned language model. This evaluates how well
    a sequence "fits" a given protein structure.

    Args:
        inputs (ProteinMPNNScoringInput): Validated input containing sequence-structure
            pairs to score.
        config (ProteinMPNNScoringConfig): Scoring configuration specifying fixed
            positions, device settings, and whether to return logits.

    Returns:
        ProteinMPNNScoringOutput: Contains SequenceScores for each input pair with:

            - ``metrics``: Dict with ``log_likelihood``, ``avg_log_likelihood``,
              ``perplexity``
            - ``logits``: Per-position logits tensor (seq_len, 21) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 21 amino acid characters (AlphaFold ordering) if
              ``return_logits=True``, otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only)
        >>> from bio_programming_tools.tools.inverse_folding.shared_data_models import SequenceStructurePair
        >>> pair = SequenceStructurePair(sequence="MVLS...", structure=structure_input)
        >>> inputs = ProteinMPNNScoringInput(sequence_structure_pairs=[pair])
        >>> config = ProteinMPNNScoringConfig()
        >>> result = run_proteinmpnn_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
        >>>
        >>> # Scoring with logits
        >>> config = ProteinMPNNScoringConfig(return_logits=True)
        >>> result = run_proteinmpnn_score(inputs, config)
        >>> print(f"Vocab: {result.scores[0].vocab}")

    Note:
        - Lower perplexity indicates better sequence-structure compatibility
        - ProteinMPNN uses structure coordinates to condition predictions
        - Set ``return_logits=False`` (default) to save memory
    """
    scores = []

    # Local venv execution
    logger.debug("Using local venv for ProteinMPNN scoring")

    for sequence_structure_pair in tqdm(
        inputs.sequence_structure_pairs,
        desc="ProteinMPNN scoring",
        unit="pair",
        disable=not config.verbose,
    ):
        input_dict = {
            "operation": "score",
            "pdb_contents": sequence_structure_pair.structure.structure_pdb,
            "chain_ids": sequence_structure_pair.structure.get_chain_ids(),
            "sequence": sequence_structure_pair.sequence,
            "seed": config.seed,
            "fixed_positions": config.fixed_positions,
            "device": config.device,
            "return_logits": config.return_logits,
        }
        result = ToolInstance.dispatch(
            "proteinmpnn",
            input_dict,
            instance=instance,
            verbose=config.verbose,
        )
        scores.append(
            SequenceScores(
                metrics=result["metrics"],
                logits=result["logits"],
                vocab=result["vocab"],
            )
        )

    return ProteinMPNNScoringOutput(scores=scores)
