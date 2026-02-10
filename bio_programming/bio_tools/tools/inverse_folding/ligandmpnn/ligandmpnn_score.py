"""LigandMPNN scoring tool."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from bio_programming.bio_tools.tools.utils import BaseConfig, ConfigField
from bio_programming.bio_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingScoringOutput,
    SequenceStructurePair,
)
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput

# ============================================================================
# Data Models
# ============================================================================
# Input:
class LigandMPNNScoringInput(BaseToolInput):
    """Input for LigandMPNN scoring.

    Attributes:
        sequence_structure_pairs: List of sequence-structure pairs to score.
            Each pair contains a sequence and a structure to score the sequence against.
    """

    sequence_structure_pairs: List[SequenceStructurePair] = Field(
        description="List of sequence-structure pairs to score"
    )


# Output:
LigandMPNNScoringOutput = InverseFoldingScoringOutput

# Config:
class LigandMPNNScoringConfig(BaseConfig):
    """Configuration for LigandMPNN scoring.

    Attributes:
        fixed_positions: Dictionary mapping chain IDs to fixed positions in the sequence.
            If None, no positions will be fixed. In scoring, fixed positions will not
            be utilized in perplexity calculation.

        seed: Random seed to use for scoring.

        device: Device to run the model on.

        verbose: Whether to print status messages during execution.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.
    """

    fixed_positions: Optional[Dict[str, List[int]]] = ConfigField(
        title="Fixed Positions",
        default=None,
        description="Dictionary mapping chain IDs to fixed positions in the sequence",
    )

    seed: int = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed to use for scoring",
        hidden=True,
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
    )

    verbose: bool = ConfigField(
        title="Verbose",
        default=False,
        description="Whether to print status messages during execution",
        hidden=True,
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Saves memory if disabled.",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def run_ligandmpnn_score(
    inputs: LigandMPNNScoringInput,
    config: LigandMPNNScoringConfig,
) -> LigandMPNNScoringOutput:
    """Score protein sequences using LigandMPNN.

    Computes the perplexity and logits for protein sequences given their
    corresponding structures via a forward pass through the model. This is
    useful for evaluating how well a sequence fits a given structure in the
    presence of ligands.

    Args:
        inputs: LigandMPNNScoringInput containing sequence-structure pairs to score.
        config: Configuration for scoring (fixed_positions, device, etc.).

    Returns:
        LigandMPNNScoringOutput with scores for each input sequence.

    Examples:
        >>> from bio_programming.bio_tools.entities.structures import Structure
        >>> structure = Structure.from_pdb_file("protein.pdb")
        >>> inputs = LigandMPNNScoringInput(
        ...     sequence_structure_pairs=[
        ...         SequenceStructurePair(sequence="MVLSPADKTN", structure=structure)
        ...     ]
        ... )
        >>> config = LigandMPNNScoringConfig()
        >>> result = run_ligandmpnn_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0].perplexity}")
    """
    _ = (inputs, config)
    # TODO: Implement LigandMPNN scoring
    pass
