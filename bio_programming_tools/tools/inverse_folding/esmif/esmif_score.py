"""ESM-IF/ProteinDPO scoring tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal

from tqdm import tqdm

from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingScoringOutput,
    SequenceScores,
    SequenceStructurePair,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig, ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ESMIFScoringInput(BaseToolInput):
    """Input for ESM-IF/ProteinDPO scoring.

    Attributes:
        sequence_structure_pairs: List of sequence-structure pairs to score.
            Each pair contains a sequence and a structure to score the sequence against.
    """

    sequence_structure_pairs: List[SequenceStructurePair] = InputField(
        description="List of sequence-structure pairs to score"
    )


ESMIFScoringOutput = InverseFoldingScoringOutput


class ESMIFScoringConfig(BaseConfig):
    """Configuration for ESM-IF/ProteinDPO scoring.

    Attributes:
        weights_variant: Which model weights to use. 'esmif' loads vanilla ESM-IF1,
            'protein_dpo' loads DPO-aligned weights optimized for protein stability.
        seed: Random seed.
        device: Device to run the model on.
    """

    weights_variant: Literal["esmif", "protein_dpo"] = ConfigField(
        title="Weights Variant",
        default="protein_dpo",
        description=(
            "Model weights to use: 'esmif' for vanilla ESM-IF1, "
            "'protein_dpo' for DPO-aligned weights optimized for stability"
        ),
        reload_on_change=True,
        examples=["esmif", "protein_dpo"],
    )

    seed: int = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed",
        hidden=True,
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    from bio_programming_tools.entities.structures import Structure

    _pdb_path = str(
        Path(__file__).parents[4]
        / "tests"
        / "dummy_data"
        / "test_structure_similarity.pdb"
    )
    return ESMIFScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence="A",
                structure=Structure(structure_filepath_or_content=_pdb_path),
            )
        ]
    )


@tool(
    key="esmif-score",
    label="ESM-IF Scoring",
    category="inverse_folding",
    input_class=ESMIFScoringInput,
    config_class=ESMIFScoringConfig,
    output_class=ESMIFScoringOutput,
    description=(
        "Score protein sequences against backbone structures using "
        "ESM-IF1 or ProteinDPO. Computes average log-likelihood and "
        "perplexity with multi-chain complex support."
    ),
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequence_structure_pairs",
    iterable_output_field="scores",
    cacheable=True,
)
def run_esmif_score(
    inputs: ESMIFScoringInput,
    config: ESMIFScoringConfig | None = None,
    instance=None,
) -> ESMIFScoringOutput:
    """Score protein sequences using ESM-IF/ProteinDPO.

    Scores each sequence against its paired structure using the full complex
    structural context (score_sequence_in_complex). Returns average
    log-likelihood and perplexity.

    Args:
        inputs: Sequence-structure pairs to score.
        config: Configuration including weights variant.

    Returns:
        ESMIFScoringOutput with scores for each input pair.
    """
    scores = []

    for pair in tqdm(
        inputs.sequence_structure_pairs,
        desc="ESM-IF scoring",
        unit="pair",
        total=len(inputs.sequence_structure_pairs),
    ):
        input_dict = {
            "operation": "score",
            "pdb_contents": pair.structure.structure_pdb,
            "chain_ids": pair.structure.get_chain_ids(),
            "sequence": pair.sequence,
            "device": config.device,
            "weights_variant": config.weights_variant,
            "verbose": config.verbose,
        }
        result = ToolInstance.dispatch(
            "esmif",
            input_dict,
            instance=instance,
            config=config,
        )
        scores.append(
            SequenceScores(
                metrics=result["metrics"],
            )
        )

    return ESMIFScoringOutput(scores=scores)
