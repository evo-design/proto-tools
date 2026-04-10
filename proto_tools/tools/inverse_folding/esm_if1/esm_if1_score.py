"""proto_tools/tools/inverse_folding/esm_if1/esm_if1_score.py.

ESM-IF1/ProteinDPO scoring tool.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingScoringOutput,
    SequenceScores,
    SequenceStructurePair,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.utils.tool_io import BaseToolInput, InputField

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ESMIF1ScoringInput(BaseToolInput):
    """Input for ESM-IF1/ProteinDPO scoring.

    Attributes:
        sequence_structure_pairs (list[SequenceStructurePair]): List of sequence-structure pairs to score.
            Each pair contains a sequence and a structure to score the sequence against.
    """

    sequence_structure_pairs: list[SequenceStructurePair] = InputField(
        description="List of sequence-structure pairs to score"
    )


ESMIF1ScoringOutput = InverseFoldingScoringOutput


class ESMIF1ScoringConfig(BaseConfig):
    """Configuration for ESM-IF1/ProteinDPO scoring.

    Attributes:
        weights_variant (Literal['esmif', 'protein_dpo']): Which model weights to use. 'esmif' loads vanilla ESM-IF1,
            'protein_dpo' loads DPO-aligned weights optimized for protein stability.
        device (str): Device to run the model on.
    """

    weights_variant: Literal["esmif", "protein_dpo"] = ConfigField(
        title="Weights Variant",
        default="protein_dpo",
        description="'esmif' for vanilla ESM-IF1, 'protein_dpo' for DPO-aligned weights",
        reload_on_change=True,
        examples=["esmif", "protein_dpo"],
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
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    from proto_tools.entities.structures import Structure

    _pdb_path = str(Path(__file__).parents[4] / "tests" / "dummy_data" / "test_structure_similarity.pdb")
    return ESMIF1ScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence="A",
                structure=Structure.from_file(_pdb_path),
            )
        ]
    )


@tool(
    key="esm-if1-score",
    label="ESM-IF1 Scoring",
    category="inverse_folding",
    input_class=ESMIF1ScoringInput,
    config_class=ESMIF1ScoringConfig,
    output_class=ESMIF1ScoringOutput,
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
def run_esm_if1_score(
    inputs: ESMIF1ScoringInput,
    config: ESMIF1ScoringConfig,
    instance: Any = None,
) -> ESMIF1ScoringOutput:
    """Score protein sequences using ESM-IF1/ProteinDPO.

    Scores each sequence against its paired structure using the full complex
    structural context (score_sequence_in_complex). Returns average
    log-likelihood and perplexity.

    Args:
        inputs (ESMIF1ScoringInput): Sequence-structure pairs to score.
        config (ESMIF1ScoringConfig): Configuration including weights variant.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMIF1ScoringOutput: ESMIF1ScoringOutput with scores for each input pair.
    """
    scores = []

    for pair in progress_bar(
        inputs.sequence_structure_pairs,
        desc="ESM-IF1 scoring",
        unit="pair",
        total=len(inputs.sequence_structure_pairs),
    ):
        input_dict = {
            "operation": "score",
            "pdb_contents": pair.structure.structure_pdb,
            "chain_ids": pair.structure.get_chain_ids(),
            "sequence": pair.sequence,
            "seed": config.resolved_seed,
            "device": config.device,
            "weights_variant": config.weights_variant,
            "verbose": config.verbose,
        }
        result = ToolInstance.dispatch(
            "esm_if1",
            input_dict,
            instance=instance,
            config=config,
        )
        scores.append(
            SequenceScores(
                metrics=result["metrics"],
            )
        )

    return ESMIF1ScoringOutput(scores=scores)
