"""proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_score.py.

ProteinMPNN scoring tool.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingScoringMetrics,
    InverseFoldingScoringOutput,
    SequenceStructurePair,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class ProteinMPNNScoringInput(BaseToolInput):
    """Input for ProteinMPNN scoring.

    Attributes:
        sequence_structure_pairs (list[SequenceStructurePair]): List of sequence-structure pairs to score.
            Each pair contains a sequence and a structure to score the sequence against.
    """

    sequence_structure_pairs: list[SequenceStructurePair] = InputField(
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
        fixed_positions (dict[str, list[int]] | None): Dictionary mapping chain
            IDs to fixed positions in the sequence. If None, no positions will be fixed.
            In scoring, fixed positions are excluded from average and total likelihood metrics.
            NOTE: Positions should match positions in the structure (generally 1-indexed).

        device (str): Device to run the model on. Options include ``"cuda"`` (NVIDIA GPU),
            ``"cpu"`` (CPU execution). Default: ``"cuda"``.

        return_logits (bool): Whether to include per-position logits in the output.
            When ``True``, returns logits for each sequence. When ``False``, only
            returns metrics (saves memory and serialization time). Default: ``False``.

        model_choice (Literal['proteinmpnn', 'v_48_002', 'v_48_010', 'v_48_030', 'abmpnn', 'soluble']): Model
            weights. ``"proteinmpnn"`` is ColabDesign's default ``v_48_020`` (medium training noise). The
            ``v_48_*`` variants are the same architecture trained at different noise levels (002 / 010 / 030).
            ``"abmpnn"`` is antibody-optimized; ``"soluble"`` is soluble-protein-trained.

    Note:
        - ProteinMPNN uses AlphaFold alphabet ordering (21 tokens including X)
        - Vocab order: ARNDCQEGHILKMFPSTWYVX
        - Logits are structure-conditioned: P(aa_i | structure, context)
    """

    fixed_positions: dict[str, list[int]] | None = ConfigField(
        title="Fixed Positions",
        default=None,
        description="Dictionary mapping chain IDs to fixed positions excluded from scoring metrics",
        examples=[{"A": [1, 2, 3]}, {"A": [10, 20], "B": [5]}],
    )

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on. Options include 'cuda' (NVIDIA GPU), 'cpu' (CPU execution)",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )

    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output. Disable to save memory.",
    )

    model_choice: Literal["proteinmpnn", "v_48_002", "v_48_010", "v_48_030", "abmpnn", "soluble"] = ConfigField(
        title="Model Choice",
        default="proteinmpnn",
        description="Weights: proteinmpnn (=v_48_020), v_48_{002,010,030} noise variants, abmpnn, soluble",
        reload_on_change=True,
        examples=["proteinmpnn", "v_48_010", "abmpnn", "soluble"],
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    from proto_tools.entities.structures import Structure

    structure = Structure.from_file(str(Path(__file__).parents[1] / "example_input_fixture.pdb"))
    sequence = "".join(structure.get_chain_sequence(chain) for chain in structure.get_chain_ids())
    return ProteinMPNNScoringInput(
        sequence_structure_pairs=[SequenceStructurePair(sequence=sequence, structure=structure)]
    )


@tool(
    key="proteinmpnn-score",
    label="ProteinMPNN Scoring",
    category="inverse_folding",
    input_class=ProteinMPNNScoringInput,
    config_class=ProteinMPNNScoringConfig,
    output_class=ProteinMPNNScoringOutput,
    metrics_class=InverseFoldingScoringMetrics,
    description="Score protein sequences using ProteinMPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="sequence_structure_pairs",
    iterable_output_field="scores",
    cacheable=True,
)
def run_proteinmpnn_score(
    inputs: ProteinMPNNScoringInput,
    config: ProteinMPNNScoringConfig,
    instance: Any = None,
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

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProteinMPNNScoringOutput: Contains an ``InverseFoldingScoringMetrics`` for
            each input pair with:

            - ``log_likelihood``, ``avg_log_likelihood``, ``perplexity`` (access via
              attribute ``score.perplexity`` or mapping ``score["perplexity"]``)
            - ``logits``: Per-position logits tensor (seq_len, 21) if
              ``return_logits=True``, otherwise ``None``
            - ``vocab``: List of 21 amino acid characters (AlphaFold ordering) if
              ``return_logits=True``, otherwise ``None``

    Examples:
        >>> # Basic scoring (metrics only)
        >>> from proto_tools.tools.inverse_folding.shared_data_models import SequenceStructurePair
        >>> pair = SequenceStructurePair(sequence="MVLS...", structure=structure_input)
        >>> inputs = ProteinMPNNScoringInput(sequence_structure_pairs=[pair])
        >>> config = ProteinMPNNScoringConfig()
        >>> result = run_proteinmpnn_score(inputs, config)
        >>> print(f"Perplexity: {result.scores[0]['perplexity']}")
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

    seed = config.seed if config.seed is not None else config.get_random_int()

    for sequence_structure_pair in progress_bar(
        inputs.sequence_structure_pairs,
        desc="ProteinMPNN scoring",
        unit="pair",
        disable=not config.verbose,
    ):
        with sequence_structure_pair.structure.temp_file() as pdb_path:
            input_dict = {
                "operation": "score",
                "pdb_path": str(pdb_path),
                "chain_ids": sequence_structure_pair.structure.get_chain_ids(),
                "sequence": sequence_structure_pair.sequence,
                "seed": seed,
                "fixed_positions": config.fixed_positions,
                "device": config.device,
                "model_choice": config.model_choice,
                "return_logits": config.return_logits,
                "verbose": config.verbose,
            }
            result = ToolInstance.dispatch(
                "proteinmpnn",
                input_dict,
                instance=instance,
                config=config,
            )
        scores.append(
            InverseFoldingScoringMetrics(
                **result["metrics"],
                logits=result["logits"],
                vocab=result["vocab"],
            )
        )

    return ProteinMPNNScoringOutput(scores=scores)
