"""LigandMPNN scoring tool."""

from pathlib import Path
from typing import Any, Literal

from proto_tools.tools.inverse_folding.ligandmpnn.ligandmpnn_sample import (
    LigandMPNNImplementation,
    local_checkpoint_on_remote,
)
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
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.progress import progress_bar

LigandMPNNScoringMode = Literal["single_aa", "autoregressive"]


class LigandMPNNScoringInput(BaseToolInput):
    """Input for LigandMPNN scoring.

    Attributes:
        sequence_structure_pairs (list[SequenceStructurePair]): Sequence and structure pairs to
            score; each pair may carry per-pair ``excluded_positions`` excluded from the metrics.
    """

    sequence_structure_pairs: list[SequenceStructurePair] = InputField(
        title="Sequence-Structure Pairs",
        description="List of sequence-structure pairs to score",
    )


LigandMPNNScoringOutput = InverseFoldingScoringOutput


class LigandMPNNScoringConfig(BaseConfig):
    """Configuration for LigandMPNN structure-conditioned scoring.

    Attributes:
        device (str): Device to run the model on.
        return_logits (bool): Whether to include per-position logits.
        scoring_mode (LigandMPNNScoringMode): Single-position or autoregressive scoring mode.
        implementation (LigandMPNNImplementation): Which LigandMPNN code runs: ``"foundry"``
            (default) or ``"original"``, the upstream LigandMPNN code (fetched on first use), which
            matches the reference repository's scores. The two prepare the structure differently,
            so scores differ between them even with the same weights.
        model_checkpoint (str): Sequence-model weights, which work with either implementation:
            ``"ligandmpnn_v_32_010_25"`` (default) or ``"ligandmpnn_v_32_020_25"``, the published
            weights trained with 0.10 Å and 0.20 Å backbone noise, or a path to your own ``.pt``
            checkpoint, which runs only locally.
        use_atom_context (bool): Whether ligand-aware variants encode ligand atom context.
        use_side_chain_context (bool): Whether to condition on fixed-residue sidechain atoms.
        cutoff_for_score (float): Ligand-residue distance cutoff (Å).
    """

    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on.",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )
    return_logits: bool = ConfigField(
        title="Return Logits",
        default=False,
        description="Whether to include per-position logits in the output.",
    )
    scoring_mode: LigandMPNNScoringMode = ConfigField(
        title="Scoring Mode",
        default="single_aa",
        description="Use single-position probabilities or one seed-determined autoregressive order.",
    )
    implementation: LigandMPNNImplementation = ConfigField(
        title="Implementation",
        default="foundry",
        description="'foundry' (default) or 'original', the upstream LigandMPNN code.",
        reload_on_change=True,
    )
    model_checkpoint: str = ConfigField(
        title="Model / Checkpoint",
        default="ligandmpnn_v_32_010_25",
        min_length=1,
        description="'ligandmpnn_v_32_010_25', 'ligandmpnn_v_32_020_25', or a path to a custom .pt checkpoint.",
        reload_on_change=True,
    )
    use_atom_context: bool = ConfigField(
        title="Use Ligand Atom Context",
        default=True,
        description="Encode ligand atom context in the message-passing graph.",
    )
    use_side_chain_context: bool = ConfigField(
        title="Use Sidechain Context",
        default=False,
        description="Condition on sidechain atoms of fixed residues.",
    )
    cutoff_for_score: float = ConfigField(
        title="Ligand Cutoff for Score",
        default=8.0,
        gt=0.0,
        description="Ligand-residue distance cutoff (Å).",
    )

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A local checkpoint path isn't present on a remote worker."""
        return local_checkpoint_on_remote(self.model_checkpoint, device)


def example_input() -> Any:
    """Minimal valid scoring input."""
    from proto_tools.entities.structures import Structure

    structure = Structure.from_file(str(Path(__file__).parents[1] / "example_input_fixture.pdb"))
    sequence = "".join(structure.get_chain_sequence(chain) for chain in structure.get_chain_ids())
    return LigandMPNNScoringInput(
        sequence_structure_pairs=[SequenceStructurePair(sequence=sequence, structure=structure)]
    )


@tool(
    key="ligandmpnn-score",
    label="LigandMPNN Scoring",
    category="inverse_folding",
    input_class=LigandMPNNScoringInput,
    config_class=LigandMPNNScoringConfig,
    output_class=LigandMPNNScoringOutput,
    metrics_class=InverseFoldingScoringMetrics,
    description="Score protein sequences using LigandMPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["sequence_structure_pairs"],
    iterable_output_field="scores",
    max_chunk_size=64,
    cacheable=True,
)
def run_ligandmpnn_score(
    inputs: LigandMPNNScoringInput,
    config: LigandMPNNScoringConfig,
    instance: Any = None,
) -> LigandMPNNScoringOutput:
    """Score sequences against structures with LigandMPNN."""
    scores = []
    seed = config.seed if config.seed is not None else config.get_random_int()

    for pair in progress_bar(
        inputs.sequence_structure_pairs,
        desc="LigandMPNN scoring",
        unit="pair",
        disable=not config.verbose,
    ):
        with pair.structure.temp_file() as pdb_path:
            result = ToolInstance.dispatch(
                "ligandmpnn",
                {
                    "operation": "score",
                    "pdb_path": str(pdb_path),
                    "chain_ids": pair.structure.get_chain_ids(),
                    "sequence": pair.sequence,
                    "seed": seed,
                    "fixed_positions": (
                        pair.excluded_positions.to_residue_numbers(pair.structure)
                        if pair.excluded_positions is not None
                        else None
                    ),
                    "device": config.device,
                    "return_logits": config.return_logits,
                    "verbose": config.verbose,
                    "implementation": config.implementation,
                    "model_checkpoint": config.model_checkpoint,
                    "scoring_mode": config.scoring_mode,
                    "ligand_mpnn_use_atom_context": config.use_atom_context,
                    "ligand_mpnn_use_side_chain_context": config.use_side_chain_context,
                    "ligand_mpnn_cutoff_for_score": config.cutoff_for_score,
                },
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

    return LigandMPNNScoringOutput(scores=scores)
