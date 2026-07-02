"""LigandMPNN scoring tool."""

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

LigandMPNNScoringMode = Literal["single_aa", "autoregressive"]
LigandMPNNBackend = Literal["foundry", "reference"]


class LigandMPNNScoringInput(BaseToolInput):
    """Input for LigandMPNN scoring.

    Attributes:
        sequence_structure_pairs (list[SequenceStructurePair]): Sequence and structure pairs to
            score; each pair may carry per-pair ``fixed_positions`` excluded from the metrics.
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
        backend (LigandMPNNBackend): Inference backend. ``"foundry"`` uses the
            managed Foundry implementation; ``"reference"`` uses a local
            reference LigandMPNN checkout for compatibility experiments.
        checkpoint_path (str | None): Optional explicit LigandMPNN checkpoint path.
        reference_backend_path (str | None): Path to a checkout containing the
            reference ``ligandmpnn`` Python package when ``backend="reference"``.
        ligand_mpnn_use_atom_context (bool): Whether ligand-aware variants encode ligand atom context.
        ligand_mpnn_use_side_chain_context (bool): Whether to condition on fixed-residue sidechain atoms.
        ligand_mpnn_cutoff_for_score (float): Ligand-residue distance cutoff (Å).
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
    backend: LigandMPNNBackend = ConfigField(
        title="Backend",
        default="foundry",
        description="Inference backend: managed Foundry implementation or local reference implementation.",
        reload_on_change=True,
    )
    checkpoint_path: str | None = ConfigField(
        title="Checkpoint Path",
        default=None,
        description="Optional explicit LigandMPNN checkpoint path.",
        reload_on_change=True,
    )
    reference_backend_path: str | None = ConfigField(
        title="Reference Backend Path",
        default=None,
        description="Path to a local reference LigandMPNN checkout when backend='reference'.",
        reload_on_change=True,
    )
    ligand_mpnn_use_atom_context: bool = ConfigField(
        title="Use Ligand Atom Context",
        default=True,
        description="Encode ligand atom context in the message-passing graph.",
    )
    ligand_mpnn_use_side_chain_context: bool = ConfigField(
        title="Use Sidechain Context",
        default=False,
        description="Condition on sidechain atoms of fixed residues.",
    )
    ligand_mpnn_cutoff_for_score: float = ConfigField(
        title="Ligand Cutoff for Score",
        default=8.0,
        gt=0.0,
        description="Ligand-residue distance cutoff (Å).",
    )


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
                    "fixed_positions": (pair.fixed_positions.chains if pair.fixed_positions is not None else None),
                    "device": config.device,
                    "return_logits": config.return_logits,
                    "verbose": config.verbose,
                    "model_type": "ligand_mpnn",
                    "backend": config.backend,
                    "checkpoint_path": config.checkpoint_path,
                    "reference_backend_path": config.reference_backend_path,
                    "scoring_mode": config.scoring_mode,
                    "ligand_mpnn_use_atom_context": config.ligand_mpnn_use_atom_context,
                    "ligand_mpnn_use_side_chain_context": config.ligand_mpnn_use_side_chain_context,
                    "ligand_mpnn_cutoff_for_score": config.ligand_mpnn_cutoff_for_score,
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
