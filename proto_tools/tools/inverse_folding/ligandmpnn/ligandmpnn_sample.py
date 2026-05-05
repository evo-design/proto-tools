"""proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py.

LigandMPNN sampling tool.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import AminoAcid, ConfigField, ToolInstance
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)

LigandMPNNModelType = Literal[
    "protein_mpnn",
    "ligand_mpnn",
    "soluble_mpnn",
    "per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn",
]


# ============================================================================
# Data Models
# ============================================================================
# Input:
LigandMPNNSampleInput = InverseFoldingInput
# Output:
LigandMPNNSampleOutput = InverseFoldingOutput


# Config:
class LigandMPNNSampleConfig(InverseFoldingConfig):
    """Configuration for LigandMPNN sampling.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits.
        excluded_amino_acids (list[AminoAcid] | None): One-letter codes of amino acids to exclude.
        seed (int): Random seed to use for sampling.
        model_type (LigandMPNNModelType): LigandMPNN variant to load.
        ligand_mpnn_use_atom_context (bool): Whether ligand-aware variants encode ligand atom context.
        ligand_mpnn_use_side_chain_context (bool): Whether to condition on fixed-residue sidechain atoms.
        ligand_mpnn_cutoff_for_score (float): Ligand-residue distance cutoff (A) for the ligand-interface
            recovery score.
    """

    model_type: LigandMPNNModelType = ConfigField(
        title="Model Type",
        default="ligand_mpnn",
        description="LigandMPNN variant: ligand-aware, protein-only, soluble, or membrane (per-residue/global)",
        reload_on_change=True,
        advanced=True,
    )
    ligand_mpnn_use_atom_context: bool = ConfigField(
        title="Use Ligand Atom Context",
        default=True,
        description="Encode ligand atom context in the message-passing graph",
        advanced=True,
    )
    ligand_mpnn_use_side_chain_context: bool = ConfigField(
        title="Use Sidechain Context",
        default=False,
        description="Condition on sidechain atoms of fixed residues",
        advanced=True,
    )
    ligand_mpnn_cutoff_for_score: float = ConfigField(
        title="Ligand Cutoff for Score",
        default=8.0,
        gt=0.0,
        description="Ligand-residue distance cutoff (A) for interface recovery score",
        advanced=True,
    )
    excluded_amino_acids: list[AminoAcid] | None = ConfigField(
        title="Excluded Amino Acids",
        default=None,
        description="Single-letter codes of amino acids to exclude (e.g. ['C'] to forbid cysteine)",
        examples=[["C"]],
    )


class LigandMPNNSequences(DesignedSequences):
    """Represents designed sequences from LigandMPNN.

    Attributes:
        sequences (list[str]): Designed amino acid sequences.
        sequence_recovery (list[float]): Per-sequence fraction of designed
            residues matching the input structure's reference sequence (0.0-1.0).
        ligand_interface_sequence_recovery (list[float]): Per-sequence recovery
            restricted to residues at the ligand interface (0.0-1.0).
    """

    sequence_recovery: list[float] = Field(
        description="Per-sequence fraction of designed residues matching the reference (0.0-1.0)",
    )
    ligand_interface_sequence_recovery: list[float] = Field(
        description="Per-sequence recovery restricted to ligand-interface residues (0.0-1.0)",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return LigandMPNNSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="ligandmpnn-sample",
    label="LigandMPNN Sampling",
    category="inverse_folding",
    input_class=LigandMPNNSampleInput,
    config_class=LigandMPNNSampleConfig,
    output_class=LigandMPNNSampleOutput,
    description="Sample protein sequences using LigandMPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="designed_sequences",
    cacheable=True,
)
def run_ligandmpnn_sample(
    inputs: LigandMPNNSampleInput,
    config: LigandMPNNSampleConfig,
    instance: Any = None,
) -> LigandMPNNSampleOutput:
    """Sample protein sequences using LigandMPNN.

    Args:
        inputs (LigandMPNNSampleInput): LigandMPNNSampleInput containing a list of structure inputs,
            and optional chain_ids/fixed_positions constraints.
        config (LigandMPNNSampleConfig): Configuration for sampling (temperature, batch_size, etc.).

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        LigandMPNNSampleOutput: LigandMPNNSampleOutput with designed sequences for each input structure.
    """
    designed_sequences = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()

    # Local venv execution
    for inp in progress_bar(
        inputs.inputs,
        desc="LigandMPNN sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_seqs: list[str] = []
        all_recovery: list[float] = []
        all_interface_recovery: list[float] = []
        remaining = config.num_sequences_per_structure
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
            input_dict = {
                "operation": "sample",
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "batch_size": chunk,
                "temperature": config.temperature,
                "fixed_positions": inp.fixed_positions,
                "excluded_amino_acids": config.excluded_amino_acids,
                "seed": base_seed + chunk_idx,
                "device": config.device,
                "verbose": config.verbose,
                "model_type": config.model_type,
                "ligand_mpnn_use_atom_context": config.ligand_mpnn_use_atom_context,
                "ligand_mpnn_use_side_chain_context": config.ligand_mpnn_use_side_chain_context,
                "ligand_mpnn_cutoff_for_score": config.ligand_mpnn_cutoff_for_score,
            }
            result = ToolInstance.dispatch(
                "ligandmpnn",
                input_dict,
                instance=instance,
                config=config,
            )
            all_seqs.extend(result["sequences"])
            all_recovery.extend(m["sequence_recovery"] for m in result["metrics"])
            all_interface_recovery.extend(m["ligand_interface_sequence_recovery"] for m in result["metrics"])
            chunk_idx += 1
            remaining -= chunk  # type: ignore[operator]
        designed_sequences.append(
            LigandMPNNSequences(
                sequences=all_seqs,
                sequence_recovery=all_recovery,
                ligand_interface_sequence_recovery=all_interface_recovery,
            )
        )

    return LigandMPNNSampleOutput(designed_sequences=designed_sequences)  # type: ignore[arg-type]
