"""proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py.

LigandMPNN sampling tool.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input:
LigandMPNNSampleInput = InverseFoldingInput
# Output:
LigandMPNNSampleOutput = InverseFoldingOutput
# Config:
LigandMPNNSampleConfig = InverseFoldingConfig


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
