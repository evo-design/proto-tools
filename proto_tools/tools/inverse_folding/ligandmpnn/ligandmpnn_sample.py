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
        ligandmpnn_metrics (list[dict[str, Any]]): Per-sequence metrics returned by
            LigandMPNN, such as sequence recovery and log-likelihood scores.
    """

    ligandmpnn_metrics: list[dict[str, Any]] = Field(
        description="Metrics returned by LigandMPNN",
        title="LigandMPNN Metrics",
        json_schema_extra={"advanced": True},
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return LigandMPNNSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(Path(__file__).parents[1] / "examples" / "example.pdb"),  # type: ignore[arg-type]
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

    # Local venv execution
    for inp in progress_bar(
        inputs.inputs,
        desc="LigandMPNN sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_seqs, all_metrics = [], []
        remaining = config.num_sequences_per_structure
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
            input_dict = {
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "batch_size": chunk,
                "temperature": config.temperature,
                "fixed_positions": inp.fixed_positions,
                "excluded_amino_acids": config.excluded_amino_acids,
                "seed": config.resolved_seed + chunk_idx,
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
            all_metrics.extend(result["metrics"])
            chunk_idx += 1
            remaining -= chunk  # type: ignore[operator]
        designed_sequences.append(
            LigandMPNNSequences(
                sequences=all_seqs,
                ligandmpnn_metrics=all_metrics,
            )
        )

    return LigandMPNNSampleOutput(designed_sequences=designed_sequences)  # type: ignore[arg-type]
