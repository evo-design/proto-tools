"""proto_tools/tools/inverse_folding/ligandmpnn/ligandmpnn_sample.py.

LigandMPNN sampling tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import Field
from tqdm import tqdm

from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ToolInstance

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
    config: LigandMPNNSampleConfig | None = None,
    instance: Any = None,
) -> LigandMPNNSampleOutput:
    """Sample protein sequences using LigandMPNN.

    Args:
        inputs (LigandMPNNSampleInput): LigandMPNNSampleInput containing a list of structure inputs,
            and optional chain_ids/fixed_positions constraints.
        config (LigandMPNNSampleConfig | None): Configuration for sampling (temperature, batch_size, etc.).

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        LigandMPNNSampleOutput: LigandMPNNSampleOutput with designed sequences for each input structure.
    """
    designed_sequences = []

    # Local venv execution
    for inp in tqdm(
        inputs.inputs,
        desc="LigandMPNN sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_seqs, all_metrics = [], []
        remaining = config.num_sequences_per_structure  # type: ignore[union-attr]
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)  # type: ignore[type-var, union-attr]
            input_dict = {
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "batch_size": chunk,
                "temperature": config.temperature,  # type: ignore[union-attr]
                "fixed_positions": inp.fixed_positions,
                "excluded_amino_acids": config.excluded_amino_acids,  # type: ignore[union-attr]
                "seed": config.seed + chunk_idx,  # type: ignore[union-attr]
                "device": config.device,  # type: ignore[union-attr]
                "verbose": config.verbose,  # type: ignore[union-attr]
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
