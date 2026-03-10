"""LigandMPNN sampling tool."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import Field
from tqdm import tqdm

from pathlib import Path

from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_instance import ToolInstance

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

    `ligandmpnn_metrics` contains per-sequence metrics returned by LigandMPNN.
    """

    ligandmpnn_metrics: List[Dict[str, Any]] = Field(
        description="Metrics returned by LigandMPNN",
        title="LigandMPNN Metrics",
        advanced=True,
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return LigandMPNNSampleInput(
        inputs=[InverseFoldingStructureInput(
            structure=str(Path(__file__).parents[4] / "tests" / "dummy_data" / "test_structure_similarity.pdb"),
        )]
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
    instance=None,
) -> LigandMPNNSampleOutput:
    """Sample protein sequences using LigandMPNN.

    Args:
        inputs: LigandMPNNSampleInput containing a list of structure inputs,
            and optional chain_ids/fixed_positions constraints.
        config: Configuration for sampling (temperature, batch_size, etc.).

    Returns:
        LigandMPNNSampleOutput with designed sequences for each input structure.
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
        remaining = config.num_sequences_per_structure
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)
            input_dict = {
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "batch_size": chunk,
                "temperature": config.temperature,
                "fixed_positions": inp.fixed_positions,
                "excluded_amino_acids": config.excluded_amino_acids,
                "seed": config.seed + chunk_idx,
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
            remaining -= chunk
        designed_sequences.append(
            LigandMPNNSequences(
                sequences=all_seqs,
                ligandmpnn_metrics=all_metrics,
            )
        )

    return LigandMPNNSampleOutput(designed_sequences=designed_sequences)
