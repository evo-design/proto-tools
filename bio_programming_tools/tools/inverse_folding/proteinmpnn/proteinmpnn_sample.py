"""ProteinMPNN sampling tool."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from pydantic import Field
from tqdm import tqdm

from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input:
ProteinMPNNSampleInput = InverseFoldingInput
# Output:
ProteinMPNNSampleOutput = InverseFoldingOutput
# Config:
ProteinMPNNSampleConfig = InverseFoldingConfig

class ProteinMPNNSequences(DesignedSequences):
    """Represents a designed sequence from the ProteinMPNN model.

    Attributes:
        sequences (List[str]): Designed protein sequences.
        perplexity (List[float]): Per-sequence perplexity values.
        sequence_identity (List[float]): Sequence identity to the PDB prompt sequence.
    """

    perplexity: List[float] = Field(
        description="Perplexity of the sequence from the ProteinMPNN model"
    )
    sequence_identity: List[float] = Field(
        description="Sequence identity to the sequence in the PDB prompt"
    )

# ============================================================================
# Tool Implementation
# ============================================================================
@tool(
    key="proteinmpnn-sample",
    label="ProteinMPNN Sampling",
    category="inverse_folding",
    input=ProteinMPNNSampleInput,
    config=ProteinMPNNSampleConfig,
    output=ProteinMPNNSampleOutput,
    description="Sample protein sequences using ProteinMPNN",
    uses_gpu=True,
)
def run_proteinmpnn_sample(
    inputs: ProteinMPNNSampleInput,
    config: ProteinMPNNSampleConfig,
    instance=None,
) -> ProteinMPNNSampleOutput:
    """Sample protein sequences using ProteinMPNN.

    Args:
        inputs: ProteinMPNNSampleInput containing a list of structure inputs,
            each with optional chain_ids/fixed_positions constraints.
        config: Configuration for sampling (temperature, batch_size, etc.).

    Returns:
        ProteinMPNNSampleOutput with designed sequences for each input structure.

    Note:
        Multi-chain sampling returns a "/"-delimited sequence preserving chain ID order.
    """
    designed_sequences = []

    # Local venv execution
    logger.debug("Using local venv for ProteinMPNN sampling")

    for inp in tqdm(
        inputs.inputs,
        desc="ProteinMPNN sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        all_seqs, all_perp, all_seqid = [], [], []
        remaining = config.num_sequences_per_structure
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)
            input_dict = {
                "operation": "sample",
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "batch_size": chunk,
                "temperature": config.temperature,
                "fixed_positions": inp.fixed_positions,
                "excluded_amino_acids": config.excluded_amino_acids,
                "seed": config.seed + chunk_idx,
                "device": config.device,
            }
            result = ToolInstance.dispatch(
                "proteinmpnn",
                input_dict,
                instance=instance,
                verbose=config.verbose,
            )
            all_seqs.extend(result["seq"])
            all_perp.extend(np.exp(result["score"]).tolist())
            all_seqid.extend(result["seqid"])
            chunk_idx += 1
            remaining -= chunk
        designed_sequences.append(
            ProteinMPNNSequences(
                sequences=all_seqs,
                perplexity=all_perp,
                sequence_identity=all_seqid,
            )
        )
    return ProteinMPNNSampleOutput(designed_sequences=designed_sequences)
