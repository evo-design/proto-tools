"""ProteinMPNN sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from pydantic import Field
from tqdm import tqdm

from bio_programming_tools.tools.infra.env_manager import EnvManager
from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.tools.utils import use_cloud_gpu

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
    input=ProteinMPNNSampleInput,
    config=ProteinMPNNSampleConfig,
    output=ProteinMPNNSampleOutput,
    description="Sample protein sequences using ProteinMPNN",
)
def run_proteinmpnn_sample(
    inputs: ProteinMPNNSampleInput,
    config: ProteinMPNNSampleConfig,
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

    if use_cloud_gpu():
        # the cloud runtime
        logger.debug("Using the cloud runtime for ProteinMPNN sampling")

        import _gpu_runtime

        ProteinMPNNService = _gpu_runtime.Cls.from_name("bio-programming", "ProteinMPNNService")
        service = ProteinMPNNService()

        for inp in tqdm(inputs.inputs, desc="ProteinMPNN sampling", unit="structure"):
            result = service.sample.remote(
                pdb_structure=inp.structure_pdb,
                chain_ids=inp.chain_ids,
                batch_size=config.batch_size,
                temperature=config.temperature,
                fixed_positions=inp.fixed_positions,
                excluded_amino_acids=config.excluded_amino_acids,
                seed=config.seed,
            )
            designed_sequences.append(
                ProteinMPNNSequences(
                    sequences=result["seq"],
                    perplexity=np.exp(result["score"]).tolist(),
                    sequence_identity=result["seqid"],
                )
            )
    else:
        # Local venv execution
        logger.debug("Using local venv for ProteinMPNN sampling")

        venv_manager = EnvManager("proteinmpnn")
        script_path = Path(__file__).parent / "standalone" / "inference.py"

        for inp in tqdm(
            inputs.inputs,
            desc="ProteinMPNN sampling",
            unit="structure",
            disable=not config.verbose,
        ):
            input_dict = {
                "operation": "sample",
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "batch_size": config.batch_size,
                "temperature": config.temperature,
                "fixed_positions": inp.fixed_positions,
                "excluded_amino_acids": config.excluded_amino_acids,
                "seed": config.seed,
                "device": config.device,
            }
            result = venv_manager.call_standalone_script_in_venv(
                script_path=script_path,
                input_dict=input_dict,
                device=config.device,
                verbose=config.verbose,
            )
            designed_sequences.append(
                ProteinMPNNSequences(
                    sequences=result["seq"],
                    perplexity=np.exp(result["score"]).tolist(),
                    sequence_identity=result["seqid"],
                )
            )
    return ProteinMPNNSampleOutput(designed_sequences=designed_sequences)
