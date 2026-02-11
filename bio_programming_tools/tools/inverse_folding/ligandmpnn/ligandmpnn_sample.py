"""LigandMPNN sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field
from tqdm import tqdm

from bio_programming_tools.tools.infra.env_manager import EnvManager
from bio_programming_tools.tools.infra.tool_cache import tool_cache
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
@tool(
    key="ligandmpnn-sample",
    label="LigandMPNN Sampling",
    input=LigandMPNNSampleInput,
    config=LigandMPNNSampleConfig,
    output=LigandMPNNSampleOutput,
    description="Sample protein sequences using LigandMPNN",
)
@tool_cache("ligandmpnn-sample")
def run_ligandmpnn_sample(
    inputs: LigandMPNNSampleInput,
    config: LigandMPNNSampleConfig,
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

    if use_cloud_gpu():
        import _gpu_runtime

        LigandMPNNService = _gpu_runtime.Cls.from_name("bio-programming", "LigandMPNNService")
        service = LigandMPNNService()

        for inp in tqdm(
            inputs.inputs,
            desc="LigandMPNN sampling",
            unit="structure",
            total=len(inputs.inputs),
        ):
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
                LigandMPNNSequences(
                    sequences=result["sequences"],
                    ligandmpnn_metrics=result["metrics"],
                )
            )
    else:
        # Local venv execution
        venv_manager = EnvManager("ligandmpnn")
        script_path = Path(__file__).parent / "standalone" / "inference.py"

        for inp in tqdm(
            inputs.inputs,
            desc="LigandMPNN sampling",
            unit="structure",
            total=len(inputs.inputs),
        ):
            input_dict = {
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
                LigandMPNNSequences(
                    sequences=result["sequences"],
                    ligandmpnn_metrics=result["metrics"],
                )
            )

    return LigandMPNNSampleOutput(designed_sequences=designed_sequences)
