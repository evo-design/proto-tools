"""ESM-IF/ProteinDPO sampling tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal

from pydantic import Field
from tqdm import tqdm

from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import ConfigField
from bio_programming_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
ESMIFSampleInput = InverseFoldingInput
ESMIFSampleOutput = InverseFoldingOutput


class ESMIFSampleConfig(InverseFoldingConfig):
    """Configuration for ESM-IF/ProteinDPO sequence sampling.

    Attributes:
        weights_variant: Which model weights to use. 'esmif' loads vanilla ESM-IF1,
            'protein_dpo' loads DPO-aligned weights optimized for protein stability.
    """

    weights_variant: Literal["esmif", "protein_dpo"] = ConfigField(
        title="Weights Variant",
        default="protein_dpo",
        description=(
            "Model weights to use: 'esmif' for vanilla ESM-IF1, "
            "'protein_dpo' for DPO-aligned weights optimized for stability"
        ),
        reload_on_change=True,
        examples=["esmif", "protein_dpo"],
    )


class ESMIFSequences(DesignedSequences):
    """Designed sequences from ESM-IF/ProteinDPO.

    Attributes:
        sequences: Designed amino acid sequences.
        log_likelihoods: Average log likelihood of each designed sequence
            under the model.
    """

    log_likelihoods: List[float] = Field(
        description="Average log likelihood of each designed sequence under the model"
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return ESMIFSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(
                    Path(__file__).parents[4]
                    / "tests"
                    / "dummy_data"
                    / "test_structure_similarity.pdb"
                ),
            )
        ]
    )


@tool(
    key="esmif-sample",
    label="ESM-IF Sampling",
    category="inverse_folding",
    input_class=ESMIFSampleInput,
    config_class=ESMIFSampleConfig,
    output_class=ESMIFSampleOutput,
    description=(
        "Sample protein sequences conditioned on backbone structure using "
        "ESM-IF1 or ProteinDPO (DPO-aligned for stability). Supports "
        "multi-chain complexes."
    ),
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="designed_sequences",
    cacheable=True,
)
def run_esmif_sample(
    inputs: ESMIFSampleInput,
    config: ESMIFSampleConfig | None = None,
    instance=None,
) -> ESMIFSampleOutput:
    """Sample protein sequences using ESM-IF/ProteinDPO.

    Args:
        inputs: Structure inputs with optional chain/fixed position constraints.
        config: Configuration including weights variant, temperature, etc.

    Returns:
        ESMIFSampleOutput with designed sequences for each input structure.
    """
    if config.excluded_amino_acids:
        raise ValueError(
            "ESM-IF does not support excluded_amino_acids. "
            "This feature may be added in a future update."
        )

    designed_sequences = []

    for inp in tqdm(
        inputs.inputs,
        desc="ESM-IF sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_seqs, all_lls = [], []
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
                "seed": config.seed + chunk_idx,
                "device": config.device,
                "weights_variant": config.weights_variant,
                "verbose": config.verbose,
            }
            result = ToolInstance.dispatch(
                "esmif",
                input_dict,
                instance=instance,
                config=config,
            )
            all_seqs.extend(result["sequences"])
            all_lls.extend(result["log_likelihoods"])
            chunk_idx += 1
            remaining -= chunk
        designed_sequences.append(
            ESMIFSequences(
                sequences=all_seqs,
                log_likelihoods=all_lls,
            )
        )

    return ESMIFSampleOutput(designed_sequences=designed_sequences)
