"""proto_tools/tools/inverse_folding/esm_if1/esm_if1_sample.py.

ESM-IF1/ProteinDPO sampling tool.
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
from proto_tools.utils import ConfigField
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
ESMIF1SampleInput = InverseFoldingInput
ESMIF1SampleOutput = InverseFoldingOutput


class ESMIF1SampleConfig(InverseFoldingConfig):
    """Configuration for ESM-IF1/ProteinDPO sequence sampling.

    Attributes:
        weights_variant (Literal['esmif', 'protein_dpo']): Which model weights to use. 'esmif' loads vanilla ESM-IF1,
            'protein_dpo' loads DPO-aligned weights optimized for protein stability.
        num_sequences_per_structure (int): Total number of sequences to generate per structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
        temperature (float): Controls randomness in sampling from logits.
        excluded_amino_acids (list[str] | None): Amino acids disallowed in the designed sequence.
        seed (int): Random seed for sampling reproducibility.
    """

    weights_variant: Literal["esmif", "protein_dpo"] = ConfigField(
        title="Weights Variant",
        default="protein_dpo",
        description="'esmif' for vanilla ESM-IF1, 'protein_dpo' for DPO-aligned weights",
        reload_on_change=True,
        examples=["esmif", "protein_dpo"],
    )


class ESMIF1Sequences(DesignedSequences):
    """Designed sequences from ESM-IF1/ProteinDPO.

    Attributes:
        sequences (list[str]): Designed amino acid sequences.
        log_likelihoods (list[float]): Average log likelihood of each designed sequence
            under the model.
    """

    log_likelihoods: list[float] = Field(description="Average log likelihood of each designed sequence under the model")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ESMIF1SampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(  # type: ignore[arg-type]
                    Path(__file__).parents[4] / "tests" / "dummy_data" / "test_structure_similarity.pdb"
                ),
            )
        ]
    )


@tool(
    key="esm-if1-sample",
    label="ESM-IF1 Sampling",
    category="inverse_folding",
    input_class=ESMIF1SampleInput,
    config_class=ESMIF1SampleConfig,
    output_class=ESMIF1SampleOutput,
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
def run_esm_if1_sample(
    inputs: ESMIF1SampleInput,
    config: ESMIF1SampleConfig,
    instance: Any = None,
) -> ESMIF1SampleOutput:
    """Sample protein sequences using ESM-IF1/ProteinDPO.

    Args:
        inputs (ESMIF1SampleInput): Structure inputs with optional chain/fixed position constraints.
        config (ESMIF1SampleConfig): Configuration including weights variant, temperature, etc.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ESMIF1SampleOutput: ESMIF1SampleOutput with designed sequences for each input structure.
    """
    if config.excluded_amino_acids:
        raise ValueError("ESM-IF1 does not support excluded_amino_acids. This feature may be added in a future update.")

    designed_sequences = []

    for inp in progress_bar(
        inputs.inputs,
        desc="ESM-IF1 sampling",
        unit="structure",
        total=len(inputs.inputs),
    ):
        all_seqs, all_lls = [], []
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
                "seed": config.resolved_seed + chunk_idx,
                "device": config.device,
                "weights_variant": config.weights_variant,
                "verbose": config.verbose,
                "fixed_positions": inp.fixed_positions,
            }
            result = ToolInstance.dispatch(
                "esm_if1",
                input_dict,
                instance=instance,
                config=config,
            )
            all_seqs.extend(result["sequences"])
            all_lls.extend(result["log_likelihoods"])
            chunk_idx += 1
            remaining -= chunk  # type: ignore[operator]
        designed_sequences.append(
            ESMIF1Sequences(
                sequences=all_seqs,
                log_likelihoods=all_lls,
            )
        )

    return ESMIF1SampleOutput(designed_sequences=designed_sequences)  # type: ignore[arg-type]
