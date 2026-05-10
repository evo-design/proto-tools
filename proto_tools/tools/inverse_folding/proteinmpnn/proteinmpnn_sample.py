"""proto_tools/tools/inverse_folding/proteinmpnn/proteinmpnn_sample.py.

ProteinMPNN sampling tool.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
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

# ============================================================================
# Data Models
# ============================================================================
# Input:
ProteinMPNNSampleInput = InverseFoldingInput


# Config:
class ProteinMPNNSampleConfig(InverseFoldingConfig):
    """Configuration for ProteinMPNN sampling.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits.
        excluded_amino_acids (list[AminoAcid] | None): One-letter codes of amino acids to exclude.
        seed (int): Random seed to use for sampling.
        model_choice (Literal['proteinmpnn', 'v_48_002', 'v_48_010', 'v_48_030', 'abmpnn', 'soluble']): Model
            weights. ``"proteinmpnn"`` is ColabDesign's default ``v_48_020`` (medium training noise). The
            ``v_48_*`` variants are the same architecture trained at different noise levels (002 / 010 / 030).
            ``"abmpnn"`` is antibody-optimized; ``"soluble"`` is soluble-protein-trained.
        backbone_noise (float): Gaussian noise (A) added to backbone coordinates before each forward pass.
    """

    model_choice: Literal["proteinmpnn", "v_48_002", "v_48_010", "v_48_030", "abmpnn", "soluble"] = ConfigField(
        title="Model Choice",
        default="proteinmpnn",
        description="Weights: proteinmpnn (=v_48_020), v_48_{002,010,030} noise variants, abmpnn, soluble",
        reload_on_change=True,
        examples=["proteinmpnn", "v_48_010", "abmpnn", "soluble"],
    )
    backbone_noise: float = ConfigField(
        title="Backbone Noise",
        default=0.0,
        ge=0.0,
        description="Gaussian noise (A) on backbone coords; raise (e.g. 0.02) for diversity",
        advanced=True,
    )
    excluded_amino_acids: list[AminoAcid] | None = ConfigField(
        title="Excluded Amino Acids",
        default=None,
        description="Single-letter codes of amino acids to exclude (e.g. ['C'] to forbid cysteine)",
        examples=[["C"]],
    )


class ProteinMPNNSequences(DesignedSequences):
    """Represents a designed sequence from the ProteinMPNN model.

    Attributes:
        sequences (list[str]): Designed protein sequences.
        perplexity (list[float]): Per-sequence perplexity values.
        sequence_recovery (list[float]): Per-sequence fraction of chains_to_redesign
            residues matching the PDB prompt reference (0.0-1.0).
    """

    perplexity: list[float] = Field(description="Perplexity of the sequence from the ProteinMPNN model")
    sequence_recovery: list[float] = Field(
        description="Per-sequence fraction of chains_to_redesign residues matching the PDB prompt reference (0.0-1.0)",
    )


class ProteinMPNNSampleOutput(InverseFoldingOutput):
    """Output of the ProteinMPNN sampling tool."""

    designed_sequences: list[ProteinMPNNSequences] = Field(  # type: ignore[assignment]
        description="ProteinMPNN-designed sequences with per-sequence perplexity and recovery metrics.",
    )


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return ProteinMPNNSampleInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="proteinmpnn-sample",
    label="ProteinMPNN Sampling",
    category="inverse_folding",
    input_class=ProteinMPNNSampleInput,
    config_class=ProteinMPNNSampleConfig,
    output_class=ProteinMPNNSampleOutput,
    description="Sample protein sequences using ProteinMPNN",
    uses_gpu=True,
    generative=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="designed_sequences",
)
def run_proteinmpnn_sample(
    inputs: ProteinMPNNSampleInput,
    config: ProteinMPNNSampleConfig,
    instance: Any = None,
) -> ProteinMPNNSampleOutput:
    """Sample protein sequences using ProteinMPNN.

    Args:
        inputs (ProteinMPNNSampleInput): ProteinMPNNSampleInput containing a list of structure inputs,
            each with optional ``chains_to_redesign`` and ``fixed_positions`` selections.
        config (ProteinMPNNSampleConfig): Configuration for sampling (temperature, batch_size, etc.).

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        ProteinMPNNSampleOutput: ProteinMPNNSampleOutput with designed sequences for each input structure.

    Note:
        Multi-chain sampling returns a "/"-delimited sequence preserving chain ID order.
    """
    designed_sequences = []

    # Local venv execution
    logger.debug("Using local venv for ProteinMPNN sampling")

    base_seed = config.seed if config.seed is not None else config.get_random_int()

    for inp in progress_bar(
        inputs.inputs,
        desc="ProteinMPNN sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        all_seqs, all_perp, all_seq_recovery = [], [], []
        remaining = config.num_sequences_per_structure
        chunk_idx = 0
        # Materialize the Structure to a tempfile once per input — reused across chunks.
        with inp.structure.temp_file() as pdb_path:
            while remaining > 0:
                chunk = min(config.batch_size, remaining)  # type: ignore[type-var]
                input_dict = {
                    "operation": "sample",
                    "pdb_path": str(pdb_path),
                    "chain_ids": inp.chain_ids_to_redesign,
                    "batch_size": chunk,
                    "temperature": config.temperature,
                    "fixed_positions": inp.fixed_positions.chains if inp.fixed_positions is not None else None,
                    "excluded_amino_acids": config.excluded_amino_acids,
                    "seed": base_seed + chunk_idx,
                    "device": config.device,
                    "model_choice": config.model_choice,
                    "verbose": config.verbose,
                    "return_logits": False,
                    "backbone_noise": config.backbone_noise,
                }
                result = ToolInstance.dispatch(
                    "proteinmpnn",
                    input_dict,
                    instance=instance,
                    config=config,
                )
                all_seqs.extend(result["seq"])
                all_perp.extend(np.exp(result["score"]).tolist())
                all_seq_recovery.extend(result["seqid"])
                chunk_idx += 1
                remaining -= chunk  # type: ignore[operator]
        designed_sequences.append(
            ProteinMPNNSequences(
                sequences=all_seqs,
                perplexity=all_perp,
                sequence_recovery=all_seq_recovery,
            )
        )
    return ProteinMPNNSampleOutput(designed_sequences=designed_sequences)
