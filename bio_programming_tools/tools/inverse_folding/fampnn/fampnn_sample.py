"""bio_programming_tools/tools/inverse_folding/fampnn/fampnn_sample.py

FAMPNN sequence sampling tool with full-atom sidechain co-generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from tqdm import tqdm

from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseToolInput,
    ConfigField,
    InputField,
    ToolInstance,
)


class FAMPNNStructureInput(InverseFoldingStructureInput):
    """FAMPNN structure input with optional sidechain position constraints.

    Extends InverseFoldingStructureInput with fixed_sidechain_positions for
    conditioning on known sidechain conformations during design/packing.

    FAMPNN introduces fixed_sidechain_positions as a separate constraint from
    fixed_positions (dict[str, list[int]] | None):
    - fixed_positions: Residue positions whose amino acid identity is kept fixed
      during sequence design (the model will not redesign these positions).
    - fixed_sidechain_positions: Residue positions whose sidechain atom
      coordinates are used as structural context (the model conditions on their
      3D geometry).

    Attributes:
        fixed_sidechain_positions (dict[str, list[int]] | None): Optional dictionary mapping chain IDs to
            residue positions whose sidechain coordinates should be used as
            context during sampling/packing. Positions are 1-indexed.
    """

    fixed_sidechain_positions: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Chain IDs to residue positions with known sidechain coordinates to condition on (1-indexed).",
    )

logger = logging.getLogger(__name__)


class FAMPNNSampleInput(BaseToolInput):
    """Input for FAMPNN sequence sampling.

    Attributes:
        inputs (list[FAMPNNStructureInput]): List of FAMPNN structure inputs, each containing a structure
            and optional chain_ids/fixed_positions/fixed_sidechain_positions.
    """

    inputs: List[FAMPNNStructureInput] = InputField(
        description="List of structure inputs for sequence design."
    )


class FAMPNNSampleConfig(InverseFoldingConfig):
    """Configuration for FAMPNN sequence sampling.

    Extends InverseFoldingConfig with FAMPNN-specific parameters for
    iterative masked language modeling and sidechain diffusion.

    Attributes:
        num_sequences_per_structure (int): Total number of sequences to generate per
            input structure.
        batch_size (int | None): Number of sequences to process simultaneously on GPU.
            Defaults to num_sequences_per_structure.
        temperature (float): Controls randomness in sampling from logits.
        excluded_amino_acids (list[str] | None): List of amino acids not allowed in the sequence.
            Not supported by FAMPNN (raises ValueError if set).
        seed (int): Random seed to use for sampling.
        model_variant (str): FAMPNN checkpoint variant. '0.3' for sequence design
            (PDB-trained, 0.3A noise), '0.0' for sidechain packing (PDB-trained,
            0.0A noise), '0.3_cath' for mutation scoring (CATH-trained).
        num_steps (int): Number of iterative unmasking steps for sequence design.
            More steps yield higher quality but slower inference. 10 steps is
            sufficient for high self-consistency; 100 for best quality.
        seq_only (bool): If True, skip sidechain generation during sampling.
        repack_last (bool): If True, repack sidechains after final sequence is determined.
        psce_threshold (float): Only condition on sidechains with predicted sidechain
            error below this threshold during iterative sampling.
        scn_diffusion_steps (int): Number of sidechain diffusion denoising steps.
        scn_step_scale (float): Step scale for sidechain diffusion (eta parameter).
    """

    model_variant: str = ConfigField(
        title="Model Variant",
        default="0.3",
        description="FAMPNN checkpoint: '0.3' (design), '0.0' (packing), '0.3_cath' (scoring)",
        examples=["0.3", "0.0", "0.3_cath"],
    )
    num_steps: int = ConfigField(
        title="Unmasking Steps",
        default=100,
        ge=1,
        description="Number of iterative unmasking steps for sequence design",
        examples=[10, 50, 100],
    )
    seq_only: bool = ConfigField(
        title="Sequence Only",
        default=False,
        description="If True, skip sidechain generation during sampling",
    )
    repack_last: bool = ConfigField(
        title="Repack Last",
        default=True,
        description="Repack sidechains after final sequence is determined",
    )
    psce_threshold: float = ConfigField(
        title="pSCE Threshold",
        default=0.3,
        ge=0.0,
        description="Only keep sidechains below this predicted error threshold during design",
        examples=[0.3, 0.5, 1.0],
    )
    scn_diffusion_steps: int = ConfigField(
        title="Sidechain Diffusion Steps",
        default=50,
        ge=1,
        description="Number of sidechain diffusion denoising steps",
        hidden=True,
    )
    scn_step_scale: float = ConfigField(
        title="Sidechain Step Scale",
        default=1.5,
        gt=0.0,
        description="Step scale (eta) for sidechain diffusion",
        hidden=True,
    )

    @field_validator("excluded_amino_acids")
    @classmethod
    def _reject_excluded_aa(cls, v):
        if v is not None:
            raise ValueError("'excluded_amino_acids' is not supported by FAMPNN")
        return v


class FAMPNNSequences(DesignedSequences):
    """Designed sequences from FAMPNN with full-atom sidechain outputs.

    Attributes:
        sequences (list[str]): Designed amino acid sequences.
        output_pdb_strings (list[str]): PDB-format strings with designed sequence and
            packed sidechain coordinates. B-factor column contains per-atom pSCE.
        psce (list[list[float]]): Per-residue predicted sidechain error (mean over atoms) in Angstroms.
    """

    output_pdb_strings: List[str] = Field(
        description="PDB strings with designed sequences and sidechain coordinates"
    )
    psce: List[List[float]] = Field(
        description="Per-residue predicted sidechain error (Angstroms)"
    )


FAMPNNSampleOutput = InverseFoldingOutput


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return FAMPNNSampleInput(
        inputs=[FAMPNNStructureInput(
            structure=str(Path(__file__).parents[4] / "tests" / "dummy_data" / "test_structure_similarity.pdb"),
        )]
    )


@tool(
    key="fampnn-sample",
    label="FAMPNN Sampling",
    category="inverse_folding",
    input_class=FAMPNNSampleInput,
    config_class=FAMPNNSampleConfig,
    output_class=FAMPNNSampleOutput,
    description="Design protein sequences with full-atom sidechain co-generation using FAMPNN",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="designed_sequences",
)
def run_fampnn_sample(
    inputs: FAMPNNSampleInput,
    config: FAMPNNSampleConfig | None = None,
    instance=None,
) -> FAMPNNSampleOutput:
    """Design protein sequences with full-atom sidechain co-generation using FAMPNN.

    FAMPNN iteratively unmasks sequence and sidechain tokens, jointly generating
    amino acid identities and sidechain conformations. The output includes
    full-atom PDB structures with predicted sidechain coordinates and per-residue
    confidence scores (pSCE).

    Args:
        inputs (FAMPNNSampleInput): FAMPNNSampleInput containing structure inputs with optional
            chain_ids, fixed_positions, and fixed_sidechain_positions.
        config (FAMPNNSampleConfig | None): Configuration for sampling (temperature, num_steps, etc.).
        instance: Optional ToolInstance for persistent execution.

    Returns:
        FAMPNNSampleOutput: FAMPNNSampleOutput with designed sequences, PDB strings, and pSCE values.
    """
    designed_sequences = []

    for inp in tqdm(
        inputs.inputs,
        desc="FAMPNN sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        all_seqs, all_pdbs, all_psce = [], [], []
        remaining = config.num_sequences_per_structure
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)
            input_dict = {
                "operation": "sample",
                "pdb_contents": inp.structure_pdb,
                "chain_ids": inp.chain_ids,
                "num_sequences": chunk,
                "temperature": config.temperature,
                "num_steps": config.num_steps,
                "seq_only": config.seq_only,
                "repack_last": config.repack_last,
                "psce_threshold": config.psce_threshold,
                "scn_diffusion_steps": config.scn_diffusion_steps,
                "scn_step_scale": config.scn_step_scale,
                "seed": config.seed + chunk_idx,
                "model_variant": config.model_variant,
                "device": config.device,
                "verbose": config.verbose,
                "fixed_positions": inp.fixed_positions,
                "fixed_sidechain_positions": inp.fixed_sidechain_positions,
            }
            result = ToolInstance.dispatch(
                "fampnn",
                input_dict,
                instance=instance,
                config=config,
            )
            all_seqs.extend(result["sequences"])
            all_pdbs.extend(result["pdb_strings"])
            all_psce.extend(result["psce"])
            chunk_idx += 1
            remaining -= chunk

        designed_sequences.append(
            FAMPNNSequences(
                sequences=all_seqs,
                output_pdb_strings=all_pdbs,
                psce=all_psce,
            )
        )

    return FAMPNNSampleOutput(designed_sequences=designed_sequences)
