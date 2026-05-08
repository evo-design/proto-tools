"""proto_tools/tools/inverse_folding/fampnn/fampnn_sample.py.

FAMPNN sequence sampling tool with full-atom sidechain co-generation.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.entities.structures import ResidueSelection
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingConfig,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseToolInput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.progress import progress_bar


class FAMPNNStructureInput(InverseFoldingStructureInput):
    """FAMPNN structure input with optional sidechain-conditioning selection.

    Extends :class:`InverseFoldingStructureInput` with ``fixed_sidechain_positions`` for
    conditioning on known sidechain conformations during design/packing.

    FAMPNN distinguishes two kinds of fixed-residue constraints:

    - ``fixed_positions`` (inherited): residue positions whose amino acid identity is kept
      fixed during sequence design (the model will not redesign these positions).
    - ``fixed_sidechain_positions``: residue positions whose sidechain atom coordinates
      are used as structural context (the model conditions on their 3D geometry).

    Attributes:
        fixed_sidechain_positions (ResidueSelection | None): Per-chain residue positions
            whose sidechain coordinates condition the model during
            sampling/packing (1-indexed). Accepts shorthand ``{"A": [1, 2]}`` at
            construction.
    """

    fixed_sidechain_positions: ResidueSelection | None = Field(
        default=None,
        description="Per-chain positions whose sidechain coordinates condition the model (1-indexed).",
    )


logger = logging.getLogger(__name__)


class FAMPNNSampleInput(BaseToolInput):
    """Input for FAMPNN sequence sampling.

    Attributes:
        inputs (list[FAMPNNStructureInput]): Per-structure inputs, each
            containing a structure and optional ``chains_to_redesign`` / ``fixed_positions`` /
            ``fixed_sidechain_positions`` selections.
    """

    inputs: list[FAMPNNStructureInput] = InputField(description="List of structure inputs for sequence design.")


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
        reload_on_change=True,
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


class FAMPNNSequences(DesignedSequences):
    """Designed sequences from FAMPNN with full-atom sidechain outputs.

    Attributes:
        sequences (list[str]): Designed amino acid sequences.
        output_pdb_strings (list[str]): PDB-format strings with designed sequence and
            packed sidechain coordinates. B-factor column contains per-atom pSCE.
        psce (list[list[float]]): Per-residue predicted sidechain error (mean over atoms) in Angstroms.
    """

    output_pdb_strings: list[str] = Field(description="PDB strings with designed sequences and sidechain coordinates")
    psce: list[list[float]] = Field(description="Per-residue predicted sidechain error (Angstroms)")


FAMPNNSampleOutput = InverseFoldingOutput


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return FAMPNNSampleInput(
        inputs=[
            FAMPNNStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
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
    cacheable=True,
    generative=True,
)
def run_fampnn_sample(
    inputs: FAMPNNSampleInput,
    config: FAMPNNSampleConfig,
    instance: Any = None,
) -> FAMPNNSampleOutput:
    """Design protein sequences with full-atom sidechain co-generation using FAMPNN.

    FAMPNN iteratively unmasks sequence and sidechain tokens, jointly generating
    amino acid identities and sidechain conformations. The output includes
    full-atom PDB structures with predicted sidechain coordinates and per-residue
    confidence scores (pSCE).

    Args:
        inputs (FAMPNNSampleInput): FAMPNNSampleInput containing structure inputs with optional
            ``chains_to_redesign``, ``fixed_positions``, and ``fixed_sidechain_positions`` selections.
        config (FAMPNNSampleConfig): Configuration for sampling (temperature, num_steps, etc.).
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        FAMPNNSampleOutput: FAMPNNSampleOutput with designed sequences, PDB strings, and pSCE values.
    """
    designed_sequences = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()

    for inp in progress_bar(
        inputs.inputs,
        desc="FAMPNN sampling",
        unit="structure",
        disable=not config.verbose,
    ):
        all_seqs, all_pdbs, all_psce = [], [], []
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
                    "num_sequences": chunk,
                    "temperature": config.temperature,
                    "num_steps": config.num_steps,
                    "seq_only": config.seq_only,
                    "repack_last": config.repack_last,
                    "psce_threshold": config.psce_threshold,
                    "scn_diffusion_steps": config.scn_diffusion_steps,
                    "scn_step_scale": config.scn_step_scale,
                    "seed": base_seed + chunk_idx,
                    "model_variant": config.model_variant,
                    "device": config.device,
                    "verbose": config.verbose,
                    "fixed_positions": inp.fixed_positions.chains if inp.fixed_positions is not None else None,
                    "fixed_sidechain_positions": (
                        inp.fixed_sidechain_positions.chains if inp.fixed_sidechain_positions is not None else None
                    ),
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
                remaining -= chunk  # type: ignore[operator]

        designed_sequences.append(
            FAMPNNSequences(
                sequences=all_seqs,
                output_pdb_strings=all_pdbs,
                psce=all_psce,
            )
        )

    return FAMPNNSampleOutput(designed_sequences=designed_sequences)  # type: ignore[arg-type]
