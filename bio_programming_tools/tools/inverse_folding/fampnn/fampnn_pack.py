"""bio_programming_tools/tools/inverse_folding/fampnn/fampnn_pack.py

FAMPNN sidechain packing tool."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pydantic import Field
from tqdm import tqdm

from bio_programming_tools.tools.inverse_folding.fampnn.fampnn_sample import (
    FAMPNNStructureInput,
)
from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)


class FAMPNNPackInput(BaseToolInput):
    """Input for FAMPNN sidechain packing.

    Attributes:
        inputs (list[FAMPNNStructureInput]): List of structure inputs for sidechain packing.
    """

    inputs: List[FAMPNNStructureInput] = InputField(
        description="List of structure inputs for sidechain packing."
    )


class FAMPNNPackConfig(BaseConfig):
    """Configuration for FAMPNN sidechain packing.

    Attributes:
        model_variant (str): Checkpoint variant. '0.0' recommended for best packing accuracy.
        num_samples_per_structure (int): Number of packing samples per input structure.
        batch_size (int): Number of samples to process simultaneously on GPU.
        scn_diffusion_steps (int): Number of sidechain diffusion denoising steps.
        scn_step_scale (float): Step scale for sidechain diffusion.
        seed (int): Random seed.
        device (str): Device to run on.
    """

    model_variant: str = ConfigField(
        title="Model Variant",
        default="0.0",
        description="FAMPNN checkpoint: '0.0' recommended for packing, '0.3' for design",
        examples=["0.0", "0.3"],
    )
    num_samples_per_structure: int = ConfigField(
        title="Samples Per Structure",
        default=1,
        ge=1,
        description="Number of packing samples per input structure.",
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=1,
        ge=1,
        description="Number of samples to process simultaneously on GPU.",
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
    seed: int = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed",
        hidden=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )


class FAMPNNPackingResult(BaseToolOutput):
    """Output for FAMPNN sidechain packing.

    Attributes:
        packed_structures (list[list[str]]): List of lists of PDB strings with packed sidechain
            coordinates. Outer list corresponds to input structures, inner list
            to packing samples. B-factor column contains per-atom pSCE.
        psce (list[list[list[float]]]): Per-residue predicted sidechain error (Angstroms) for each sample.
    """

    packed_structures: List[List[str]] = Field(
        description="PDB strings with packed sidechains (outer=structures, inner=samples)"
    )
    psce: List[List[List[float]]] = Field(
        description="Per-residue pSCE for each sample (outer=structures, inner=samples)"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["pdb", "json"]

    @property
    def output_format_default(self) -> str:
        return "pdb"

    def _export_output(self, export_path, file_format):
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)

        if file_format == "pdb":
            for i, pdb_list in enumerate(self.packed_structures):
                for j, pdb_str in enumerate(pdb_list):
                    out_file = path / f"packed_{i}_sample_{j}.pdb"
                    out_file.write_text(pdb_str)
        elif file_format == "json":
            import json
            for i, (pdb_list, psce_list) in enumerate(zip(self.packed_structures, self.psce)):
                out_file = path / f"packed_{i}.json"
                with open(out_file, "w") as f:
                    json.dump({"pdb_strings": pdb_list, "psce": psce_list}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input():
    """Minimal valid input for testing and examples."""
    return FAMPNNPackInput(
        inputs=[FAMPNNStructureInput(
            structure=str(Path(__file__).parents[4] / "tests" / "dummy_data" / "test_structure_similarity.pdb"),
        )]
    )


@tool(
    key="fampnn-pack",
    label="FAMPNN Sidechain Packing",
    category="inverse_folding",
    input_class=FAMPNNPackInput,
    config_class=FAMPNNPackConfig,
    output_class=FAMPNNPackingResult,
    description="Pack protein sidechains using FAMPNN with per-atom confidence (pSCE)",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="packed_structures",
)
def run_fampnn_pack(
    inputs: FAMPNNPackInput,
    config: FAMPNNPackConfig | None = None,
    instance=None,
) -> FAMPNNPackingResult:
    """Pack protein sidechains using FAMPNN with per-atom confidence scores.

    Given a protein backbone and sequence, predicts sidechain coordinates using
    per-token Euclidean diffusion. Output PDB files contain per-atom predicted
    sidechain error (pSCE) in the B-factor column.

    Args:
        inputs (FAMPNNPackInput): FAMPNNPackInput containing structure inputs.
        config (FAMPNNPackConfig | None): Configuration for packing.
        instance: Optional ToolInstance for persistent execution.

    Returns:
        FAMPNNPackingResult: FAMPNNPackingResult with packed PDB structures and pSCE values.
    """
    all_packed = []
    all_psce = []

    for inp in tqdm(
        inputs.inputs,
        desc="FAMPNN packing",
        unit="structure",
        disable=not config.verbose,
    ):
        struct_pdbs, struct_psce = [], []
        remaining = config.num_samples_per_structure
        chunk_idx = 0
        while remaining > 0:
            chunk = min(config.batch_size, remaining)
            input_dict = {
                "operation": "pack",
                "pdb_contents": inp.structure_pdb,
                "num_samples": chunk,
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
            struct_pdbs.extend(result["pdb_strings"])
            struct_psce.extend(result["psce"])
            chunk_idx += 1
            remaining -= chunk

        all_packed.append(struct_pdbs)
        all_psce.append(struct_psce)

    return FAMPNNPackingResult(
        packed_structures=all_packed,
        psce=all_psce,
    )
