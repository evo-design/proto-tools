"""proto_tools/tools/inverse_folding/caliby/caliby_pack.py.

Caliby diffusion-based sidechain packing.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.caliby.caliby_sample import (
    PACKER_CHECKPOINTS,
    validate_checkpoint,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)


class CalibyPackInput(BaseToolInput):
    """Input for Caliby sidechain packing.

    Attributes:
        structures (list[Structure]): Structures to pack sidechains onto. Accepts file
            paths, raw PDB/CIF content strings, or ``Structure`` objects per item. Each
            must already carry the sequence to pack. Packing places sidechain atoms for
            the residue identities the structure already has.
    """

    structures: list[Structure] = InputField(
        title="Structures",
        description="Structures to pack sidechains onto, each carrying the sequence to pack.",
    )

    @field_validator("structures", mode="before")
    @classmethod
    def _wrap_single_in_list(cls, value: Any) -> list[Any]:
        """Wrap a single Structure / path / dict input in a list; per-item coercion handles the rest."""
        if isinstance(value, (str, Path, Structure, dict)):
            return [value]
        return list(value)


class CalibyPackConfig(BaseConfig):
    """Configuration for Caliby sidechain packing.

    Attributes:
        model_checkpoint (str): Bundled packer checkpoint name or a path to a ``.ckpt``
            file. The suffix is the training noise level in Angstroms:
            ``'caliby_packer_010'`` (0.1 A) is recommended for most cases,
            ``'caliby_packer_000'`` for noise-free backbones, ``'caliby_packer_030'``
            for noisier ones. Packing needs a packer checkpoint; design checkpoints are
            rejected.
        diffusion_steps (int): Sidechain diffusion denoising steps.
        step_scale (float): Step scale (eta) for sidechain diffusion.
        device (str): Device to run the model on.
    """

    model_checkpoint: str = ConfigField(
        title="Model Checkpoint",
        default="caliby_packer_010",
        description="Bundled packer checkpoint name, or a path to a custom .ckpt file.",
        examples=["caliby_packer_010", "caliby_packer_000", "caliby_packer_030"],
        reload_on_change=True,
    )
    diffusion_steps: int = ConfigField(
        title="Diffusion Steps",
        default=50,
        ge=1,
        description="Number of sidechain diffusion denoising steps.",
    )
    step_scale: float = ConfigField(
        title="Step Scale",
        default=1.5,
        gt=0.0,
        description="Step scale (eta) for sidechain diffusion.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on, e.g. 'cuda', 'cuda:0' or 'cpu'.",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )

    @field_validator("model_checkpoint")
    @classmethod
    def validate_model_checkpoint(cls, value: str) -> str:
        """Reject a design checkpoint; packing needs a ``task='scn_pack'`` model."""
        return validate_checkpoint(value, PACKER_CHECKPOINTS, "packing")

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A custom ``.ckpt`` path is not present on a hosted worker."""
        if self.model_checkpoint.endswith(".ckpt"):
            return (
                f"model_checkpoint points to a local .ckpt file not available on device='{device}'. "
                "Use a bundled checkpoint name, or run locally with device='cuda'."
            )
        return None


class CalibyPackOutput(BaseToolOutput):
    """Output of the Caliby sidechain packing tool.

    Attributes:
        packed_structures (list[Structure]): Packed full-atom structures, one per input
            structure, in input order.
    """

    packed_structures: list[Structure] = Field(
        title="Packed Structures",
        description="Packed full-atom structures, one per input structure, in input order.",
    )

    def __len__(self) -> int:
        """Get the number of packed structures."""
        return len(self.packed_structures)

    def __getitem__(self, index: int) -> Structure:
        """Get a packed structure by index."""
        return self.packed_structures[index]

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["cif", "pdb"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "cif"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)

        if file_format == "cif":
            for i, structure in enumerate(self.packed_structures):
                structure.write_cif(path / f"packed_{i}.cif")
        elif file_format == "pdb":
            for i, structure in enumerate(self.packed_structures):
                structure.write_pdb(path / f"packed_{i}.pdb")
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CalibyPackInput(structures=[str(Path(__file__).parents[1] / "example_input_fixture.pdb")])


@tool(
    key="caliby-pack",
    label="Caliby Sidechain Packing",
    category="inverse_folding",
    input_class=CalibyPackInput,
    config_class=CalibyPackConfig,
    output_class=CalibyPackOutput,
    description="Pack protein sidechains onto a backbone using Caliby's diffusion packer",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["structures"],
    iterable_output_field="packed_structures",
    max_chunk_size=32,
    cacheable=True,
    stochastic=True,
)
def run_caliby_pack(
    inputs: CalibyPackInput,
    config: CalibyPackConfig,
    instance: Any = None,
) -> CalibyPackOutput:
    """Pack protein sidechains onto a backbone with Caliby's diffusion packer.

    Places sidechain atoms for the residue identities each input structure already
    carries, by per-token Euclidean diffusion. This is the companion step to
    ``caliby-sample``, whose designs are backbone-only.

    Args:
        inputs (CalibyPackInput): Structures to pack, each carrying the sequence to pack.
        config (CalibyPackConfig): Packing configuration.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        CalibyPackOutput: Packed full-atom structures, one per input, in input order.

    Examples:
        >>> designs = run_caliby_sample(sample_inputs, CalibySampleConfig())
        >>> packed = run_caliby_pack(CalibyPackInput(structures=[designs[0][0].structure]))
    """
    packed: list[Structure] = []

    base_seed = config.seed if config.seed is not None else config.get_random_int()

    logger.debug("Using local venv for caliby sidechain packing")

    for dispatch_idx, structure in enumerate(
        progress_bar(
            inputs.structures,
            desc="Caliby packing",
            unit="structure",
            disable=not config.verbose,
        )
    ):
        # PDB, not CIF: Caliby reads chain IDs from label_asym_id, which write_cif leaves unset.
        with structure.temp_file("pdb") as structure_path:
            input_dict = {
                "operation": "pack",
                "primary_path": str(structure_path),
                "diffusion_steps": config.diffusion_steps,
                "step_scale": config.step_scale,
                "seed": base_seed + dispatch_idx,
                "model_checkpoint": config.model_checkpoint,
                "device": config.device,
            }
            result = ToolInstance.dispatch("caliby", input_dict, instance=instance, config=config)

        packed.append(Structure(structure=result["cif_string"], structure_format="cif", source="caliby-pack"))

    return CalibyPackOutput(packed_structures=packed)
