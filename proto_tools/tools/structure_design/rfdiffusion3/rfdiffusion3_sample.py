"""proto_tools/tools/structure_design/rfdiffusion3/rfdiffusion3_sample.py.

Example:
    >>> from proto_tools.tools.structure_design.rfdiffusion3 import (
    ...     run_rfdiffusion3,
    ...     RFdiffusion3Input,
    ...     RFdiffusion3Config,
    ...     RFdiffusion3DesignSpec,
    ... )
    >>> # Simple unconditional design
    >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])
    >>> config = RFdiffusion3Config(diffusion_batch_size=4)
    >>> result = run_rfdiffusion3(inputs, config)
    >>> print(f"Designed {len(result.output_structures)} structures")
"""

import json
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================
# Input Data Models


class RFdiffusion3DesignSpec(BaseModel):
    r"""Single design specification for RFdiffusion3.

    This class represents a single design task with its constraints. Multiple
    design specs can be provided to generate designs under different conditions.

    For the full list of available InputSpecification fields, see:
    https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md

    Attributes:
        input_structure (str | None): Path to input PDB/CIF file or PDB content string.
            Required for motif scaffolding, binder design, or any task that needs
            structural context. Can be omitted for unconditional de novo design.

        contig (str | None): Contig string specifying the design topology.
            Format: comma-separated segments with chain breaks as ``\\0``.

    Examples:
                - ``"50-80"`` - design 50-80 residue monomer
                - ``"A1-100,50,A150-200"`` - scaffold around residues A1-100 and A150-200
                - ``"50,\\0,B1-50"`` - design 50 residues, chain break, then keep B1-50

        length (str | None): Total design length constraint. Can be an integer
            (exact length) or range ``"min-max"``. Used for unconditional design
            when no contig is specified.

        ligand (str | None): Ligand selection by residue name from input structure.
            Comma-separated list of 3-letter codes (e.g., ``"HAX,OAA"``).

        unindex (str | dict[str, str] | None): Unindexed motif components whose
            sequence position is unknown to the model. Useful for active site scaffolding
            where catalytic residues should be placed but their position in the final
            sequence is flexible. Components must not overlap with ``contig``.
            Example: ``"A244,A274,A320"`` lists multiple unindexed components.

        select_fixed_atoms (bool | str | dict[str, str] | None): Atoms to fix
            in 3D space during diffusion. Accepts InputSelection format:
            - ``True``: Fix all atoms from input (default when input provided)
            - ``False``: Unfix all atoms
            - Contig string: Specify residues to fix (e.g., ``"A1-10,B1-3"``)
            - Dict: Map chain/residue to atom selection using ``"BKBN"`` (backbone),
              ``"TIP"`` (tip atom), ``"ALL"`` (all atoms), or explicit atom names
              (e.g., ``{"A1": "N,CA,C,O,CB", "A2-10": "BKBN"}``)

        select_unfixed_sequence (bool | str | dict[str, str] | None): Residues
            whose sequence can change during design. Accepts InputSelection format:
            - ``True``: All input regions have fixed sequences (default)
            - ``False``: All atoms have unfixed/diffused sequences
            - Contig string: Components to unfix sequence for (e.g., ``"A5-10,B1-3"``)
            Note: Ligands and DNA always have fixed sequences.

        select_hotspots (str | dict[str, str] | None): Atom-level or residue-level
            hotspots for protein-protein interaction design. Hotspots will typically be
            at most 4.5 Angstroms to any heavy atom in the designed structure. Typically
            used for designing binders.

        partial_t (float | None): Noise level (in Angstroms) for partial diffusion.
            Lower values preserve more of the input structure. Recommended values
            are 5.0-15.0 Angstroms. Useful for refinement or local redesign tasks.

    Note:
        Additional RFdiffusion3 InputSpecification fields can be passed as
        ``**kwargs``. Examples: ``symmetry``, ``select_buried``,
        ``select_hbond_donor``. See
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    model_config = ConfigDict(extra="allow")

    input_structure: str | None = Field(
        default=None,
        description="Path to input PDB/CIF file or PDB content string",
    )
    contig: str | None = Field(
        default=None,
        description="Contig string specifying design topology (e.g., '50-80,\\0,A1-100')",
    )
    length: str | None = Field(
        default=None,
        description="Total design length constraint ('min-max' or int)",
    )
    ligand: str | None = Field(
        default=None,
        description="Ligand selection by residue name (e.g., 'HAX,OAA')",
    )
    unindex: str | dict[str, str] | None = Field(
        default=None,
        description="Unindexed motif components for flexible positioning (e.g., 'A244,A274,A320')",
    )
    select_fixed_atoms: bool | str | dict[str, str] | None = Field(
        default=None,
        description="Atoms to fix in 3D space (True/False, contig string, or dict with BKBN/TIP/ALL)",
    )
    select_unfixed_sequence: bool | str | dict[str, str] | None = Field(
        default=None,
        description="Residues whose sequence can change (True/False, contig string, or dict)",
    )
    select_hotspots: str | dict[str, str] | None = Field(
        default=None,
        description="Atom/residue-level hotspots for binder design (contig string or dict)",
    )
    partial_t: float | None = Field(
        default=None,
        ge=0.0,
        description="Noise level (Angstroms) for partial diffusion (5.0-15.0 recommended)",
    )

    @model_validator(mode="after")
    def validate_has_design_params(self) -> Any:
        """Validate that at least one design parameter is provided."""
        has_params = any(
            [
                self.contig is not None,
                self.length is not None,
                self.model_extra,
            ]
        )
        if not has_params:
            raise ValueError("At least one of 'contig', 'length', or other design parameters must be provided")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert design spec to RFdiffusion3 JSON format.

        Returns a dictionary compatible with RFdiffusion3's InputSpecification format.
        See https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
        """
        spec: dict[str, Any] = {}

        # Start with extra kwargs (allows advanced options to be passed through)
        if self.model_extra:
            spec.update(self.model_extra)

        # Add typed fields (these override extra kwargs if both provided)
        if self.input_structure is not None:
            spec["input"] = self.input_structure
        if self.contig is not None:
            spec["contig"] = self.contig
        if self.length is not None:
            spec["length"] = self.length
        if self.ligand is not None:
            spec["ligand"] = self.ligand
        if self.unindex is not None:
            spec["unindex"] = self.unindex
        if self.select_fixed_atoms is not None:
            spec["select_fixed_atoms"] = self.select_fixed_atoms
        if self.select_unfixed_sequence is not None:
            spec["select_unfixed_sequence"] = self.select_unfixed_sequence
        if self.select_hotspots is not None:
            spec["select_hotspots"] = self.select_hotspots
        if self.partial_t is not None:
            spec["partial_t"] = self.partial_t

        return spec


class RFdiffusion3Input(BaseToolInput):
    """Input for RFdiffusion3 structure design.

    This class provides a flexible interface for specifying one or more design
    tasks for RFdiffusion3. Supports both structured Pydantic models for common
    use cases and raw JSON passthrough for advanced users.

    Attributes:
        design_specs (list[RFdiffusion3DesignSpec]): List of design specifications. Each
            spec represents an independent design task with its own constraints.
            Multiple specs will be processed in a single run.

        raw_json (str | None): Raw JSON string for advanced users who need
            full RFdiffusion3 flexibility. If provided, ``design_specs`` will be ignored
            and this JSON will be passed directly to RFdiffusion3.

    Examples:
        >>> # Simple unconditional design
        >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100-150")])
        >>>
        >>> # Motif scaffolding
        >>> inputs = RFdiffusion3Input(
        ...     design_specs=[
        ...         RFdiffusion3DesignSpec(
        ...             input_structure="path/to/motif.pdb",
        ...             contig="50-80,A10-25,30-50",
        ...         )
        ...     ]
        ... )
        >>>
        >>> # Raw JSON for advanced use
        >>> inputs = RFdiffusion3Input(raw_json='{"spec-1": {"length": "100", "is_non_loopy": true}}')
    """

    design_specs: list[RFdiffusion3DesignSpec] = InputField(
        default_factory=list,
        description="List of design specifications",
    )
    raw_json: str | None = InputField(
        default=None,
        description="Raw JSON string for advanced RFdiffusion3 configuration",
    )

    @model_validator(mode="after")
    def validate_has_input(self) -> Any:
        """Validate that either design_specs or raw_json is provided."""
        if not self.design_specs and not self.raw_json:
            raise ValueError("Either 'design_specs' (non-empty) or 'raw_json' must be provided")
        return self

    def to_json_spec(self) -> str:
        """Convert input to RFdiffusion3 JSON specification format."""
        if self.raw_json:
            return self.raw_json

        spec_dict: dict[str, Any] = {}
        for i, spec in enumerate(self.design_specs):
            spec_dict[f"spec-{i}"] = spec.to_dict()

        return json.dumps(spec_dict, indent=2)

    def __len__(self) -> int:
        """Get the number of design specs."""
        if self.raw_json:
            try:
                parsed = json.loads(self.raw_json)
                return len(parsed)
            except json.JSONDecodeError:
                return 1
        return len(self.design_specs)


# Config Data Models


class RFdiffusion3Config(BaseConfig):
    """Configuration for RFdiffusion3 structure design.

    This class defines configuration parameters for running RFdiffusion3,
    controlling the number of designs generated, diffusion parameters,
    and execution options.

    Attributes:
        n_batches (int): number of batches to generate per input key (default: 1).

        diffusion_batch_size (int): number of diffusion samples (designs) per batch (default: 8).

        num_timesteps (int): diffusion timesteps for sampling (default: 200).

        step_scale (float): scales diffusion step size; higher -> less diverse,
            more designable (default: 1.5).

        low_memory_mode (bool): memory-efficient tokenization mode; set True if
            GPU RAM is tight (default: False).

        ckpt_path (str): String containing the path and file name of the checkpoint
            path you want to use (default: rfd3).

        input_dir (str | None): Optional directory containing input files for
            local execution. If not set, input files are written to a temporary directory.

        output_dir (str | None): Optional directory to write local output files.
            If not set, local execution uses a temporary directory.

        device (str): Device to run the model on (e.g., ``"cuda"``, ``"cpu"``).
            Default: ``"cuda"``.

    Note:
        Additional CLI ``**kwargs`` are passed directly to RFdiffusion3. See
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md

        Total number of designs = n_batches * diffusion_batch_size * num_specs.
        Memory usage scales with diffusion_batch_size and protein length.
    """

    model_config = ConfigDict(extra="allow")

    n_batches: int = ConfigField(
        title="Number of Batches",
        default=1,
        ge=1,
        description="number of batches to generate per input key (default: 1).",
    )
    diffusion_batch_size: int = ConfigField(
        title="Diffusion Batch Size",
        default=8,
        ge=1,
        description="number of diffusion samples (designs) per batch (default: 8).",
        advanced=True,
    )
    num_timesteps: int = ConfigField(
        title="Number of Timesteps",
        default=200,
        ge=1,
        description="diffusion timesteps for sampling (default: 200).",
        advanced=True,
    )
    step_scale: float = ConfigField(
        title="Step Scale",
        default=1.5,
        ge=0.1,
        description="scales diffusion step size; higher -> less diverse, more designable (default: 1.5).",
        advanced=True,
    )
    low_memory_mode: bool = ConfigField(
        title="Low Memory Mode",
        default=False,
        description="memory-efficient tokenization mode; set True if GPU RAM is tight (default: False).",
        advanced=True,
    )
    ckpt_path: str = ConfigField(
        title="Checkpoint Path",
        default="rfd3",
        description="String containing the path and file name of the checkpoint path you want to use (default: rfd3).",
        hidden=True,
    )
    input_dir: str | None = ConfigField(
        title="Input Directory",
        default=None,
        description="Optional input directory for local execution inputs",
        hidden=True,
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional output directory for local execution outputs",
        hidden=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        hidden=True,
        include_in_key=False,
    )

    def get_cli_kwargs(self) -> dict[str, Any]:
        """Get all CLI arguments to pass to the inference script."""
        cli_kwargs = {
            "n_batches": self.n_batches,
            "diffusion_batch_size": self.diffusion_batch_size,
            "num_timesteps": self.num_timesteps,
            "step_scale": self.step_scale,
            "low_memory_mode": self.low_memory_mode,
            "ckpt_path": self.ckpt_path,
        }
        if self.model_extra:
            cli_kwargs.update(self.model_extra)
        return cli_kwargs


# Output Data Models


class RFdiffusion3Structure(BaseModel):
    """Represents a single designed structure from RFdiffusion3.

    Attributes:
        structure (Structure): The designed 3D structure containing
            atomic coordinates in PDB/CIF format.

        sequence (str): The designed amino acid sequence in single-letter code.
            For multi-chain designs, chains are separated by ``/``.

        spec_key (str): Identifier of the input specification that produced
            this design (e.g., ``"spec-0"``).

        design_index (int): Index of this design within its batch.
            Combined with spec_key uniquely identifies the design.

        metadata (dict[str, Any]): Additional metadata from RFdiffusion3 output,
            including sampled contig, chain info, and any logged metrics.
    """

    structure: Structure = Field(description="The designed 3D structure")
    sequence: str = Field(description="The designed amino acid sequence")
    spec_key: str = Field(description="Identifier of the input spec that produced this design")
    design_index: int = Field(ge=0, description="Index of this design within its batch")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from RFdiffusion3 output",
    )


class RFdiffusion3Output(BaseToolOutput):
    """Output from RFdiffusion3 structure design.

    This class encapsulates the results of RFdiffusion3 structure design, containing
    one or more designed structures with their sequences and metadata.

    Attributes:
        output_structures (list[RFdiffusion3Structure]): List of designed
            structures. Each structure includes 3D coordinates, sequence,
            and metadata. The number of structures depends on the configuration
            (n_batches * diffusion_batch_size * num_specs).

    Note:
        This class supports list-like operations for convenient access:
        indexing (``output[0]``), iteration (``for struct in output``),
        and length (``len(output)``).
    """

    output_structures: list[RFdiffusion3Structure] = Field(
        default_factory=list,
        description="List of designed structures",
    )

    def __len__(self) -> int:
        """Get the number of designed structures."""
        return len(self.output_structures)

    def __getitem__(self, index: int) -> RFdiffusion3Structure:
        """Get a designed structure by index."""
        return self.output_structures[index]

    def __iter__(self) -> Iterator[RFdiffusion3Structure]:  # type: ignore[override]
        """Iterate over designed structures."""
        return iter(self.output_structures)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"RFdiffusion3Output(output_structures=[{len(self.output_structures)} structures])"

    def __str__(self) -> str:
        """Get human-readable string representation."""
        return self.__repr__()

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["pdb", "cif"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "pdb"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        if file_format not in ["pdb", "cif"]:
            raise ValueError(f"Unsupported format: {file_format}")

        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)
        for i, struct in enumerate(self.output_structures):
            out_file = path / f"rfdiffusion3_design_{i}.{file_format}"
            if file_format == "pdb":
                struct.structure.write_pdb(out_file)
            elif file_format == "cif":
                struct.structure.write_cif(out_file)


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])


@tool(
    key="rfdiffusion3-design",
    label="RFdiffusion3 Structure Design",
    category="structure_design",
    input_class=RFdiffusion3Input,
    config_class=RFdiffusion3Config,
    output_class=RFdiffusion3Output,
    description="De novo protein structure design using RFdiffusion3",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_field="design_specs",
    iterable_output_field="output_structures",
    cacheable=True,
)
def run_rfdiffusion3(inputs: RFdiffusion3Input, config: RFdiffusion3Config, instance: Any = None) -> RFdiffusion3Output:
    """Design protein structures using RFdiffusion3.

    Uses RFdiffusion3, a diffusion-based generative model, to design novel
    protein structures under specified constraints. Unlike structure prediction,
    RFdiffusion3 generates both structure AND sequence, making it suitable for:

    - De novo protein design (unconditional generation)
    - Motif scaffolding (design around fixed structural motifs)
    - Protein binder design (design proteins that bind to targets)
    - Enzyme design (scaffold around catalytic sites)
    - Symmetric protein design (design homo-oligomers)

    Runs via local GPU execution in isolated Python environments.

    Args:
        inputs (RFdiffusion3Input): Validated input containing one or more design
            specifications. Each spec defines constraints for a design task.
        config (RFdiffusion3Config): Validated configuration specifying diffusion
            parameters and execution options.

        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        RFdiffusion3Output: Structured output containing:
            - ``output_structures``: List of ``RFdiffusion3Structure`` instances
            - Each structure includes:
                - 3D coordinates (as Structure)
                - Designed sequence
                - Metadata (sampled contig, chain info, etc.)

    See Also:
        - RFdiffusion3 GitHub: https://github.com/RosettaCommons/foundry
        - RFdiffusion3 paper: https://doi.org/10.1101/2025.09.18.676967

    Example:
        >>> # Unconditional design of 100-residue protein
        >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])
        >>> config = RFdiffusion3Config(diffusion_batch_size=4, num_timesteps=200, verbose=True)
        >>> result = run_rfdiffusion3(inputs, config)
        >>> print(f"Designed {len(result)} structures")
        >>> print(f"First sequence: {result[0].sequence}")

    Note:
        - RFdiffusion3 generates both structure AND sequence
        - Memory usage scales with batch size and protein length
        - Dependency isolation is handled automatically via venv subprocess
    """
    json_spec = inputs.to_json_spec()

    logger.debug("Using local GPU for RFdiffusion3 structure design...")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        input_dir = Path(config.input_dir) if config.input_dir else temp_dir
        output_dir = Path(config.output_dir) if config.output_dir else temp_dir / "rfdiffusion3_output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_json_path = input_dir / "rfdiffusion3_input.json"

        with open(input_json_path, "w") as f:
            f.write(json_spec)

        input_data = {
            "input_json_path": str(input_json_path),
            "output_dir": str(output_dir),
            "device": config.device,
            "verbose": config.verbose,
            "seed": config.seed,
            **config.get_cli_kwargs(),
        }
        output_data = ToolInstance.dispatch(
            "rfdiffusion3",
            input_data,
            instance=instance,
            config=config,
        )

    output_structures = []
    for design_data in output_data.get("designs", []):
        structure = Structure(
            structure=design_data["structure_content"],
            source="rfdiffusion3-design",
        )
        output_structures.append(
            RFdiffusion3Structure(
                structure=structure,
                sequence=design_data["sequence"],
                spec_key=design_data["spec_key"],
                design_index=design_data["design_index"],
                metadata=design_data.get("metadata", {}),
            )
        )

    return RFdiffusion3Output(output_structures=output_structures)
