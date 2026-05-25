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
    >>> n_designs = sum(len(b) for b in result.designed_structures)
    >>> print(f"Designed {n_designs} structures across {len(result)} input specs")
"""

import json
import logging
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
            - ``True``: Fix all atoms from input
            - ``False``: Unfix all atoms
            - Contig string: Specify residues to fix (e.g., ``"A1-10,B1-3"``)
            - Dict: Map chain/residue to atom selection using ``"BKBN"`` (backbone),
              ``"TIP"`` (tip atom), ``"ALL"`` (all atoms), or explicit atom names
              (e.g., ``{"A1": "N,CA,C,O,CB", "A2-10": "BKBN"}``)

        select_unfixed_sequence (bool | str | dict[str, str] | None): Residues
            whose sequence can change during design. Accepts InputSelection format:
            - ``True``: All input regions have fixed sequences (upstream default)
            - ``False``: All atoms have unfixed/diffused sequences
            - Contig string: Components to unfix sequence for (e.g., ``"A5-10,B1-3"``)
            Note: Ligands and DNA always have fixed sequences.

        select_hotspots (bool | str | dict[str, str] | None): Atom or residue
            hotspots for binder/PPI design (typically <=4.5 Angstroms to any
            heavy atom in the designed structure).

        symmetry (str | dict[str, Any] | None): Symmetry for homo-oligomer design.
            A group-id string (e.g. ``"C3"``) is wrapped as ``{"id": "C3"}``; a
            full ``SymmetryConfig`` dict is passed through. Pair with
            ``RFdiffusion3Config.sampler_kind="symmetry"``.

        select_buried (bool | str | dict[str, str] | None): RASA selector for
            buried residues.

        select_partially_buried (bool | str | dict[str, str] | None): RASA
            selector for partially buried residues.

        select_exposed (bool | str | dict[str, str] | None): RASA selector for
            solvent-exposed residues.

        select_hbond_donor (dict[str, list[str]] | None): Atom-wise H-bond
            donor flags, e.g. ``{"A40": ["NE2"]}``.

        select_hbond_acceptor (dict[str, list[str]] | None): Atom-wise H-bond
            acceptor flags, e.g. ``{"A45": ["OD1"]}``.

        redesign_motif_sidechains (bool | None): Keep motif backbone fixed,
            redesign side-chains.

        plddt_enhanced (bool | None): Enable pLDDT-based denoising enhancement
            (upstream default ``True``; ``None`` keeps it).

        infer_ori_strategy (Literal['com', 'hotspots'] | None): Origin
            placement — ``com`` (center of mass) or ``hotspots``.

        ori_token (list[float] | None): ``[x, y, z]`` origin override (Angstroms).

        partial_t (float | None): Noise level (in Angstroms) for partial diffusion.
            Lower values preserve more of the input structure. Recommended values
            are 5.0-15.0 Angstroms. Useful for refinement or local redesign tasks.

        is_non_loopy (bool | None): If ``True``/``False``, produces output
            structures with fewer/more loops. ``None`` uses the model's
            native default.

    Note:
        Additional RFdiffusion3 InputSpecification fields can be passed as
        ``**kwargs`` via this model's ``extra="allow"`` mode (e.g.
        ``dialect``, ``cif_parser_args``). See
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    model_config = ConfigDict(extra="allow")

    input_structure: str | None = Field(
        default=None,
        title="Input Structure",
        description="Path to input PDB/CIF file or PDB content string",
    )
    contig: str | None = Field(
        default=None,
        title="Contig",
        description="Contig string specifying design topology (e.g., '50-80,\\0,A1-100')",
        examples=["50-80", "A1-100,50,A150-200"],
    )
    length: str | None = Field(
        default=None,
        title="Length",
        description="Total design length constraint ('min-max' or int as string)",
        examples=["100", "80-120"],
    )
    ligand: str | None = Field(
        default=None,
        title="Ligand",
        description="Ligand selection by residue name (e.g., 'HAX,OAA')",
    )
    unindex: str | dict[str, str] | None = Field(
        default=None,
        title="Unindexed Motifs",
        description="Unindexed motif components for flexible positioning (e.g., 'A244,A274,A320')",
    )
    select_fixed_atoms: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Fixed Atoms",
        description="Atoms to fix in 3D space (True/False, contig string, or dict with BKBN/TIP/ALL)",
    )
    select_unfixed_sequence: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Unfixed Sequence",
        description="Residues whose sequence can change (True/False, contig string, or dict)",
    )
    select_hotspots: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Hotspots",
        description="Atom/residue-level hotspots for binder design (True/False, contig string, or dict)",
        examples=["A24,A35,A50"],
    )
    symmetry: str | dict[str, Any] | None = Field(
        default=None,
        title="Symmetry",
        description=(
            "Symmetry for homo-oligomer design: group-id string (e.g. 'C3') or "
            "SymmetryConfig dict; pair with sampler_kind='symmetry'"
        ),
        examples=["C3", {"id": "C3"}],
    )
    select_buried: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Buried Residues",
        description="RASA selector for residues that should be buried (True/False, contig string, or dict)",
    )
    select_partially_buried: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Partially Buried Residues",
        description="RASA selector for partially buried residues (True/False, contig string, or dict)",
    )
    select_exposed: bool | str | dict[str, str] | None = Field(
        default=None,
        title="Exposed Residues",
        description="RASA selector for solvent-exposed residues (True/False, contig string, or dict)",
    )
    select_hbond_donor: dict[str, list[str]] | None = Field(
        default=None,
        title="H-bond Donor Flags",
        description="Atom-wise hydrogen-bond donor flags, e.g. {'A40': ['NE2']}",
    )
    select_hbond_acceptor: dict[str, list[str]] | None = Field(
        default=None,
        title="H-bond Acceptor Flags",
        description="Atom-wise hydrogen-bond acceptor flags, e.g. {'A45': ['OD1']}",
    )
    redesign_motif_sidechains: bool | None = Field(
        default=None,
        title="Redesign Motif Sidechains",
        description="Keep motif backbone fixed but redesign side-chains",
    )
    plddt_enhanced: bool | None = Field(
        default=None,
        title="pLDDT-Enhanced Denoising",
        description="Enable pLDDT-based denoising enhancement (upstream default: True)",
    )
    infer_ori_strategy: Literal["com", "hotspots"] | None = Field(
        default=None,
        title="Origin Strategy",
        description="Origin placement strategy: 'com' (center of mass) or 'hotspots'",
    )
    ori_token: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        title="Origin (x, y, z)",
        description="Origin override for center-of-mass placement, in Ångstroms (shape [x, y, z])",
        examples=[[0.0, 0.0, 0.0]],
    )
    partial_t: float | None = Field(
        default=None,
        ge=0.0,
        title="Partial Diffusion Noise",
        description="Noise level in Ångstroms for partial diffusion; 5.0-15.0 recommended",
        examples=[5.0, 10.0, 15.0],
    )
    is_non_loopy: bool | None = Field(
        default=None,
        title="Suppress Loops",
        description="When True, produces structures with fewer loops; when False, more loops",
    )

    @field_validator("symmetry", mode="before")
    @classmethod
    def _normalize_symmetry(cls, value: Any) -> Any:
        """Wrap a group-id string (e.g. ``"c3"``) as the ``{"id": "C3"}`` dict rfd3 requires."""
        if isinstance(value, str):
            return {"id": value.strip().upper()}
        return value

    @model_validator(mode="after")
    def validate_has_design_params(self) -> Any:
        """Reject specs with no constraints (any typed field or extras suffices)."""
        typed_design_fields = (
            self.contig,
            self.length,
            self.input_structure,
            self.ligand,
            self.unindex,
            self.select_fixed_atoms,
            self.select_unfixed_sequence,
            self.select_hotspots,
            self.symmetry,
            self.select_buried,
            self.select_partially_buried,
            self.select_exposed,
            self.select_hbond_donor,
            self.select_hbond_acceptor,
            self.redesign_motif_sidechains,
            self.plddt_enhanced,
            self.infer_ori_strategy,
            self.ori_token,
            self.partial_t,
            self.is_non_loopy,
        )
        if not any(f is not None for f in typed_design_fields) and not self.model_extra:
            raise ValueError(
                "At least one design parameter (contig, length, symmetry, "
                "select_*, partial_t, etc.) or **kwargs must be provided"
            )
        return self

    @model_validator(mode="after")
    def validate_selections_require_input_structure(self) -> Any:
        """Reject ``contig``, ``unindex``, and ``select_*`` fields without ``input_structure``."""
        # Upstream rfd3 resolves all of these against an atom array built from
        # input_structure. ``contig`` is not a substitute — even chain-referencing
        # contigs fail without input. ``length`` is the only way to design without
        # an atom source (unconditional generation, no per-residue constraints).
        requires_input = (
            "contig",
            "unindex",
            "select_fixed_atoms",
            "select_unfixed_sequence",
            "select_hotspots",
            "select_buried",
            "select_partially_buried",
            "select_exposed",
            "select_hbond_donor",
            "select_hbond_acceptor",
        )
        offending = [name for name in requires_input if getattr(self, name) is not None]
        if offending and self.input_structure is None:
            raise ValueError(
                f"Fields {offending} require 'input_structure' (PDB/CIF) to resolve "
                "against an atom array. For unconditional design with no per-residue "
                "constraints, use 'length' instead."
            )
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
        if self.symmetry is not None:
            spec["symmetry"] = self.symmetry
        if self.select_buried is not None:
            spec["select_buried"] = self.select_buried
        if self.select_partially_buried is not None:
            spec["select_partially_buried"] = self.select_partially_buried
        if self.select_exposed is not None:
            spec["select_exposed"] = self.select_exposed
        if self.select_hbond_donor is not None:
            spec["select_hbond_donor"] = self.select_hbond_donor
        if self.select_hbond_acceptor is not None:
            spec["select_hbond_acceptor"] = self.select_hbond_acceptor
        if self.redesign_motif_sidechains is not None:
            spec["redesign_motif_sidechains"] = self.redesign_motif_sidechains
        if self.plddt_enhanced is not None:
            spec["plddt_enhanced"] = self.plddt_enhanced
        if self.infer_ori_strategy is not None:
            spec["infer_ori_strategy"] = self.infer_ori_strategy
        if self.ori_token is not None:
            spec["ori_token"] = self.ori_token
        if self.partial_t is not None:
            spec["partial_t"] = self.partial_t
        if self.is_non_loopy is not None:
            spec["is_non_loopy"] = self.is_non_loopy

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
        >>> inputs = RFdiffusion3Input(raw_json='{"spec-1": {"length": "100", "select_buried": "ALA"}}')
    """

    design_specs: list[RFdiffusion3DesignSpec] = InputField(
        default_factory=list,
        title="Design Specs",
        description="List of design specifications",
    )
    raw_json: str | None = InputField(
        default=None,
        title="Raw JSON",
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
        n_batches (int): Independent batches per spec
            (total designs = ``n_batches * diffusion_batch_size * num_specs``).
        diffusion_batch_size (int): Designs sampled in parallel per batch.
        num_timesteps (int): Diffusion timesteps; more = slower, generally higher quality.
        step_scale (float): Step size scale; higher = less diverse, more designable.
        sampler_kind (Literal['default', 'symmetry']): Sampler kind;
            ``'symmetry'`` for homo-oligomer design (paired with ``DesignSpec.symmetry``).
        center_option (Literal['all', 'motif', 'diffuse']): Coordinate-frame
            centering — ``all`` (whole structure), ``motif`` (input motif),
            ``diffuse`` (diffused region only).
        use_classifier_free_guidance (bool): Enable CFG sampling. ``cfg_scale``,
            ``cfg_features``, and ``cfg_t_max`` are no-ops when ``False``.
        cfg_scale (float): CFG scale factor (typical 1.0-3.0).
        cfg_features (list[Literal['active_donor', 'active_acceptor', 'ref_atomwise_rasa']] | None):
            CFG feature names; ``None`` keeps upstream default (all three).
        cfg_t_max (float | None): Diffusion-timestep cap for CFG (0.0-1.0);
            ``None`` keeps upstream default.
        gamma_0 (float): Sampler stochasticity; lower = more designable,
            less diverse; ``0.0`` = deterministic ODE. Must be ``> 0.5``
            when ``sampler_kind="symmetry"``.
        low_memory_mode (bool): Memory-efficient tokenization (slower);
            enable only if GPU RAM is tight.
        dump_trajectories (bool): Save diffusion trajectory frames (debugging).
        align_trajectory_structures (bool): Align trajectory frames across
            timesteps (only when ``dump_trajectories=True``).
        prevalidate_inputs (bool): Fail-fast input JSON validation.
        ckpt_path (str): Checkpoint path or alias (``"rfd3"`` = production preset).
        input_dir (str | None): Local-execution input directory; ``None`` uses a tempdir.
        output_dir (str | None): Local-execution output directory; ``None`` uses a tempdir.
        device (str): ``"cuda"`` or ``"cpu"``.

    Note:
        Additional Hydra ``**kwargs`` pass through ``extra="allow"`` to the
        rfd3 CLI. Sampler sub-keys must use dotted paths
        (e.g. ``inference_sampler.noise_scale=1.003``); flat keys are silently
        ignored. See
        https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md
    """

    model_config = ConfigDict(extra="allow")

    n_batches: int = ConfigField(
        title="Number of Batches",
        default=1,
        ge=1,
        description="Independent batches per spec (total designs per spec = n_batches * diffusion_batch_size)",
    )
    diffusion_batch_size: int = ConfigField(
        title="Diffusion Batch Size",
        default=8,
        ge=1,
        description="Designs sampled in parallel per batch",
    )
    num_timesteps: int = ConfigField(
        title="Number of Timesteps",
        default=200,
        ge=1,
        description="Diffusion timesteps for sampling (more = slower, generally higher quality)",
    )
    step_scale: float = ConfigField(
        title="Step Scale",
        default=1.5,
        ge=0.1,
        description="Diffusion step size scaling; higher = less diverse, more designable (typical: 1.0-2.0)",
    )
    sampler_kind: Literal["default", "symmetry"] = ConfigField(
        title="Sampler Kind",
        default="default",
        description="Inference sampler kind ('symmetry' for homo-oligomer design)",
        reload_on_change=True,
    )
    center_option: Literal["all", "motif", "diffuse"] = ConfigField(
        title="Center Option",
        default="all",
        description="Coordinate-frame centering mode (all/motif/diffuse)",
    )
    use_classifier_free_guidance: bool = ConfigField(
        title="Use Classifier-Free Guidance",
        default=False,
        description="Enable CFG sampling (cfg_scale is a no-op when False)",
    )
    cfg_scale: float = ConfigField(
        title="CFG Scale",
        default=1.5,
        ge=0.0,
        description="CFG scale (typical: 1.0-3.0); requires use_classifier_free_guidance=True",
    )
    cfg_features: list[Literal["active_donor", "active_acceptor", "ref_atomwise_rasa"]] | None = ConfigField(
        title="CFG Features",
        default=None,
        description="CFG steering feature names; None uses upstream default (donor/acceptor/RASA)",
        examples=[["active_donor", "active_acceptor", "ref_atomwise_rasa"]],
    )
    cfg_t_max: float | None = ConfigField(
        title="CFG t_max",
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum diffusion timestep (0.0-1.0) at which CFG is applied",
    )
    gamma_0: float = ConfigField(
        title="Gamma 0",
        default=0.6,
        ge=0.0,
        description=(
            "Sampler stochasticity; lower = more designable, less diverse; "
            "0.0 = deterministic ODE. Must be > 0.5 when sampler_kind='symmetry'."
        ),
    )
    low_memory_mode: bool = ConfigField(
        title="Low Memory Mode",
        default=False,
        description="Memory-efficient tokenization (slower); set True only if GPU RAM is tight",
    )
    dump_trajectories: bool = ConfigField(
        title="Dump Trajectories",
        default=False,
        description="Save diffusion trajectory frames to the output directory",
        include_in_key=False,
    )
    align_trajectory_structures: bool = ConfigField(
        title="Align Trajectory Structures",
        default=False,
        description="Align trajectory structures across timesteps",
        include_in_key=False,
    )
    prevalidate_inputs: bool = ConfigField(
        title="Prevalidate Inputs",
        default=False,
        description="Validate the full input JSON before launching diffusion",
        include_in_key=False,
    )
    ckpt_path: str = ConfigField(
        title="Checkpoint Path",
        default="rfd3",
        description="Checkpoint path or canonical alias (default: rfd3 production preset)",
    )
    input_dir: str | None = ConfigField(
        title="Input Directory",
        default=None,
        description="Optional input directory for local execution inputs",
        include_in_key=False,
    )
    output_dir: str | None = ConfigField(
        title="Output Directory",
        default=None,
        description="Optional output directory for local execution outputs",
        include_in_key=False,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on (e.g., 'cuda', 'cpu')",
        include_in_key=False,
    )

    @model_validator(mode="after")
    def validate_gamma_0_with_symmetry_sampler(self) -> Any:
        """Upstream rfd3 invariant: ``gamma_0 > 0.5`` is required when ``sampler_kind == "symmetry"``."""
        if self.sampler_kind == "symmetry" and self.gamma_0 <= 0.5:
            raise ValueError(f"gamma_0 must be > 0.5 when sampler_kind='symmetry'; got {self.gamma_0}")
        return self

    def get_cli_kwargs(self) -> dict[str, Any]:
        """Build CLI args for the rfd3 inference script.

        Sampler knobs use dotted Hydra paths (``inference_sampler.<key>``).
        On collision, typed fields win over ``model_extra`` — same precedence
        as :meth:`RFdiffusion3DesignSpec.to_dict`.
        """
        cli_kwargs: dict[str, Any] = {}
        # Extras first so typed fields below override them on collision.
        if self.model_extra:
            cli_kwargs.update(self.model_extra)
        cli_kwargs.update(
            {
                "n_batches": self.n_batches,
                "diffusion_batch_size": self.diffusion_batch_size,
                "num_timesteps": self.num_timesteps,
                "step_scale": self.step_scale,
                "inference_sampler.kind": self.sampler_kind,
                "inference_sampler.center_option": self.center_option,
                "inference_sampler.use_classifier_free_guidance": self.use_classifier_free_guidance,
                "inference_sampler.cfg_scale": self.cfg_scale,
                "inference_sampler.gamma_0": self.gamma_0,
                "low_memory_mode": self.low_memory_mode,
                "dump_trajectories": self.dump_trajectories,
                "align_trajectory_structures": self.align_trajectory_structures,
                "prevalidate_inputs": self.prevalidate_inputs,
                "ckpt_path": self.ckpt_path,
            }
        )
        # Forward optional sampler knobs only when explicitly set; otherwise upstream defaults stand.
        if self.cfg_features is not None:
            cli_kwargs["inference_sampler.cfg_features"] = self.cfg_features
        if self.cfg_t_max is not None:
            cli_kwargs["inference_sampler.cfg_t_max"] = self.cfg_t_max
        return cli_kwargs


# Output Data Models


class RFdiffusion3Structure(BaseModel):
    """A single designed structure from RFdiffusion3.

    One per generated design. Multiple ``RFdiffusion3Structure``s are bundled
    inside an ``RFdiffusion3Designs`` (one bundle per input spec). The
    bundle's position in ``RFdiffusion3Output.designed_structures`` matches
    the position of the originating spec in ``RFdiffusion3Input.design_specs``.

    Attributes:
        structure (Structure): The designed 3D structure containing
            atomic coordinates in PDB/CIF format.

        sequence (str): The designed amino acid sequence in single-letter code.
            For multi-chain designs, chains are separated by ``/``.

        metadata (dict[str, Any]): Additional metadata from RFdiffusion3 output,
            including sampled contig, chain info, and any logged metrics.
    """

    structure: Structure = Field(
        title="Designed Structure",
        description="The designed 3D structure",
    )
    sequence: str = Field(
        title="Sequence",
        description="The designed amino acid sequence",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        title="Design Metadata",
        description="Additional metadata from RFdiffusion3 output",
    )


class RFdiffusion3Designs(BaseModel):
    """All designs produced for a single input spec.

    Bundling N designs into one wrapper preserves the framework's 1:1
    cardinality between ``iterable_input_field`` and ``iterable_output_field``
    even though RFdiffusion3 fans out (N = ``n_batches * diffusion_batch_size``
    designs per spec). Mirrors the per-input bundling pattern used by
    ``ProteinMPNNSequences``.

    Attributes:
        spec_key (str): Identifier of the input specification that produced
            these designs (e.g. ``"spec-0"``). For ``raw_json`` callers this
            is the user-supplied key; otherwise it is positional
            (``f"spec-{i}"`` from ``RFdiffusion3Input.to_json_spec``).

        structures (list[RFdiffusion3Structure]): The designs generated for
            this spec, in the order RFdiffusion3 emitted them. List length
            equals ``n_batches * diffusion_batch_size``.
    """

    spec_key: str = Field(
        title="Spec Key",
        description="Identifier of the input spec that produced these designs",
    )
    structures: list[RFdiffusion3Structure] = Field(
        default_factory=list,
        title="Designs",
        description="Designs generated for this spec",
    )

    def __len__(self) -> int:
        """Get the number of designs for this spec."""
        return len(self.structures)

    def __getitem__(self, index: int) -> RFdiffusion3Structure:
        """Get one designed structure within this bundle by index."""
        return self.structures[index]

    def __iter__(self) -> Iterator[RFdiffusion3Structure]:  # type: ignore[override]
        """Iterate over the designs in this bundle."""
        return iter(self.structures)


class RFdiffusion3Output(BaseToolOutput):
    """Output from RFdiffusion3 structure design.

    Designs are grouped into ``RFdiffusion3Designs`` bundles — one per input
    spec — so the field cardinality matches ``RFdiffusion3Input.design_specs``
    1:1, consistent with the framework's iterable-tool contract.

    Attributes:
        designed_structures (list[RFdiffusion3Designs]): One bundle per input
            spec. Total design count is
            ``len(design_specs) * n_batches * diffusion_batch_size``.

    Note:
        This class supports list-like operations that iterate over the
        per-spec bundles: ``output[i]`` returns the i-th bundle,
        ``for designs in output`` walks bundles, ``len(output)`` is the
        number of bundles. To walk every individual design, use
        ``for bundle in output: for s in bundle: ...``.
    """

    designed_structures: list[RFdiffusion3Designs] = Field(
        default_factory=list,
        title="Designed Structures",
        description="Per-spec bundles of designed structures",
    )

    def __len__(self) -> int:
        """Get the number of per-spec design bundles."""
        return len(self.designed_structures)

    def __getitem__(self, index: int) -> RFdiffusion3Designs:
        """Get the design bundle for the i-th input spec."""
        return self.designed_structures[index]

    def __iter__(self) -> Iterator[RFdiffusion3Designs]:  # type: ignore[override]
        """Iterate over per-spec design bundles."""
        return iter(self.designed_structures)

    def __repr__(self) -> str:
        """Get string representation."""
        n_bundles = len(self.designed_structures)
        n_designs = sum(len(b) for b in self.designed_structures)
        return f"RFdiffusion3Output(designed_structures=[{n_bundles} bundles, {n_designs} designs])"

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
        for bundle in self.designed_structures:
            for i, struct in enumerate(bundle.structures):
                out_file = path / f"{bundle.spec_key}_design_{i}.{file_format}"
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
    iterable_output_field="designed_structures",
    cacheable=True,
    stochastic=True,
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
            - ``designed_structures``: One ``RFdiffusion3Designs`` bundle per
              input spec, in input order. Each bundle has a ``spec_key`` and
              a list of ``RFdiffusion3Structure``s (one per generated design).
            - Each ``RFdiffusion3Structure`` includes:
                - 3D coordinates (as Structure)
                - Designed sequence
                - Metadata (sampled contig, chain info, etc.)

    See Also:
        - RFdiffusion3 GitHub: https://github.com/RosettaCommons/foundry
        - RFdiffusion3 paper: https://doi.org/10.1101/2025.09.18.676967

    Example:
        >>> # Unconditional design of 100-residue protein, 4 designs in one batch
        >>> inputs = RFdiffusion3Input(design_specs=[RFdiffusion3DesignSpec(length="100")])
        >>> config = RFdiffusion3Config(diffusion_batch_size=4, num_timesteps=200, verbose=True)
        >>> result = run_rfdiffusion3(inputs, config)
        >>> bundle = result[0]  # one bundle per input spec
        >>> print(f"Spec {bundle.spec_key}: {len(bundle)} designs")
        >>> print(f"First sequence: {bundle[0].sequence}")

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
            "operation": "design",
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

    # Group designs by originating spec so the output has 1:1 cardinality with
    # inputs.design_specs. Subprocess emits a flat list; the spec-key prefix on
    # each design (set in to_json_spec at line 380-384) lets us bucket them.
    buckets: dict[str, list[RFdiffusion3Structure]] = defaultdict(list)
    for design_data in output_data.get("designs", []):
        structure = Structure(
            structure=design_data["structure_content"],
            source="rfdiffusion3-design",
        )
        buckets[design_data["spec_key"]].append(
            RFdiffusion3Structure(
                structure=structure,
                sequence=design_data["sequence"],
                metadata=design_data.get("metadata", {}),
            )
        )

    # Preserve input order. to_json_spec assigns spec-0, spec-1, ... by
    # enumeration, so re-emit bundles in that order.
    designed_structures = [
        RFdiffusion3Designs(spec_key=key, structures=buckets[key]) for key in sorted(buckets, key=_spec_key_sort_index)
    ]
    return RFdiffusion3Output(designed_structures=designed_structures)


def _spec_key_sort_index(key: str) -> tuple[int, str]:
    """Sort spec keys by trailing integer when present, falling back to lexical.

    Positional inputs produce ``spec-0``, ``spec-1``, ... — sort those
    numerically. ``raw_json`` callers may use arbitrary key names; those fall
    through to lexical ordering.
    """
    suffix = key.rsplit("-", 1)[-1] if "-" in key else key
    try:
        return (0, f"{int(suffix):020d}")
    except ValueError:
        return (1, key)
