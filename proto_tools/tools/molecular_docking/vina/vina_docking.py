"""AutoDock Vina rigid-receptor molecular docking tool."""

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

Vector3 = tuple[float, float, float]
_SIGNED_INT32_MAX = 2**31 - 1
_MAX_SEARCH_BOX_AXIS = 100.0
_MIN_GRID_SPACING = 0.1
_MAX_GRID_POINTS = 2_000_000


class VinaSearchBox(BaseModel):
    """Explicit AutoDock Vina search-box coordinates.

    Attributes:
        center (Vector3): Box center ``(x, y, z)`` in angstroms.
        size (Vector3): Positive box dimensions ``(x, y, z)`` in angstroms.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["coordinates"] = Field(
        default="coordinates",
        title="Box Mode",
        description="Use an explicitly positioned search box.",
    )
    center: Vector3 = Field(
        title="Box Center",
        description="Search-box center (x, y, z) in angstroms.",
    )
    size: Vector3 = Field(
        title="Box Size",
        description="Search-box dimensions (x, y, z) in angstroms.",
    )

    @field_validator("center")
    @classmethod
    def _validate_center(cls, value: Vector3) -> Vector3:
        """Reject non-finite coordinates before dispatch."""
        if not all(math.isfinite(component) for component in value):
            raise ValueError("search-box center coordinates must be finite")
        return value

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: Vector3) -> Vector3:
        """Require finite, positive dimensions on every axis."""
        if not all(math.isfinite(component) and component > 0 for component in value):
            raise ValueError("search-box dimensions must be finite and greater than zero")
        if any(component > _MAX_SEARCH_BOX_AXIS for component in value):
            raise ValueError(f"search-box dimensions must not exceed {_MAX_SEARCH_BOX_AXIS:g} angstroms per axis")
        return value


class VinaReferenceLigandBox(BaseModel):
    """Search box derived from a coordinate-bearing reference ligand.

    Attributes:
        reference_ligand (Structure): Ligand coordinates in the receptor coordinate frame.
        padding (float): Padding added to both sides of each ligand bounding-box axis.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["reference_ligand"] = Field(
        default="reference_ligand",
        title="Box Mode",
        description="Center and size the search box around reference-ligand coordinates.",
    )
    reference_ligand: Structure = Field(
        title="Reference Ligand",
        description="Coordinate-bearing ligand structure aligned to the receptor.",
    )
    padding: float = Field(
        default=4.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Box Padding",
        description="Padding added on each side of the ligand bounding box in angstroms.",
    )

    def resolve(self) -> VinaSearchBox:
        """Resolve the reference-ligand bounds into explicit Vina coordinates."""
        structure = self.reference_ligand.gemmi_struct
        if len(structure) == 0:
            raise ValueError("reference_ligand contains no coordinate models")

        coordinates = [
            (atom.pos.x, atom.pos.y, atom.pos.z) for chain in structure[0] for residue in chain for atom in residue
        ]
        if not coordinates:
            raise ValueError("reference_ligand contains no atoms")
        if not all(math.isfinite(component) for xyz in coordinates for component in xyz):
            raise ValueError("reference_ligand contains non-finite coordinates")

        minima = tuple(min(xyz[axis] for xyz in coordinates) for axis in range(3))
        maxima = tuple(max(xyz[axis] for xyz in coordinates) for axis in range(3))
        center = (
            (minima[0] + maxima[0]) / 2.0,
            (minima[1] + maxima[1]) / 2.0,
            (minima[2] + maxima[2]) / 2.0,
        )
        size = (
            (maxima[0] - minima[0]) + 2.0 * self.padding,
            (maxima[1] - minima[1]) + 2.0 * self.padding,
            (maxima[2] - minima[2]) + 2.0 * self.padding,
        )
        return VinaSearchBox(center=center, size=size)


def _validated_grid_point_count(search_box: VinaSearchBox, spacing: float) -> int:
    """Return the Vina grid size after enforcing a practical allocation limit."""
    dimensions = tuple(math.ceil(axis_size / spacing) + 1 for axis_size in search_box.size)
    grid_points = math.prod(dimensions)
    if grid_points > _MAX_GRID_POINTS:
        raise ValueError(
            f"search box and grid_spacing require {grid_points:,} grid points "
            f"({dimensions[0]} x {dimensions[1]} x {dimensions[2]}); "
            f"the maximum is {_MAX_GRID_POINTS:,}"
        )
    return grid_points


class VinaDockingInput(BaseToolInput):
    """Inputs for rigid-receptor docking with AutoDock Vina.

    Attributes:
        receptor (Structure): Receptor coordinates to parameterize as a rigid PDBQT model.
        ligand (Fragment): Single small molecule to dock; a SMILES string is accepted directly.
        search_box (VinaSearchBox | VinaReferenceLigandBox): Explicit search box or a box
            derived from a reference ligand in the receptor coordinate frame.
    """

    receptor: Structure = InputField(
        title="Receptor",
        description="Rigid receptor structure in PDB or mmCIF format.",
    )
    ligand: Fragment = InputField(
        title="Ligand",
        description="Single ligand Fragment or SMILES string to dock.",
    )
    search_box: VinaSearchBox | VinaReferenceLigandBox = InputField(
        title="Search Box",
        description="Explicit search-box coordinates or a coordinate-bearing reference ligand.",
    )

    @field_validator("ligand", mode="before")
    @classmethod
    def _coerce_smiles(cls, value: Any) -> Any:
        """Accept a bare SMILES string as shorthand for ``Fragment``."""
        if isinstance(value, str):
            return {"smiles": value}
        if isinstance(value, dict):
            return value.copy()
        return value

    @model_validator(mode="after")
    def _validate_resolved_box(self) -> "VinaDockingInput":
        """Resolve reference coordinates during validation so malformed boxes fail early."""
        self.resolved_search_box()
        return self

    def resolved_search_box(self) -> VinaSearchBox:
        """Return explicit coordinates for either supported search-box input."""
        if isinstance(self.search_box, VinaReferenceLigandBox):
            return self.search_box.resolve()
        return self.search_box


class VinaDockingPoseMetrics(Metrics):
    """AutoDock Vina score and distance bounds for one ranked pose."""

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "affinity": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "unit": "kcal/mol",
            "better_values_are": "lower",
        },
        "rmsd_lower_bound": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "angstrom",
            "better_values_are": "context-dependent",
        },
        "rmsd_upper_bound": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
            "unit": "angstrom",
            "better_values_are": "context-dependent",
        },
    }
    primary_metric: str | None = Field(
        default="affinity",
        title="Primary Metric",
        description="Headline metric used to rank docking poses.",
    )


class VinaDockingPose(BaseModel):
    """One ranked docking pose with metrics and molecular-file representations."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, title="Pose Rank", description="One-based Vina pose rank.")
    metrics: VinaDockingPoseMetrics = Field(
        title="Pose Metrics",
        description="Affinity and RMSD-from-best-mode bounds reported by Vina.",
    )
    sdf: str = Field(
        min_length=1,
        title="SDF Pose",
        description="This pose as one SDF record with preserved bond orders.",
    )
    pdbqt: str = Field(
        min_length=1,
        title="PDBQT Pose",
        description="This pose as one Vina PDBQT MODEL block.",
    )


class VinaDockingOutput(BaseToolOutput):
    """Ranked AutoDock Vina poses and reproducibility metadata.

    Attributes:
        poses (list[VinaDockingPose]): Ranked poses in ascending affinity order.
        seed (int): Concrete random seed used by Vina and ligand conformer generation.
        search_box (VinaSearchBox): Resolved search-box coordinates.
        scoring_function (Literal["vina", "vinardo"]): Scoring function used.
        poses_sdf (str): All returned poses in a combined SDF payload.
        poses_pdbqt (str): All returned poses in a combined PDBQT payload.
    """

    poses: list[VinaDockingPose] = Field(
        title="Docking Poses",
        description="Ranked docking poses with affinity and RMSD metrics.",
    )
    seed: int = Field(
        ge=1,
        lt=2**31,
        title="Random Seed",
        description="Concrete random seed used for conformer generation and docking.",
    )
    search_box: VinaSearchBox = Field(
        title="Resolved Search Box",
        description="Explicit center and dimensions used to compute Vina affinity maps.",
    )
    scoring_function: Literal["vina", "vinardo"] = Field(
        title="Scoring Function",
        description="Vina-family scoring function used for docking.",
    )
    poses_sdf: str = Field(
        min_length=1,
        title="Combined SDF",
        description="All returned docking poses as SDF records.",
    )
    poses_pdbqt: str = Field(
        min_length=1,
        title="Combined PDBQT",
        description="All returned docking poses as PDBQT MODEL blocks.",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported docking export formats."""
        return ["sdf", "pdbqt", "csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Use SDF as the default interoperable molecular format."""
        return "sdf"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "sdf":
            path.write_text(self.poses_sdf)
        elif file_format == "pdbqt":
            path.write_text(self.poses_pdbqt)
        elif file_format == "csv":
            center = self.search_box.center
            size = self.search_box.size
            fieldnames = [
                "rank",
                "affinity",
                "rmsd_lower_bound",
                "rmsd_upper_bound",
                "seed",
                "scoring_function",
                "center_x",
                "center_y",
                "center_z",
                "size_x",
                "size_y",
                "size_z",
            ]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for pose in self.poses:
                    writer.writerow(
                        {
                            "rank": pose.rank,
                            **dict(pose.metrics.items()),
                            "seed": self.seed,
                            "scoring_function": self.scoring_function,
                            "center_x": center[0],
                            "center_y": center[1],
                            "center_z": center[2],
                            "size_x": size[0],
                            "size_y": size[1],
                            "size_z": size[2],
                        }
                    )
        elif file_format == "json":
            path.write_text(json.dumps(self.model_dump(mode="json"), indent=2))
        else:
            raise ValueError(f"Unsupported format: {file_format}")


class VinaDockingConfig(BaseConfig):
    """Configuration for AutoDock Vina docking.

    Attributes:
        scoring_function (Literal["vina", "vinardo"]): Vina-family scoring function.
        exhaustiveness (int): Number of independent Monte Carlo search runs.
        num_poses (int): Maximum poses retained and returned.
        energy_range (float): Maximum affinity difference from the best returned pose.
        min_rmsd (float): Minimum RMSD separation between retained poses.
        max_evaluations (int): Search evaluation cap; zero uses Vina's heuristic.
        cpu (int): Worker threads; zero lets Vina use all available CPUs.
        grid_spacing (float): Affinity-map grid spacing in angstroms.
        allow_bad_residues (bool): Delete receptor residues Meeko cannot parameterize.
        seed (int | None): Positive signed 32-bit seed; generated and returned when omitted.
    """

    scoring_function: Literal["vina", "vinardo"] = ConfigField(
        default="vina",
        title="Scoring Function",
        description="Scoring function: Vina or Vinardo.",
    )
    exhaustiveness: int = ConfigField(
        default=8,
        ge=1,
        le=_SIGNED_INT32_MAX,
        title="Exhaustiveness",
        description="Independent Monte Carlo runs; higher values search more thoroughly.",
    )
    num_poses: int = ConfigField(
        default=9,
        ge=1,
        le=_SIGNED_INT32_MAX,
        title="Number of Poses",
        description="Maximum number of ranked poses to retain and return.",
    )
    energy_range: float = ConfigField(
        default=3.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Energy Range",
        description="Maximum affinity difference from the best returned pose in kcal/mol.",
    )
    min_rmsd: float = ConfigField(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Minimum RMSD",
        description="Minimum RMSD separation between retained poses in angstroms.",
    )
    max_evaluations: int = ConfigField(
        default=0,
        ge=0,
        le=_SIGNED_INT32_MAX,
        title="Maximum Evaluations",
        description="Search evaluation cap; zero uses Vina's heuristic.",
    )
    cpu: int = ConfigField(
        default=0,
        ge=0,
        le=_SIGNED_INT32_MAX,
        title="CPU Threads",
        description="CPU threads used by Vina; zero uses all available CPUs.",
        include_in_key=False,
    )
    grid_spacing: float = ConfigField(
        default=0.375,
        ge=_MIN_GRID_SPACING,
        allow_inf_nan=False,
        title="Grid Spacing",
        description=f"Affinity-map grid spacing in angstroms; minimum {_MIN_GRID_SPACING:g}.",
    )
    allow_bad_residues: bool = ConfigField(
        default=False,
        title="Allow Bad Residues",
        description="Delete receptor residues that Meeko cannot match to chemical templates.",
    )
    seed: int | None = ConfigField(
        default=None,
        ge=1,
        lt=2**31,
        title="Random Seed",
        description="Positive signed 32-bit seed; generated and returned when omitted.",
    )

    @classmethod
    def minimal(cls, **kwargs: Any) -> "VinaDockingConfig":
        """Return a low-cost configuration for smoke and infrastructure tests."""
        kwargs.setdefault("exhaustiveness", 1)
        kwargs.setdefault("num_poses", 1)
        kwargs.setdefault("cpu", 1)
        return cls(**kwargs)


def example_input() -> VinaDockingInput:
    """Return the bundled c-Abl/imatinib redocking example."""
    toolkit_dir = Path(__file__).parent
    return VinaDockingInput(
        receptor=toolkit_dir / "example_receptor_1iep.pdb",  # type: ignore[arg-type]
        ligand="Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # type: ignore[arg-type]
        search_box=VinaSearchBox(center=(15.190, 53.903, 16.917), size=(20.0, 20.0, 20.0)),
    )


@tool(
    key="vina-docking",
    label="AutoDock Vina Docking",
    category="molecular_docking",
    input_class=VinaDockingInput,
    config_class=VinaDockingConfig,
    output_class=VinaDockingOutput,
    metrics_class=VinaDockingPoseMetrics,
    description="Dock a small molecule into a rigid receptor with AutoDock Vina or Vinardo scoring.",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
    stochastic=True,
)
def run_vina_docking(
    inputs: VinaDockingInput,
    config: VinaDockingConfig,
    instance: ToolInstance | None = None,
) -> VinaDockingOutput:
    """Dock one small molecule into a rigid receptor with AutoDock Vina."""
    logger.debug("Using local venv for AutoDock Vina docking")

    search_box = inputs.resolved_search_box()
    grid_point_count = _validated_grid_point_count(search_box, config.grid_spacing)
    seed = config.seed if config.seed is not None else max(1, config.get_random_int())
    receptor_pdb, _ = inputs.receptor.to_pdb_with_chain_mapping()
    output_data = ToolInstance.dispatch(
        "vina",
        {
            "receptor_pdb": receptor_pdb,
            "ligand_smiles": inputs.ligand.smiles,
            "search_box": search_box.model_dump(),
            "config": {
                "scoring_function": config.scoring_function,
                "exhaustiveness": config.exhaustiveness,
                "num_poses": config.num_poses,
                "energy_range": config.energy_range,
                "min_rmsd": config.min_rmsd,
                "max_evaluations": config.max_evaluations,
                "cpu": config.cpu,
                "grid_spacing": config.grid_spacing,
                "allow_bad_residues": config.allow_bad_residues,
                "seed": seed,
                "verbose": config.verbose,
            },
        },
        instance=instance,
        config=config,
    )

    poses = [
        VinaDockingPose(
            rank=pose["rank"],
            metrics=VinaDockingPoseMetrics(
                affinity=pose["affinity"],
                rmsd_lower_bound=pose["rmsd_lower_bound"],
                rmsd_upper_bound=pose["rmsd_upper_bound"],
            ),
            sdf=pose["sdf"],
            pdbqt=pose["pdbqt"],
        )
        for pose in output_data["poses"]
    ]
    receptor_preparation = output_data.get("receptor_preparation", {})
    ligand_preparation = output_data.get("ligand_preparation", {})
    return VinaDockingOutput(
        poses=poses,
        seed=output_data["seed"],
        search_box=search_box,
        scoring_function=config.scoring_function,
        poses_sdf=output_data["poses_sdf"],
        poses_pdbqt=output_data["poses_pdbqt"],
        warnings=output_data.get("warnings", []),
        metadata={
            "vina_version": output_data.get("vina_version"),
            "meeko_version": output_data.get("meeko_version"),
            "rdkit_version": output_data.get("rdkit_version"),
            "scoring_function": config.scoring_function,
            "exhaustiveness": config.exhaustiveness,
            "requested_num_poses": config.num_poses,
            "returned_num_poses": len(poses),
            "energy_range": config.energy_range,
            "min_rmsd": config.min_rmsd,
            "max_evaluations": config.max_evaluations,
            "cpu": config.cpu,
            "grid_spacing": config.grid_spacing,
            "grid_point_count": output_data.get("grid_point_count", grid_point_count),
            "allow_bad_residues": config.allow_bad_residues,
            "ignored_receptor_residues": receptor_preparation.get("ignored_residue_ids", []),
            "receptor_template_assignments": receptor_preparation.get("template_assignments", []),
            "ligand_optimization_method": ligand_preparation.get("optimization_method"),
            "ligand_optimization_converged": ligand_preparation.get("optimization_converged"),
            "ligand_optimization_attempts": ligand_preparation.get("optimization_attempts"),
            "ligand_optimization_iteration_limits": ligand_preparation.get("optimization_iteration_limits", []),
        },
    )
