"""Metal3D metal-ion site prediction for protein structures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from proto_tools.entities.structures import ResidueSelection, Structure, StructureInputBase
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


class Metal3DStructureInput(StructureInputBase):
    """A structure plus optional candidate residues to score with Metal3D.

    Attributes:
        structure (Structure): Protein structure to evaluate.
        candidate_residues (ResidueSelection | None): Optional per-chain residue
            positions to evaluate. When omitted, Metal3D evaluates all canonical
            metal-binding residue types in the protein.
    """

    candidate_residues: ResidueSelection | None = Field(
        default=None,
        title="Candidate Residues",
        description="Optional per-chain residue positions to evaluate; absent means all metal-binding residue types.",
    )


class Metal3DPredictionInput(BaseToolInput):
    """Input for Metal3D metal-site prediction.

    Attributes:
        inputs (list[Metal3DStructureInput]): Structures to evaluate.
    """

    inputs: list[Metal3DStructureInput] = InputField(
        title="Structures",
        description="Structures to evaluate with Metal3D.",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Normalize a single structure or structure-input dict to a list."""
        if isinstance(value, (str, Path, Structure, Metal3DStructureInput)):
            return [value]
        if isinstance(value, dict):
            return [value]
        return value


class Metal3DPredictionConfig(BaseConfig):
    """Configuration for Metal3D metal-ion location prediction.

    Attributes:
        model_checkpoint (Literal["metal3d-cat", "metal3d-clean", "metal3d-original"]): Checkpoint
            variant to use. ``metal3d-cat`` is the catalytic-metal dEVA checkpoint;
            ``metal3d-clean`` is the cleaned Metal3D variant; ``metal3d-original`` is the
            original Metal3D zinc checkpoint from the Nature Communications paper.
        probability_threshold (float): Probability threshold used to decide whether a
            predicted site should be annotated as a zinc atom.
        cluster_distance_threshold (float): Agglomerative clustering distance threshold
            in Angstroms for merging high-probability grid points into sites.
        max_sites (int): Maximum number of clustered sites to return per structure.
        device (str): Runtime device.
    """

    model_checkpoint: Literal["metal3d-cat", "metal3d-clean", "metal3d-original"] = ConfigField(
        default="metal3d-cat",
        title="Model Checkpoint",
        description="Metal3D checkpoint variant to use.",
        examples=["metal3d-cat", "metal3d-clean", "metal3d-original"],
        reload_on_change=True,
    )
    probability_threshold: float = ConfigField(
        default=0.2,
        ge=0.0,
        le=1.0,
        title="Probability Threshold",
        description="Threshold for reporting and annotating predicted metal sites.",
    )
    cluster_distance_threshold: float = ConfigField(
        default=7.0,
        gt=0.0,
        title="Cluster Distance Threshold",
        description="Maximum complete-linkage distance in Angstroms for merging high-probability grid points.",
    )
    max_sites: int = ConfigField(
        default=8,
        ge=1,
        title="Max Sites",
        description="Maximum number of clustered sites to return per structure.",
    )
    device: str = ConfigField(
        default="cuda",
        title="Device",
        description="Device to run the model on.",
        include_in_key=False,
    )


class Metal3DResidueProbability(Metrics):
    """Per-candidate residue Metal3D probability.

    Metrics documented in ``metric_spec``:
        probability (float): Maximum predicted metal-site probability over the
            32x32x32 residue-centered voxel box. Higher indicates stronger
            metal-binding-site signal.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "probability": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="probability",
        title="Primary Metric",
        description="Per-residue Metal3D probability.",
    )

    chain_id: str | None = Field(default=None, title="Chain ID", description="Candidate residue chain ID.")
    residue_id: int | None = Field(default=None, title="Residue ID", description="Candidate residue number.")
    residue_name: str | None = Field(default=None, title="Residue Name", description="Candidate residue name.")


class Metal3DSite(Metrics):
    """Clustered Metal3D metal-site prediction.

    Metrics documented in ``metric_spec``:
        probability (float): Maximum grid probability inside the clustered site.
            Higher indicates stronger metal-site signal.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "probability": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="probability",
        title="Primary Metric",
        description="Clustered site probability.",
    )

    x: float = Field(title="X", description="Predicted metal-site x coordinate in Angstroms.")
    y: float = Field(title="Y", description="Predicted metal-site y coordinate in Angstroms.")
    z: float = Field(title="Z", description="Predicted metal-site z coordinate in Angstroms.")


class Metal3DPredictionResult(Metrics):
    """Metal3D prediction result for one input structure.

    Metrics documented in ``metric_spec``:
        pmetal (float): Highest clustered site probability, or ``0.0`` when no
            site passes the reporting threshold.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "pmetal": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }
    primary_metric: str | None = Field(
        default="pmetal",
        title="Primary Metric",
        description="Highest predicted metal-site probability.",
    )

    found: bool = Field(title="Found", description="Whether Metal3D found a site above the threshold.")
    sites: list[Metal3DSite] = Field(default_factory=list, title="Sites", description="Clustered metal-site calls.")
    residue_probabilities: list[Metal3DResidueProbability] = Field(
        default_factory=list,
        title="Residue Probabilities",
        description="Per-candidate-residue Metal3D probabilities.",
    )
    annotated_structure: Structure = Field(
        title="Annotated Structure",
        description="Input structure with the top predicted zinc site appended when it passes the threshold.",
    )

    @model_validator(mode="after")
    def _mirror_primary_metric(self) -> Metal3DPredictionResult:
        """Populate the primary metric from clustered sites when omitted."""
        if "pmetal" not in self:
            self["pmetal"] = max((site["probability"] for site in self.sites), default=0.0)
        return self


class Metal3DPredictionOutput(BaseToolOutput):
    """Output from Metal3D metal-ion site prediction.

    Attributes:
        results (list[Metal3DPredictionResult]): One Metal3D prediction result per input structure.
    """

    results: list[Metal3DPredictionResult] = Field(
        default_factory=list,
        title="Prediction Results",
        description="One Metal3D prediction result per input structure.",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json", "pdb"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path)
        if file_format == "pdb":
            path.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(self.results):
                result.annotated_structure.write_pdb(path / f"metal3d_{i}.pdb")
            return
        if file_format == "json":
            target = path.with_suffix(".json") if path.suffix == "" else path
            with open(target, "w") as f:
                json.dump(self.model_dump(mode="json")["results"], f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for schema examples and warmup."""
    return Metal3DPredictionInput(
        inputs=[
            Metal3DStructureInput(
                structure=Structure(
                    structure=str(Path(__file__).parent.parent / "structure_metrics" / "example_input_fixture.pdb")
                ),
            )
        ]
    )


def _remap_residue_selection(
    selection: ResidueSelection | None,
    mapping: dict[str, str],
) -> dict[str, list[int]] | None:
    if selection is None:
        return None
    return {mapping.get(chain_id, chain_id): list(positions) for chain_id, positions in selection.chains.items()}


@tool(
    key="metal3d-prediction",
    label="Metal3D Prediction",
    category="structure_scoring",
    input_class=Metal3DPredictionInput,
    config_class=Metal3DPredictionConfig,
    output_class=Metal3DPredictionOutput,
    metrics_class=Metal3DPredictionResult,
    description="Predict catalytic or structural metal-ion sites in protein structures using Metal3D/dEVA checkpoints.",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    cacheable=True,
)
def run_metal3d_prediction(
    inputs: Metal3DPredictionInput,
    config: Metal3DPredictionConfig,
    instance: ToolInstance | None = None,
) -> Metal3DPredictionOutput:
    """Predict metal-ion sites in one or more protein structures with Metal3D."""
    results: list[Metal3DPredictionResult] = []

    for inp in inputs.inputs:
        pdb_content, chain_mapping = inp.structure.to_pdb_with_chain_mapping()
        pdb_to_original = {pdb_chain: original_chain for original_chain, pdb_chain in chain_mapping.items()}
        residue_selection = _remap_residue_selection(inp.candidate_residues, chain_mapping)

        payload = {
            "operation": "predict",
            "pdb_content": pdb_content,
            "candidate_residues": residue_selection,
            "model_checkpoint": config.model_checkpoint,
            "probability_threshold": config.probability_threshold,
            "cluster_distance_threshold": config.cluster_distance_threshold,
            "max_sites": config.max_sites,
            "device": config.device,
            "verbose": config.verbose,
        }
        worker_result = ToolInstance.dispatch("metal3d", payload, instance=instance, config=config)

        sites = [
            Metal3DSite(
                x=float(site["x"]), y=float(site["y"]), z=float(site["z"]), probability=float(site["probability"])
            )
            for site in worker_result.get("sites", [])
        ]
        residue_probs = [
            Metal3DResidueProbability(
                chain_id=pdb_to_original.get(item.get("chain_id"), item.get("chain_id")),
                residue_id=item.get("residue_id"),
                residue_name=item.get("residue_name"),
                probability=float(item["probability"]),
            )
            for item in worker_result.get("residue_probabilities", [])
        ]
        annotated_structure = Structure(
            structure=worker_result["annotated_pdb"],
            structure_format="pdb",
            source="metal3d-prediction",
        )
        results.append(
            Metal3DPredictionResult(
                pmetal=float(worker_result.get("pmetal", 0.0)),
                found=bool(worker_result.get("found", False)),
                sites=sites,
                residue_probabilities=residue_probs,
                annotated_structure=annotated_structure,
            )
        )

    return Metal3DPredictionOutput(results=results)
