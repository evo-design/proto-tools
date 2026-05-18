"""PyRosetta Spatial Aggregation Propensity (SAP) scoring tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import PyRosettaRelaxConfig
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    ScoringStructureInput,
    prepare_pdb_and_chain_maps,
    relax_inputs_via_pyrosetta,
    remap_per_residue_chain_ids,
    warn_about_dropped_residues,
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
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class ResidueSAP(BaseModel):
    """Per-residue SAP contribution.

    Attributes:
        chain_id (str): Chain identifier.
        residue_index (int): 1-indexed residue position.
        residue_name (str): Three-letter residue code.
        sap_score (float): Per-residue SAP contribution. Higher values indicate
            that this residue contributes more to surface hydrophobic patches.
    """

    model_config = ConfigDict(extra="forbid")

    chain_id: str = Field(description="Chain identifier")
    residue_index: int = Field(description="1-indexed residue position")
    residue_name: str = Field(description="Three-letter residue code")
    sap_score: float = Field(description="Per-residue SAP contribution")


class PyRosettaSAPMetrics(Metrics):
    """SAP score for a single structure.

    Metrics documented in ``metric_spec``:
        sap_score (float): Spatial Aggregation Propensity score. Higher values
            indicate more aggregation-prone surface hydrophobicity. Always present.

    Attributes:
        per_residue (list[ResidueSAP]): Per-residue SAP contributions. Declared
            as a real field (not a metric) because each entry carries
            chain/residue identifiers alongside the score.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "sap_score": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": None,
        },
    }
    primary_metric: str | None = "sap_score"

    per_residue: list[ResidueSAP] = Field(description="Per-residue SAP contributions")


class PyRosettaSAPInput(BaseToolInput):
    """Input for PyRosetta SAP scoring.

    Attributes:
        inputs (list[ScoringStructureInput]): Protein structures to score,
            each with optional chain selection. Accepts bare Structure objects,
            PDB file paths, or PDB content strings for convenience.
    """

    inputs: list[ScoringStructureInput] = InputField(
        description="Protein structures with optional chain selection for SAP scoring"
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Normalize a single structure/input to a list."""
        if isinstance(value, (str, Path, Structure, ScoringStructureInput)):
            value = [value]
        if isinstance(value, dict):
            value = [value]
        return value


class PyRosettaSAPConfig(BaseConfig):
    """Configuration for PyRosetta SAP scoring.

    Attributes:
        pre_relax_structures (bool): If ``True``, run ``pyrosetta-relax`` on
            each input structure before scoring. Default ``False``.
        relax_config (PyRosettaRelaxConfig): Settings used when
            ``pre_relax_structures=True``. Ignored otherwise.
    """

    pre_relax_structures: bool = ConfigField(
        title="Pre-relax Structures",
        default=False,
        description="If True, run pyrosetta-relax on each input structure before scoring.",
    )
    relax_config: PyRosettaRelaxConfig = ConfigField(
        default_factory=PyRosettaRelaxConfig,
        title="Relax Config",
        description="Settings used when pre_relax_structures=True. Ignored otherwise.",
    )

    def preprocess(self, inputs: PyRosettaSAPInput) -> PyRosettaSAPInput:  # type: ignore[override]
        """Apply optional FastRelax preprocess to input structures."""
        if not self.pre_relax_structures:
            return inputs
        return inputs.model_copy(update={"inputs": relax_inputs_via_pyrosetta(inputs.inputs, self.relax_config)})

    @property
    def cpus_per_instance(self) -> int | None:
        """Opt in to ToolPool CPU fan-out — PyRosetta runs single-threaded per pose."""
        return 1


class PyRosettaSAPOutput(BaseToolOutput):
    """Output from PyRosetta SAP scoring.

    Attributes:
        results (list[PyRosettaSAPMetrics]): SAP scores with per-residue breakdown,
            one per input structure.
    """

    results: list[PyRosettaSAPMetrics] = Field(
        default_factory=list,
        description="SAP scores with per-residue breakdown, one per input structure",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        import pandas as pd

        path = Path(export_path).with_suffix(f".{file_format}")
        rows = [
            {
                "structure_index": i,
                "sap_score_total": result["sap_score"],
                "chain_id": res.chain_id,
                "residue_index": res.residue_index,
                "residue_name": res.residue_name,
                "sap_score": res.sap_score,
            }
            for i, result in enumerate(self.results)
            for res in result.per_residue
        ]
        df = pd.DataFrame(rows)
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return PyRosettaSAPInput(
        inputs=[
            ScoringStructureInput(
                structure=Structure(structure=str(Path(__file__).parent / "examples" / "example.pdb"))
            )
        ]
    )


@tool(
    key="pyrosetta-sap",
    label="PyRosetta SAP Score",
    category="structure_scoring",
    input_class=PyRosettaSAPInput,
    config_class=PyRosettaSAPConfig,
    output_class=PyRosettaSAPOutput,
    metrics_class=PyRosettaSAPMetrics,
    description="Compute Spatial Aggregation Propensity (SAP) scores for protein structures using PyRosetta",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
    cacheable=True,
)
def run_pyrosetta_sap(
    inputs: PyRosettaSAPInput,
    config: PyRosettaSAPConfig | None = None,
    instance: ToolInstance | None = None,
) -> PyRosettaSAPOutput:
    """Compute Spatial Aggregation Propensity (SAP) scores using PyRosetta.

    SAP quantifies the aggregation propensity of a protein's surface by
    measuring exposed hydrophobicity. Higher scores indicate greater
    aggregation risk. Per-residue contributions identify which residues
    drive aggregation propensity.

    Chain selection controls which residues are scored, while the full
    structure is always used for SASA and burial context.

    Args:
        inputs (PyRosettaSAPInput): Structures to score with optional chain selection.
        config (PyRosettaSAPConfig | None): Configuration (no parameters needed).
        instance (ToolInstance | None): Optional ToolInstance for persistent execution.

    Returns:
        PyRosettaSAPOutput: SAP scores with per-residue breakdown.
    """
    logger.debug("Using local venv for PyRosetta SAP scoring")

    pdb_contents, chain_ids_list, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps(inputs.inputs)

    input_data = {
        "operation": "sap",
        "pdb_contents": pdb_contents,
        "chain_ids_list": chain_ids_list,
        "device": "cpu",
    }

    output_data = ToolInstance.dispatch(
        "pyrosetta",
        input_data,
        instance=instance,
        config=config,
    )

    warn_about_dropped_residues(output_data["results"])
    remap_per_residue_chain_ids(output_data["results"], pdb_to_mmcif_maps)
    results = [PyRosettaSAPMetrics(**r) for r in output_data["results"]]

    return PyRosettaSAPOutput(
        metadata={"num_structures": len(inputs.inputs)},
        results=results,
    )
