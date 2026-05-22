"""DSSP secondary-structure assignment tool."""

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.entities.structures import Structure
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    InputField,
    ToolInstance,
)
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)

# PDB format stores chain IDs in a single character column. The Structure
# conversion path maps arbitrary chain labels into A-Z, a-z, 0-9.
MAX_CHAINS_FOR_PDB = 62


class DSSPSecondaryStructureMetrics(Metrics):
    """DSSP secondary-structure percentages for one structure chain.

    Metrics documented in ``metric_spec``:
        helix_pct (float): Percentage of residues assigned H/G/I by DSSP.
        sheet_pct (float): Percentage of residues assigned E by DSSP.
        loop_pct (float): Percentage of residues assigned any other DSSP state.

    Attributes:
        chain_id (str): Analyzed chain label in the input structure namespace.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "helix_pct": {"availability": "always", "type": "float", "min": 0.0, "max": 100.0, "unit": "%"},
        "sheet_pct": {"availability": "always", "type": "float", "min": 0.0, "max": 100.0, "unit": "%"},
        "loop_pct": {"availability": "always", "type": "float", "min": 0.0, "max": 100.0, "unit": "%"},
    }

    chain_id: str = Field(title="Chain ID", description="Analyzed chain label in the input structure namespace")


class DSSPStructureInput(BaseModel):
    """A structure plus the chain to assign with DSSP.

    Attributes:
        structure (Structure): Protein structure to analyze.
        chain_id (str): Chain label to analyze.
    """

    model_config = ConfigDict(extra="forbid")

    structure: Structure = Field(title="Input Structure", description="Protein structure to analyze")
    chain_id: str = Field(default="A", title="Chain ID", description="Chain label to analyze")

    @model_validator(mode="before")
    @classmethod
    def _coerce_structure(cls, data: Any) -> Any:
        if isinstance(data, (str, Path)):
            return {"structure": Structure(structure=str(data))}
        if isinstance(data, Structure):
            return {"structure": data}
        if isinstance(data, dict):
            structure = data.get("structure")
            if isinstance(structure, (str, Path)):
                data = {**data, "structure": Structure(structure=str(structure))}
        return data

    @model_validator(mode="after")
    def _validate_chain(self) -> "DSSPStructureInput":
        available = set(self.structure.get_chain_ids())
        if len(available) > MAX_CHAINS_FOR_PDB:
            raise ValueError(
                f"Structure has {len(available)} chains, but DSSP dispatch uses PDB format which supports "
                f"at most {MAX_CHAINS_FOR_PDB} single-character chain IDs.",
            )
        if self.chain_id not in available:
            raise ValueError(f"Chain {self.chain_id!r} not found in structure. Available chains: {sorted(available)}")
        return self


class DSSPSecondaryStructureInput(BaseToolInput):
    """Input for DSSP secondary-structure assignment.

    Attributes:
        inputs (list[DSSPStructureInput]): Structures and chains to analyze.
    """

    inputs: list[DSSPStructureInput] = InputField(
        title="Structures",
        description="Structures and chains to analyze with DSSP",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, value: Any) -> Any:
        """Normalize a single structure/input to a list."""
        if isinstance(value, (str, Path, Structure, DSSPStructureInput)):
            value = [value]
        if isinstance(value, dict):
            value = [value]
        return value


class DSSPSecondaryStructureConfig(BaseConfig):
    """Configuration for DSSP secondary-structure assignment.

    DSSP secondary-structure assignment has no tool-specific parameters.
    """


class DSSPSecondaryStructureOutput(BaseToolOutput):
    """Output from DSSP secondary-structure assignment.

    Attributes:
        results (list[DSSPSecondaryStructureMetrics]): Per-input secondary-structure percentages.
    """

    results: list[DSSPSecondaryStructureMetrics] = Field(
        default_factory=list,
        title="SS Percentages",
        description="DSSP secondary-structure percentages, one per input",
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
        df = pd.DataFrame([dict(r.items()) | {"chain_id": r.chain_id} for r in self.results])
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return DSSPSecondaryStructureInput(
        inputs=[
            DSSPStructureInput(
                structure=Structure(
                    structure=str(Path(__file__).parent.parent / "structure_metrics" / "example_input_fixture.pdb")
                ),
                chain_id="A",
            )
        ]
    )


@tool(
    key="dssp-secondary-structure",
    label="DSSP Secondary Structure",
    category="structure_scoring",
    input_class=DSSPSecondaryStructureInput,
    config_class=DSSPSecondaryStructureConfig,
    output_class=DSSPSecondaryStructureOutput,
    metrics_class=DSSPSecondaryStructureMetrics,
    description="Assign helix/sheet/loop percentages using the DSSP binary",
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
    cacheable=True,
)
def run_dssp_secondary_structure(
    inputs: DSSPSecondaryStructureInput,
    config: DSSPSecondaryStructureConfig | None = None,
    instance: ToolInstance | None = None,
) -> DSSPSecondaryStructureOutput:
    """Assign secondary structure with DSSP in a standalone tool environment."""
    logger.debug("Using local venv for DSSP secondary-structure assignment")

    pdb_contents: list[str] = []
    pdb_chain_ids: list[str] = []
    chain_ids: list[str] = []
    for inp in inputs.inputs:
        pdb_content, mmcif_to_pdb = inp.structure.to_pdb_with_chain_mapping()
        pdb_contents.append(pdb_content)
        pdb_chain_ids.append(mmcif_to_pdb[inp.chain_id])
        chain_ids.append(inp.chain_id)

    input_data = {
        "pdb_contents": pdb_contents,
        "chain_ids": pdb_chain_ids,
        "device": "cpu",
    }
    output_data = ToolInstance.dispatch(
        "dssp",
        input_data,
        instance=instance,
        config=config,
    )

    results = [
        DSSPSecondaryStructureMetrics(chain_id=chain_id, **result)
        for chain_id, result in zip(chain_ids, output_data["results"], strict=True)
    ]
    return DSSPSecondaryStructureOutput(
        metadata={"num_structures": len(inputs.inputs)},
        results=results,
    )
