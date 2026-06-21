"""proto_tools/tools/structure_alignment/usalign/usalign.py.

Wraps the USalign binary (Zhang et al., 2022) as a ToolInstance-dispatched tool.
Accepts two PDB text blobs, calls the binary with ``-mm 1 -ter 1`` flags for
multimer support, and parses the two TM-scores from stdout.
"""

import json
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from proto_tools.entities import Structure
from proto_tools.tools.structure_alignment.superposition import (
    SuperpositionTransform,
    build_superimposed_pdb,
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

logger = getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class USalignInput(BaseToolInput):
    """Input for USalign pairwise structure alignment.

    Attributes:
        query_structure (Structure): Query / candidate structure.
        reference_structure (Structure): Reference / target structure.

    Both fields accept a ``Structure`` object, a file path, or raw PDB/CIF
    content; each is normalised to a ``Structure``.
    """

    query_structure: Structure = InputField(
        title="Query Structure",
        description="Query / candidate structure (Structure object, file path, or raw PDB/CIF string)",
    )
    reference_structure: Structure = InputField(
        title="Reference Structure",
        description="Reference / target structure (Structure object, file path, or raw PDB/CIF string)",
    )


class USalignConfig(BaseConfig):
    """Configuration for USalign alignment.

    Attributes:
        include_superimposed_pdb (bool): When true, also return a multi-model PDB
            overlaying the aligned structures (off by default to keep results small).
    """

    include_superimposed_pdb: bool = ConfigField(
        title="Include Superimposed PDB",
        default=False,
        description="Also return a multi-model PDB overlaying the aligned structures.",
    )


class USalignMetrics(Metrics):
    """Pairwise alignment scores emitted by USalign.

    Metrics documented in ``metric_spec``:
        tm_score_structure_1 (float): TM-score normalized by the length of
            Structure 1 (query). Always present. Range 0-1.
        tm_score_structure_2 (float): TM-score normalized by the length of
            Structure 2 (reference). Always present. Range 0-1.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "tm_score_structure_1": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "tm_score_structure_2": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
    }


class USalignOutput(BaseToolOutput):
    """Output from USalign pairwise structure alignment.

    Attributes:
        metrics (USalignMetrics): Pairwise alignment scores. Access metrics via
            ``output.metrics.tm_score_structure_1`` or
            ``output.tm_score_structure_1`` (the forwarded shortcut from
            :class:`BaseToolOutput`).
        superposition (SuperpositionTransform | None): Rigid-body transform that
            superposes the query structure onto the reference. ``None`` if USalign
            didn't emit a parseable matrix.
        superimposed_pdb (str | None): Multi-model PDB overlaying the aligned
            structures (MODEL 1 = transformed query, MODEL 2 = reference). Present only
            when ``config.include_superimposed_pdb`` is set and a transform was found.
    """

    metrics: USalignMetrics = Field(
        default_factory=USalignMetrics,
        title="Alignment Scores",
        description="Pairwise alignment scores",
    )
    superposition: SuperpositionTransform | None = Field(
        default=None,
        title="Superposition Transform",
        description="Rigid-body transform that superposes the query onto the reference.",
    )
    superimposed_pdb: str | None = Field(
        default=None,
        title="Superimposed PDB",
        description="Multi-model PDB overlaying the aligned structures, for download.",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats; ``pdb`` is offered when a superimposed overlay exists."""
        return ["json", "pdb"] if self.superimposed_pdb else ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if file_format == "pdb":
            if self.superimposed_pdb is None:
                raise ValueError("No superimposed structure; rerun with include_superimposed_pdb=True.")
            path.with_suffix(".pdb").write_text(self.superimposed_pdb)
            return
        if file_format == "json":
            with open(path.with_suffix(".json"), "w") as f:
                json.dump(
                    {
                        "tm_score_structure_1": self.metrics["tm_score_structure_1"],
                        "tm_score_structure_2": self.metrics["tm_score_structure_2"],
                    },
                    f,
                    indent=2,
                )
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    structure = Structure.from_file(_EXAMPLE_PDB_PATH)
    return USalignInput(query_structure=structure, reference_structure=structure)


@tool(
    key="usalign-alignment",
    label="USalign Structure Alignment",
    category="structure_alignment",
    input_class=USalignInput,
    config_class=USalignConfig,
    output_class=USalignOutput,
    metrics_class=USalignMetrics,
    uses_gpu=False,
    description=(
        "Universal structure alignment using USalign (Zhang et al., 2022). "
        "Supports monomers and multimers. Returns TM-scores normalized by "
        "each structure."
    ),
    example_input=example_input,
)
def run_usalign(inputs: USalignInput, config: USalignConfig, instance: Any = None) -> USalignOutput:
    """Run USalign on two PDB structures."""
    input_data = {
        "pdb_text_1": inputs.query_structure.structure_pdb,
        "pdb_text_2": inputs.reference_structure.structure_pdb,
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "usalign",
        input_data,
        instance=instance,
        config=config,
    )

    superposition = SuperpositionTransform.from_optional(output_data.get("rotation"), output_data.get("translation"))
    superimposed_pdb = None
    if config.include_superimposed_pdb and superposition is not None:
        superimposed_pdb = build_superimposed_pdb(
            inputs.query_structure.structure_pdb,
            inputs.reference_structure.structure_pdb,
            superposition,
        )

    return USalignOutput(
        metrics=USalignMetrics(
            tm_score_structure_1=output_data["tm_score_structure_1"],
            tm_score_structure_2=output_data["tm_score_structure_2"],
        ),
        superposition=superposition,
        superimposed_pdb=superimposed_pdb,
        metadata={"tool": "usalign"},
    )
