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

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
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
        pdb_text_1 (str): Raw PDB content of the first structure (query / candidate).
        pdb_text_2 (str): Raw PDB content of the second structure (reference / target).
    """

    pdb_text_1: str = InputField(title="Structure 1 PDB", description="PDB content of structure 1 (query)")
    pdb_text_2: str = InputField(title="Structure 2 PDB", description="PDB content of structure 2 (reference)")


class USalignConfig(BaseConfig):
    """Configuration for USalign alignment.

    No tool-specific parameters. Uses base parameters (verbose, device, timeout).
    """


class USalignMetrics(Metrics):
    """Pairwise alignment scores emitted by USalign.

    Metrics documented in ``metric_spec``:
        tm_score_structure_1 (float): TM-score normalized by the length of
            Structure 1 (query). Always present. Range 0-1.
        tm_score_structure_2 (float): TM-score normalized by the length of
            Structure 2 (reference). Always present. Range 0-1.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "tm_score_structure_1": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
        "tm_score_structure_2": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0},
    }


class USalignOutput(BaseToolOutput):
    """Output from USalign pairwise structure alignment.

    Attributes:
        metrics (USalignMetrics): Pairwise alignment scores. Access metrics via
            ``output.metrics.tm_score_structure_1`` or
            ``output.tm_score_structure_1`` (the forwarded shortcut from
            :class:`BaseToolOutput`).
    """

    metrics: USalignMetrics = Field(
        default_factory=USalignMetrics,
        title="Alignment Scores",
        description="Pairwise alignment scores",
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:  # noqa: ARG002 — required by base class _export_output interface
        path = Path(export_path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "tm_score_structure_1": self.metrics["tm_score_structure_1"],
                    "tm_score_structure_2": self.metrics["tm_score_structure_2"],
                },
                f,
                indent=2,
            )


# ============================================================================
# Tool Implementation
# ============================================================================
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "example_input_fixture.pdb")


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    _pdb_text = Path(_EXAMPLE_PDB_PATH).read_text()
    return USalignInput(pdb_text_1=_pdb_text, pdb_text_2=_pdb_text)


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
        "pdb_text_1": inputs.pdb_text_1,
        "pdb_text_2": inputs.pdb_text_2,
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "usalign",
        input_data,
        instance=instance,
        config=config,
    )

    return USalignOutput(
        metrics=USalignMetrics(
            tm_score_structure_1=output_data["tm_score_structure_1"],
            tm_score_structure_2=output_data["tm_score_structure_2"],
        ),
        metadata={"tool": "usalign"},
    )
