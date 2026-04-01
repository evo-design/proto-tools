"""proto_tools/tools/structure_alignment/usalign/usalign.py.

Wraps the USalign binary (Zhang et al., 2022) as a ToolInstance-dispatched tool.
Accepts two PDB text blobs, calls the binary with ``-mm 1 -ter 1`` flags for
multimer support, and parses the two TM-scores from stdout.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

from pydantic import Field

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    InputField,
    ToolInstance,
)

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

    pdb_text_1: str = InputField(description="PDB content of structure 1 (query)")
    pdb_text_2: str = InputField(description="PDB content of structure 2 (reference)")


class USalignConfig(BaseConfig):
    """Configuration for USalign alignment.

    No tool-specific parameters. Uses base parameters (verbose, device, timeout).
    """


class USalignOutput(BaseToolOutput):
    """Output from USalign pairwise structure alignment.

    Attributes:
        tm_score_structure_1 (float): TM-score normalized by the length of Structure 1 (query).
        tm_score_structure_2 (float): TM-score normalized by the length of Structure 2 (reference).
    """

    tm_score_structure_1: float = Field(description="TM-score normalized by length of Structure 1 (query)")
    tm_score_structure_2: float = Field(description="TM-score normalized by length of Structure 2 (reference)")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:  # noqa: ARG002 — required by base class _export_output interface
        import json as json_mod

        path = Path(export_path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json_mod.dump(
                {
                    "tm_score_structure_1": self.tm_score_structure_1,
                    "tm_score_structure_2": self.tm_score_structure_2,
                },
                f,
                indent=2,
            )


# ============================================================================
# Tool Implementation
# ============================================================================
_EXAMPLE_PDB_PATH = str(Path(__file__).parents[1] / "examples" / "example.pdb")


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
    uses_gpu=False,
    description=(
        "Universal structure alignment using USalign (Zhang et al., 2022). "
        "Supports monomers and multimers. Returns TM-scores normalized by "
        "each structure."
    ),
    example_input=example_input,
)
def run_usalign(inputs: USalignInput, config: USalignConfig | None = None, instance: Any = None) -> USalignOutput:
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
        tm_score_structure_1=output_data["tm_score_structure_1"],
        tm_score_structure_2=output_data["tm_score_structure_2"],
        metadata={"tool": "usalign"},
    )
