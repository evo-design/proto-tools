"""proto_tools/tools/structure_alignment/tmalign/tmalign.py.

Wraps the TMalign binary (Zhang & Skolnick, 2005) as a ToolInstance-dispatched
tool.  Accepts two PDB text blobs, writes them to temp files, calls the binary,
and parses the two TM-scores from stdout.
"""

import json
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
class TMalignInput(BaseToolInput):
    """Input for TMalign pairwise structure alignment.

    Attributes:
        pdb_text_1 (str): Raw PDB content of the first structure (query / candidate).
        pdb_text_2 (str): Raw PDB content of the second structure (reference / target).
    """

    pdb_text_1: str = InputField(description="PDB content of structure 1 (query)")
    pdb_text_2: str = InputField(description="PDB content of structure 2 (reference)")


class TMalignConfig(BaseConfig):
    """Configuration for TMalign alignment.

    No tool-specific parameters. Uses base parameters (verbose, device, timeout).
    """


class TMalignOutput(BaseToolOutput):
    """Output from TMalign pairwise structure alignment.

    Attributes:
        tm_score_chain_1 (float): TM-score normalized by the length of Chain 1 (query).
        tm_score_chain_2 (float): TM-score normalized by the length of Chain 2 (reference).
    """

    tm_score_chain_1: float = Field(description="TM-score normalized by length of Chain 1 (query)")
    tm_score_chain_2: float = Field(description="TM-score normalized by length of Chain 2 (reference)")

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
                    "tm_score_chain_1": self.tm_score_chain_1,
                    "tm_score_chain_2": self.tm_score_chain_2,
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
    return TMalignInput(pdb_text_1=_pdb_text, pdb_text_2=_pdb_text)


@tool(
    key="tmalign-alignment",
    label="TMalign Structure Alignment",
    category="structure_alignment",
    input_class=TMalignInput,
    config_class=TMalignConfig,
    output_class=TMalignOutput,
    uses_gpu=False,
    description=(
        "Pairwise protein structure alignment using TMalign "
        "(Zhang & Skolnick, 2005). Returns TM-scores normalized by each chain."
    ),
    example_input=example_input,
)
def run_tmalign(inputs: TMalignInput, config: TMalignConfig, instance: Any = None) -> TMalignOutput:
    """Run TMalign on two PDB structures."""
    input_data = {
        "pdb_text_1": inputs.pdb_text_1,
        "pdb_text_2": inputs.pdb_text_2,
    }

    input_data["device"] = "cpu"
    output_data = ToolInstance.dispatch(
        "tmalign",
        input_data,
        instance=instance,
        config=config,
    )

    return TMalignOutput(
        tm_score_chain_1=output_data["tm_score_chain_1"],
        tm_score_chain_2=output_data["tm_score_chain_2"],
        metadata={"tool": "tmalign"},
    )
