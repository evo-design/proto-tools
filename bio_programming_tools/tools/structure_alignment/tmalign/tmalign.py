"""
TMalign -- pairwise protein structure alignment by TM-score.

Wraps the TMalign binary (Zhang & Skolnick, 2005) as a ToolInstance-dispatched
tool.  Accepts two PDB text blobs, writes them to temp files, calls the binary,
and parses the two TM-scores from stdout.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import List, Union

from pydantic import Field

from bio_programming_tools.tools.tool_registry import tool
from bio_programming_tools.utils import BaseConfig
from bio_programming_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class TMalignInput(BaseToolInput):
    """Input for TMalign pairwise structure alignment.

    Attributes:
        pdb_text_1: Raw PDB content of the first structure (query / candidate).
        pdb_text_2: Raw PDB content of the second structure (reference / target).
    """

    pdb_text_1: str = Field(description="PDB content of structure 1 (query)")
    pdb_text_2: str = Field(description="PDB content of structure 2 (reference)")


class TMalignConfig(BaseConfig):
    """Configuration for TMalign alignment.

    No tool-specific parameters. Uses base parameters (verbose, device, timeout).
    """


class TMalignOutput(BaseToolOutput):
    """Output from TMalign pairwise structure alignment.

    Attributes:
        tm_score_chain_1: TM-score normalized by the length of Chain 1 (query).
        tm_score_chain_2: TM-score normalized by the length of Chain 2 (reference).
    """

    tm_score_chain_1: float = Field(
        description="TM-score normalized by length of Chain 1 (query)"
    )
    tm_score_chain_2: float = Field(
        description="TM-score normalized by length of Chain 2 (reference)"
    )

    @property
    def output_format_options(self) -> List[str]:
        return ["json"]

    @property
    def output_format_default(self) -> str:
        return "json"

    def _export_output(self, export_path: Union[Path, str], file_format: str):
        import json as json_mod

        path = Path(export_path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json_mod.dump(
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
_EXAMPLE_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"


def example_input():
    """Minimal valid input for testing and examples."""
    return TMalignInput(pdb_text_1=_EXAMPLE_PDB, pdb_text_2=_EXAMPLE_PDB)


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
def run_tmalign(
    inputs: TMalignInput, config: TMalignConfig | None = None, instance=None
) -> TMalignOutput:
    """Run TMalign on two PDB structures."""
    from bio_programming_tools.utils.tool_instance import ToolInstance

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
