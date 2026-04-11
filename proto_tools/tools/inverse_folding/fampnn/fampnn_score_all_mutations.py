"""proto_tools/tools/inverse_folding/fampnn/fampnn_score_all_mutations.py.

FAMPNN exhaustive single-mutation scoring tool.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
from proto_tools.utils.progress import progress_bar

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
class FAMPNNScoreAllMutationsInput(BaseToolInput):
    """Input for scoring all possible single mutations.

    Attributes:
        inputs (list[Structure]): List of structures to score all mutations for.
    """

    inputs: list[Structure] = InputField(description="List of structures to score all possible single mutations.")


class FAMPNNScoreAllMutationsConfig(BaseConfig):
    """Configuration for exhaustive FAMPNN mutation scoring.

    Attributes:
        model_variant (str): Checkpoint variant. '0.3_cath' recommended for scoring.
        batch_size (int): Number of positions to score simultaneously on GPU.
        device (str): Device to run on.
    """

    model_variant: str = ConfigField(
        title="Model Variant",
        default="0.3_cath",
        description="FAMPNN checkpoint: '0.3_cath' recommended for scoring",
        examples=["0.3_cath", "0.3", "0.0"],
    )
    batch_size: int = ConfigField(
        title="Batch Size",
        default=16,
        ge=1,
        description="Number of positions to score simultaneously on GPU.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )


class AllMutationsScoreResult(BaseModel):
    """All single-mutation scores for a structure.

    Attributes:
        scores (dict[str, dict[str, float]]): Dictionary mapping position labels (e.g., '1A' for position 1,
            wild-type Ala) to dictionaries of {mutant_residue: score}. Scores
            are log-likelihood ratios (positive = favored over wild-type).
    """

    model_config = ConfigDict(extra="forbid")

    scores: dict[str, dict[str, float]] = Field(
        description="Position label -> {mutant_residue: log-likelihood ratio score}"
    )


class FAMPNNScoreAllMutationsOutput(BaseToolOutput):
    """Output for exhaustive FAMPNN mutation scoring.

    Attributes:
        results (list[AllMutationsScoreResult]): List of AllMutationsScoreResult objects, one per input structure.
    """

    results: list[AllMutationsScoreResult] = Field(description="All-mutations scoring results, one per input structure")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: Any, file_format: Any) -> None:
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)

        if file_format == "csv":
            for i, result in enumerate(self.results):
                out_file = path / f"all_scores_{i}.csv"
                if not result.scores:
                    continue
                # Get all residue types from any position
                residue_types = list(next(iter(result.scores.values())).keys())
                with open(out_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["position", *residue_types])
                    for pos_label, scores in result.scores.items():
                        row = [pos_label] + [scores.get(r, 0.0) for r in residue_types]
                        writer.writerow(row)
        elif file_format == "json":
            for i, result in enumerate(self.results):
                out_file = path / f"all_scores_{i}.json"
                with open(out_file, "w") as f:
                    json.dump(result.scores, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return FAMPNNScoreAllMutationsInput(
        inputs=[
            Structure.from_file(str(Path(__file__).parents[1] / "examples" / "example.pdb")),
        ]
    )


@tool(
    key="fampnn-score-all-mutations",
    label="FAMPNN Score All Mutations",
    category="inverse_folding",
    input_class=FAMPNNScoreAllMutationsInput,
    config_class=FAMPNNScoreAllMutationsConfig,
    output_class=FAMPNNScoreAllMutationsOutput,
    description="Score every possible single mutation at every position using FAMPNN",
    uses_gpu=True,
    cacheable=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
)
def run_fampnn_score_all_mutations(
    inputs: FAMPNNScoreAllMutationsInput,
    config: FAMPNNScoreAllMutationsConfig,
    instance: Any = None,
) -> FAMPNNScoreAllMutationsOutput:
    """Score every possible single amino acid substitution at every position.

    For each position in the protein, masks that position and computes the
    log-likelihood ratio of each possible mutation relative to the wild-type
    residue. Useful for generating comprehensive mutational landscapes.

    Args:
        inputs (FAMPNNScoreAllMutationsInput): FAMPNNScoreAllMutationsInput containing structures.
        config (FAMPNNScoreAllMutationsConfig): Configuration for scoring.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        FAMPNNScoreAllMutationsOutput: FAMPNNScoreAllMutationsOutput with per-position mutation scores.
    """
    results = []

    for structure in progress_bar(
        inputs.inputs,
        desc="FAMPNN scoring all mutations",
        unit="structure",
        disable=not config.verbose,
    ):
        input_dict = {
            "operation": "score_all_mutations",
            "pdb_contents": structure.structure_pdb,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "model_variant": config.model_variant,
            "device": config.device,
            "verbose": config.verbose,
        }
        result = ToolInstance.dispatch(
            "fampnn",
            input_dict,
            instance=instance,
            config=config,
        )
        results.append(AllMutationsScoreResult(scores=result["scores"]))

    return FAMPNNScoreAllMutationsOutput(results=results)
