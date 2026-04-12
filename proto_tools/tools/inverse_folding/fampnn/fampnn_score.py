"""proto_tools/tools/inverse_folding/fampnn/fampnn_score.py.

FAMPNN mutation scoring tool.
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
class MutationInput(BaseModel):
    """A single mutation specification.

    Attributes:
        structure (Structure): Protein structure to evaluate mutations against.
        mutations (list[str]): List of mutation strings. Each mutation uses the format
            '<WT><1-indexed_position><MUT>' with single-letter amino acid codes.
            Multiple simultaneous mutations are joined with colons: 'N1P:N2R'.
    """

    model_config = ConfigDict(extra="forbid")

    structure: Structure = Field(description="Protein structure to evaluate mutations against")
    mutations: list[str] = Field(
        description="List of mutation strings (format: 'A1V' or 'A1V:G5L' for multi-site, 1-indexed)"
    )


class FAMPNNScoreInput(BaseToolInput):
    """Input for FAMPNN mutation scoring.

    Attributes:
        inputs (list[MutationInput]): List of MutationInput objects, each containing a structure
            and mutations to score.
    """

    inputs: list[MutationInput] = InputField(
        description="List of mutation inputs, each with a structure and mutations."
    )


class FAMPNNScoreConfig(BaseConfig):
    """Configuration for FAMPNN mutation scoring.

    Attributes:
        model_variant (str): Checkpoint variant. '0.3_cath' recommended for scoring.
        batch_size (int): Number of mutations to score simultaneously on GPU.
        seq_only (bool): If True, score without sidechain context (backbone-only).
        scn_diffusion_steps (int): Number of sidechain diffusion denoising steps.
        scn_step_scale (float): Step scale for sidechain diffusion.
        seed (int): Random seed.
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
        description="Number of mutations to score simultaneously on GPU.",
    )
    seq_only: bool = ConfigField(
        title="Sequence Only",
        default=False,
        description="If True, score without sidechain context",
    )
    scn_diffusion_steps: int = ConfigField(
        title="Sidechain Diffusion Steps",
        default=50,
        ge=1,
        description="Number of sidechain diffusion denoising steps",
        hidden=True,
    )
    scn_step_scale: float = ConfigField(
        title="Sidechain Step Scale",
        default=1.5,
        gt=0.0,
        description="Step scale (eta) for sidechain diffusion",
        hidden=True,
    )
    seed: int = ConfigField(
        title="Random Seed",
        default=42,
        description="Random seed",
        hidden=True,
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        hidden=True,
        include_in_key=False,
    )


class MutationScoreResult(BaseModel):
    """Mutation score for a single structure.

    Attributes:
        mutations (list[str]): Mutation strings that were scored.
        scores (list[float]): Log-likelihood ratio scores for each mutation.
            Positive = mutation is more likely than wild-type.
    """

    model_config = ConfigDict(extra="forbid")

    mutations: list[str] = Field(description="Mutation strings that were scored")
    scores: list[float] = Field(description="Log-likelihood ratio scores (positive = favored over wild-type)")


class FAMPNNScoreOutput(BaseToolOutput):
    """Output for FAMPNN mutation scoring.

    Attributes:
        results (list[MutationScoreResult]): List of MutationScoreResult objects, one per input structure.
    """

    results: list[MutationScoreResult] = Field(description="Scoring results, one per input structure")

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

        if file_format == "csv":
            path.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(self.results):
                out_file = path / f"scores_{i}.csv"
                with open(out_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["mutation", "score"])
                    for mut, score in zip(result.mutations, result.scores, strict=False):
                        writer.writerow([mut, score])
        elif file_format == "json":
            path.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(self.results):
                out_file = path / f"scores_{i}.json"
                with open(out_file, "w") as f:
                    json.dump({"mutations": result.mutations, "scores": result.scores}, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return FAMPNNScoreInput(
        inputs=[
            MutationInput(
                structure=Structure.from_file(str(Path(__file__).parents[1] / "example_input_fixture.pdb")),
                mutations=["M1V"],
            )
        ]
    )


@tool(
    key="fampnn-score",
    label="FAMPNN Mutation Scoring",
    category="inverse_folding",
    input_class=FAMPNNScoreInput,
    config_class=FAMPNNScoreConfig,
    output_class=FAMPNNScoreOutput,
    description="Score protein mutations with full-atom context using FAMPNN",
    uses_gpu=True,
    cacheable=True,
    example_input=example_input,
    iterable_input_field="inputs",
    iterable_output_field="results",
)
def run_fampnn_score(
    inputs: FAMPNNScoreInput,
    config: FAMPNNScoreConfig,
    instance: Any = None,
) -> FAMPNNScoreOutput:
    """Score protein mutations with full-atom sidechain context using FAMPNN.

    Evaluates mutation fitness by masking the mutated position's sequence and
    sidechain, then computing the conditional log-likelihood ratio of the mutant
    versus wild-type residue. FAMPNN's advantage is conditioning on the full-atom
    structure of surrounding residues.

    Args:
        inputs (FAMPNNScoreInput): FAMPNNScoreInput containing structures and mutations to score.
        config (FAMPNNScoreConfig): Configuration for scoring.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        FAMPNNScoreOutput: FAMPNNScoreOutput with log-likelihood ratio scores for each mutation.
    """
    results = []

    for inp in progress_bar(
        inputs.inputs,
        desc="FAMPNN scoring",
        unit="structure",
        disable=not config.verbose,
    ):
        input_dict = {
            "operation": "score_mutations",
            "pdb_contents": inp.structure.structure_pdb,
            "mutations": inp.mutations,
            "batch_size": config.batch_size,
            "seq_only": config.seq_only,
            "scn_diffusion_steps": config.scn_diffusion_steps,
            "scn_step_scale": config.scn_step_scale,
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
        results.append(
            MutationScoreResult(
                mutations=result["mutations"],
                scores=result["scores"],
            )
        )

    return FAMPNNScoreOutput(results=results)
