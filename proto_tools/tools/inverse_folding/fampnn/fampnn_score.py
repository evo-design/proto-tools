"""proto_tools/tools/inverse_folding/fampnn/fampnn_score.py.

FAMPNN mutation scoring tool.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from proto_tools.entities.structures import Structure
from proto_tools.tools.inverse_folding.fampnn._readable import fampnn_readable_positions
from proto_tools.tools.inverse_folding.shared_data_models import flat_readable_positions, to_model_numbering
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

    structure: Structure = Field(
        title="Input Structure",
        description="Protein structure to evaluate mutations against",
    )
    mutations: list[str] = Field(
        title="Mutations",
        description="Mutation strings (format: 'A1V' or 'A1V:G5L' for multi-site, 1-indexed)",
    )


def _to_model_numbering(mutations: list[str], structure: Structure) -> tuple[list[str], list[str]]:
    """Renumber mutation positions from the structure's numbering into FAMPNN's.

    FAMPNN numbers what its parser reads from 1 with no gaps, so a caller's position has to
    be renumbered before it names the same residue to the model. The two diverge after the
    first residue it skips, which would otherwise score the neighbouring residue instead.

    A mutation naming a residue the parser does not read has no counterpart to score, so it
    is dropped with a warning rather than failing the batch. The returned lists stay aligned
    with each other, so the caller can be answered in their own numbering.

    Args:
        mutations (list[str]): Mutation strings, ``'<WT><1-indexed position><MUT>'``, joined
            with ``:`` for simultaneous mutations, in the structure's numbering.
        structure (Structure): The structure the mutations apply to.

    Returns:
        tuple[list[str], list[str]]: The scorable mutations in FAMPNN's numbering, and the
            same mutations as the caller wrote them.

    Raises:
        ValueError: If a mutation is not ``'<WT><position><MUT>'``, optionally ``:``-joined.
            Renumbering only the parts that parse would put two numberings in one string, so
            a malformed part rejects its whole mutation.
    """
    readable = flat_readable_positions(fampnn_readable_positions(structure), structure)
    renumbered: list[str] = []
    kept: list[str] = []
    for mutation in mutations:
        if mutation.lower() == "wt":
            renumbered.append(mutation)
            kept.append(mutation)
            continue
        matches = [re.fullmatch(r"([A-Z])(\d+)([A-Z])", part) for part in mutation.split(":")]
        if any(match is None for match in matches):
            raise ValueError(
                f"fampnn-score: invalid mutation {mutation!r}; expected '<WT><position><MUT>' such as "
                "'A1V', with ':' between simultaneous mutations."
            )
        parts: list[str] = []
        unreadable: list[int] = []
        for wild_type, position_text, mutant in (match.groups() for match in matches if match is not None):
            position = int(position_text)
            model_position = to_model_numbering(readable, [position])
            if not model_position:
                unreadable.append(position)
                continue
            parts.append(f"{wild_type}{model_position[0]}{mutant}")
        if unreadable:
            logger.warning(
                "fampnn-score: mutation %r names position %s, which FAMPNN's structure parser does "
                "not read, so it cannot be scored and is omitted from the results. Models skip "
                "residues they cannot read, such as a chromophore or a residue with no recognized "
                "atom.",
                mutation,
                ", ".join(str(position) for position in unreadable),
            )
            continue
        renumbered.append(":".join(parts))
        kept.append(mutation)
    return renumbered, kept


class FAMPNNScoreInput(BaseToolInput):
    """Input for FAMPNN mutation scoring.

    Attributes:
        inputs (list[MutationInput]): List of MutationInput objects, each containing a structure
            and mutations to score.
    """

    inputs: list[MutationInput] = InputField(
        title="Mutation Inputs",
        description="List of mutation inputs, each with a structure and mutations.",
    )


class FAMPNNScoreConfig(BaseConfig):
    """Configuration for FAMPNN mutation scoring.

    Attributes:
        model_variant (str): Checkpoint variant. '0.3_cath' recommended for scoring.
        batch_size (int): Number of mutations to score simultaneously on GPU.
        seq_only (bool): If True, score without sidechain context (backbone-only).
        scn_diffusion_steps (int): Number of sidechain diffusion denoising steps.
        scn_step_scale (float): Step scale for sidechain diffusion.
        device (str): Device to run on.
    """

    model_variant: str = ConfigField(
        title="Model Variant",
        default="0.3_cath",
        description="FAMPNN checkpoint: '0.3_cath' recommended for scoring",
        examples=["0.3_cath", "0.3", "0.0"],
        reload_on_change=True,
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
    )
    scn_step_scale: float = ConfigField(
        title="Sidechain Step Scale",
        default=1.5,
        gt=0.0,
        description="Step scale (eta) for sidechain diffusion",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on",
        include_in_key=False,
    )


class MutationScoreResult(BaseModel):
    """Mutation score for a single structure.

    Attributes:
        mutations (list[str]): Mutation strings that were scored, as the caller wrote them.
            A requested mutation naming a residue FAMPNN's parser does not read cannot be
            scored and is absent here, so this stays aligned with ``scores``.
        scores (list[float]): Log-likelihood ratio scores for each mutation.
            Positive = mutation is more likely than wild-type.
    """

    model_config = ConfigDict(extra="forbid")

    mutations: list[str] = Field(title="Mutations", description="Mutation strings that were scored")
    scores: list[float] = Field(
        title="Scores",
        description="Log-likelihood ratio scores (positive = favored over wild-type)",
    )


class FAMPNNScoreOutput(BaseToolOutput):
    """Output for FAMPNN mutation scoring.

    Attributes:
        results (list[MutationScoreResult]): List of MutationScoreResult objects, one per input structure.
    """

    results: list[MutationScoreResult] = Field(
        title="Results",
        description="Scoring results, one per input structure",
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
    iterable_input_fields=["inputs"],
    iterable_output_field="results",
    max_chunk_size=32,
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
        model_mutations, scored_mutations = _to_model_numbering(inp.mutations, inp.structure)
        if not model_mutations:
            logger.warning(
                "fampnn-score: none of the %d requested mutation(s) name a residue FAMPNN reads; "
                "returning no scores for this structure.",
                len(inp.mutations),
            )
            results.append(MutationScoreResult(mutations=[], scores=[]))
            continue
        with inp.structure.temp_file() as pdb_path:
            input_dict = {
                "operation": "score_mutations",
                "pdb_path": str(pdb_path),
                "mutations": model_mutations,
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
                mutations=scored_mutations,
                scores=result["scores"],
            )
        )

    return FAMPNNScoreOutput(results=results)
