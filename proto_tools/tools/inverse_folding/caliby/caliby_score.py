"""proto_tools/tools/inverse_folding/caliby/caliby_score.py.

Caliby Potts-energy scoring of a structure's native sequence.
"""

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from proto_tools.entities.structures import Structure, StructureInputBase
from proto_tools.tools.inverse_folding.caliby.caliby_sample import (
    DESIGN_CHECKPOINTS,
    ensemble_paths,
    validate_checkpoint,
    validate_conformer_alignment,
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
from proto_tools.utils.device import RemoteDevice
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = logging.getLogger(__name__)


class CalibyScoreStructureInput(StructureInputBase):
    """A structure whose native sequence is scored, with an optional conformer ensemble.

    Caliby scores the sequence carried by ``structure`` itself; there is no separate
    sequence field. With an ensemble, the primary conformer's sequence is scored against
    the whole set of backbones and the other members' sequences are ignored.

    Attributes:
        structure (Structure): Primary conformer, whose native sequence is scored.
            Accepts a file path, raw PDB/CIF content string, ``Structure`` object, or a
            dict in the shape produced by ``Structure.model_dump(mode='json')``.
        conformers (list[Structure]): Additional conformers of the same molecule. Each
            must have the same chain IDs and per-chain residue counts as ``structure``.
            Empty means single-structure scoring.
    """

    conformers: list[Structure] = Field(
        default_factory=list,
        title="Conformers",
        description="Additional conformers of the same molecule; empty = single-structure scoring.",
    )

    @model_validator(mode="after")
    def _validate_conformer_alignment(self) -> "CalibyScoreStructureInput":
        """Require every conformer to align with the primary conformer."""
        validate_conformer_alignment(self.structure, self.conformers)
        return self

    @property
    def ensemble_structures(self) -> list[Structure]:
        """The full ensemble in Caliby order: the primary conformer first, then ``conformers``."""
        return [self.structure, *self.conformers]


class CalibyScoreInput(BaseToolInput):
    """Input for Caliby Potts-energy scoring.

    Attributes:
        inputs (list[CalibyScoreStructureInput]): Per-structure inputs, each a structure
            whose native sequence is scored plus optional ensemble members.
    """

    inputs: list[CalibyScoreStructureInput] = InputField(
        title="Structure Inputs",
        description="Per-structure inputs, each scored on its own native sequence.",
    )


class CalibyScoreConfig(BaseConfig):
    """Configuration for Caliby Potts-energy scoring.

    Attributes:
        model_checkpoint (str): Bundled design checkpoint name or a path to a ``.ckpt``
            file. Scoring uses the same models as design.
        return_local_energies (bool): Include the per-position local conditional energy
            matrix in the output. Adds roughly ``seq_len x 22`` floats per structure.
        device (str): Device to run the model on.
    """

    model_checkpoint: str = ConfigField(
        title="Model Checkpoint",
        default="caliby",
        description="Bundled design checkpoint name, or a path to a custom .ckpt file.",
        examples=["caliby", "soluble_caliby"],
        reload_on_change=True,
    )
    return_local_energies: bool = ConfigField(
        title="Return Local Energies",
        default=False,
        description="Include the per-position local conditional energy matrix in the output.",
    )
    device: str = ConfigField(
        title="Device",
        default="cuda",
        description="Device to run the model on, e.g. 'cuda', 'cuda:0' or 'cpu'.",
        include_in_key=False,
        examples=["cuda", "cpu"],
    )

    @field_validator("model_checkpoint")
    @classmethod
    def validate_model_checkpoint(cls, value: str) -> str:
        """Reject a sidechain-packing checkpoint; scoring needs a ``task='seq_des'`` model."""
        return validate_checkpoint(value, DESIGN_CHECKPOINTS, "scoring")

    def remote_unsupported_reason(self, device: RemoteDevice) -> str | None:
        """A custom ``.ckpt`` path is not present on a hosted worker."""
        if self.model_checkpoint.endswith(".ckpt"):
            return (
                f"model_checkpoint points to a local .ckpt file not available on device='{device}'. "
                "Use a bundled checkpoint name, or run locally with device='cuda'."
            )
        return None


class CalibyScoringMetrics(Metrics):
    """Potts-energy metrics for one scored sequence-structure pair.

    Caliby reports an unnormalized Potts energy rather than a log-likelihood, so there is
    no perplexity to report: probability is proportional to ``exp(-U/T)`` for a sampling
    temperature ``T`` that scoring never fixes.

    Metrics documented in ``metric_spec``:
        potts_energy (float): Potts energy of the structure's native sequence, summed
            over positions. Lower is more favorable. Scales with sequence length.
            Always present.
        avg_potts_energy (float): ``potts_energy`` divided by sequence length, for
            comparison across structures of different sizes. Always present.

    Attributes:
        sequence (str): The scored sequence, with ``':'`` separating chains in
            alphabetical chain order.
        local_conditional_energies (list[list[float]] | None): Per-position local
            conditional energies, shape ``(seq_len, len(potts_tokens))``. ``None``
            unless ``return_local_energies`` is set.
        potts_tokens (list[str] | None): Token ordering for
            ``local_conditional_energies``. ``None`` when those are absent.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "potts_energy": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
        "avg_potts_energy": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = Field(
        default="potts_energy",
        title="Primary Metric",
        description="Headline metric used to rank results.",
    )

    sequence: str = Field(
        default="",
        title="Sequence",
        description="The scored sequence, with ':' separating chains in alphabetical order.",
    )
    local_conditional_energies: list[list[float]] | None = Field(
        default=None,
        title="Local Energies",
        description="Per-position local conditional energies (seq_len, len(potts_tokens)).",
    )
    potts_tokens: list[str] | None = Field(
        default=None,
        title="Potts Tokens",
        description="Token ordering for local_conditional_energies.",
    )


class CalibyScoreOutput(BaseToolOutput):
    """Output of the Caliby scoring tool.

    Attributes:
        scores (list[CalibyScoringMetrics]): One entry per input structure, in input
            order.
    """

    scores: list[CalibyScoringMetrics] = Field(
        title="Scores",
        description="Potts-energy scores, one per input structure, in input order.",
    )

    @property
    def potts_tokens(self) -> list[str] | None:
        """Token ordering for the local energy matrices; derived from the first score."""
        return self.scores[0].potts_tokens if self.scores else None

    def __len__(self) -> int:
        """Get the number of scored structures."""
        return len(self.scores)

    def __getitem__(self, index: int) -> CalibyScoringMetrics:
        """Get the score for a structure by index."""
        return self.scores[index]

    def __iter__(self) -> Iterator[CalibyScoringMetrics]:  # type: ignore[override]
        """Iterate over the per-structure scores."""
        return iter(self.scores)

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["csv", "json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "csv"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")

        if file_format == "json":
            data = []
            for score in self.scores:
                entry: dict[str, Any] = dict(score.items())
                entry["sequence"] = score.sequence
                if score.local_conditional_energies is not None:
                    entry["local_conditional_energies"] = score.local_conditional_energies
                    entry["potts_tokens"] = score.potts_tokens
                data.append(entry)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        elif file_format == "csv":
            if self.scores:
                # The local energy matrix is 2-D, so CSV carries the scalars plus the sequence.
                fieldnames = ["sequence", *self.scores[0].keys()]
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for score in self.scores:
                        writer.writerow({"sequence": score.sequence, **dict(score.items())})
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return CalibyScoreInput(
        inputs=[
            CalibyScoreStructureInput(
                structure=str(Path(__file__).parents[1] / "example_input_fixture.pdb"),  # type: ignore[arg-type]
            )
        ]
    )


@tool(
    key="caliby-score",
    label="Caliby Scoring",
    category="inverse_folding",
    input_class=CalibyScoreInput,
    config_class=CalibyScoreConfig,
    output_class=CalibyScoreOutput,
    description="Score a structure's native sequence under the Caliby Potts model",
    uses_gpu=True,
    example_input=example_input,
    iterable_input_fields=["inputs"],
    iterable_output_field="scores",
    max_chunk_size=32,
    cacheable=True,
    metrics_class=CalibyScoringMetrics,
)
def run_caliby_score(
    inputs: CalibyScoreInput,
    config: CalibyScoreConfig,
    instance: Any = None,
) -> CalibyScoreOutput:
    """Score a structure's native sequence under the Caliby Potts model.

    Computes Potts parameters from the input backbone and evaluates the energy of the
    sequence the structure already carries. When an input supplies ``conformers``, the
    parameters are aggregated across the ensemble and the primary conformer's sequence is
    the one scored.

    Args:
        inputs (CalibyScoreInput): Structure inputs with optional conformers.
        config (CalibyScoreConfig): Scoring configuration.
        instance (Any): Optional ToolInstance for persistent execution.

    Returns:
        CalibyScoreOutput: One ``CalibyScoringMetrics`` per input structure, in input order.

    Examples:
        >>> inputs = CalibyScoreInput(inputs=[CalibyScoreStructureInput(structure="protein.pdb")])
        >>> output = run_caliby_score(inputs)
        >>> output[0]["potts_energy"]
    """
    scores: list[CalibyScoringMetrics] = []

    logger.debug("Using local venv for caliby scoring")

    for inp in progress_bar(
        inputs.inputs,
        desc="Caliby scoring",
        unit="structure",
        disable=not config.verbose,
    ):
        with ensemble_paths(inp.ensemble_structures) as (primary_path, conformer_paths):
            input_dict = {
                "operation": "score",
                "primary_path": str(primary_path),
                "conformer_paths": [str(p) for p in conformer_paths],
                "return_local_energies": config.return_local_energies,
                "model_checkpoint": config.model_checkpoint,
                "device": config.device,
            }
            result = ToolInstance.dispatch("caliby", input_dict, instance=instance, config=config)

        sequence = result["sequence"]
        energy = result["potts_energy"]
        # Chain separators are not scored positions, so they are excluded from the mean.
        sequence_length = len(sequence.replace(":", ""))
        scores.append(
            CalibyScoringMetrics(
                potts_energy=energy,
                avg_potts_energy=energy / sequence_length if sequence_length else 0.0,
                sequence=sequence,
                local_conditional_energies=result.get("local_conditional_energies"),
                potts_tokens=result.get("potts_tokens"),
            )
        )

    return CalibyScoreOutput(scores=scores)
