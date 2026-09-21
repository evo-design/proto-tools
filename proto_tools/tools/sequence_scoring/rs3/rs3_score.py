"""Rule Set 3 sequence-based SpCas9 guide activity scoring."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseConfig, BaseToolInput, BaseToolOutput, ConfigField, InputField, ToolInstance
from proto_tools.utils.tool_io import Metrics, MetricSpec

RS3_VERSION = "0.0.18"
RS3_MODEL = "Rule Set 3 sequence model"


class RS3ScoreInput(BaseToolInput):
    """Thirty-nucleotide SpCas9 target contexts to score.

    Attributes:
        sequences (list[str]): Contexts arranged as 4-nt upstream flank,
            20-nt protospacer, 3-nt PAM, and 3-nt downstream flank.
    """

    sequences: list[str] = InputField(
        title="Target Contexts",
        description="30-nt DNA contexts: 4-nt flank + 20-nt protospacer + 3-nt PAM + 3-nt flank",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def normalize_list(cls, value: Any) -> list[Any]:
        """Normalize a single context to a one-item batch."""
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: list[str]) -> list[str]:
        """Normalize and validate the model's exact context contract."""
        normalized = []
        for index, sequence in enumerate(sequences):
            context = sequence.upper().replace("U", "T")
            if len(context) != 30:
                raise ValueError(f"sequences[{index}] must be exactly 30 nt; got {len(context)}")
            invalid = sorted(set(context) - set("ACGT"))
            if invalid:
                raise ValueError(f"sequences[{index}] contains non-canonical bases: {invalid}")
            if context[25:27] != "GG":
                raise ValueError(f"sequences[{index}] must contain an NGG PAM at positions 25-27")
            normalized.append(context)
        return normalized

    def __len__(self) -> int:
        return len(self.sequences)


class RS3ScoreConfig(BaseConfig):
    """Rule Set 3 sequence-model configuration.

    Attributes:
        tracr_rna (Literal["Hsu2013", "Chen2013"]): tracrRNA architecture used experimentally.
    """

    tracr_rna: Literal["Hsu2013", "Chen2013"] = ConfigField(
        default="Hsu2013",
        title="tracrRNA Architecture",
        description="Rule Set 3 tracrRNA feature set matching the experimental scaffold.",
    )


class RS3Metrics(Metrics):
    """Raw Rule Set 3 activity prediction."""

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "description": "Raw mean-centered Rule Set 3 sequence-model activity score.",
            "better_values_are": "higher",
        }
    }
    primary_metric: str | None = Field(default="score", title="Primary Metric", description="Primary ranking metric")


class RS3ScoreResult(BaseModel):
    """One model-scored target context.

    Attributes:
        sequence (str): Scored 30-nt context.
        protospacer (str): Positions 5-24 of the context.
        pam (str): Positions 25-27 of the context.
        metrics (RS3Metrics): Raw model prediction.
    """

    sequence: str = Field(title="Target Context", description="Scored 30-nt target context")
    protospacer: str = Field(title="Protospacer", description="20-nt guide protospacer")
    pam: str = Field(title="PAM", description="Three-nucleotide NGG PAM")
    metrics: RS3Metrics = Field(title="Metrics", description="Rule Set 3 activity metrics")


class RS3ScoreOutput(BaseToolOutput):
    """Batch of Rule Set 3 activity predictions.

    Attributes:
        results (list[RS3ScoreResult]): Per-context activity predictions.
        model_name (str): Scoring model identity.
        model_version (str): Pinned rs3 package version.
        tracr_rna (str): Scaffold feature set used by the model.
    """

    results: list[RS3ScoreResult] = Field(title="Results", description="Per-context activity predictions")
    model_name: str = Field(title="Model Name", description="Scoring model identity")
    model_version: str = Field(title="Model Version", description="Pinned rs3 package version")
    tracr_rna: str = Field(title="tracrRNA Architecture", description="Scaffold feature set used by the model")

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[RS3ScoreResult]:  # type: ignore[override]
        return iter(self.results)

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Path | str, file_format: str) -> None:
        """Export model results as JSON or a flat CSV table."""
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            path.write_text(json.dumps(self.model_dump(mode="json"), indent=2))
            return
        if file_format == "csv":
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sequence", "protospacer", "pam", "score", "tracr_rna", "model_version"])
                for result in self.results:
                    writer.writerow(
                        [
                            result.sequence,
                            result.protospacer,
                            result.pam,
                            result.metrics.score,
                            self.tracr_rna,
                            self.model_version,
                        ]
                    )
            return
        raise ValueError(f"Unsupported format: {file_format}")


def example_input() -> RS3ScoreInput:
    """Return a documented model input from the RS3 project."""
    return RS3ScoreInput(sequences=["GACGAAAGCGACAACGCGTTCATCCGGGCA"])


@tool(
    key="rs3-score",
    label="Rule Set 3 On-Target Score",
    category="sequence_scoring",
    input_class=RS3ScoreInput,
    config_class=RS3ScoreConfig,
    output_class=RS3ScoreOutput,
    metrics_class=RS3Metrics,
    description="Predict SpCas9 guide activity with the Rule Set 3 sequence model",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=10000,
    cacheable=True,
)
def run_rs3_score(inputs: RS3ScoreInput, config: RS3ScoreConfig, instance: Any = None) -> RS3ScoreOutput:
    """Score SpCas9 target contexts with the published Rule Set 3 model."""
    payload = ToolInstance.dispatch(
        "rs3",
        {"sequences": inputs.sequences, "tracr_rna": config.tracr_rna, "device": "cpu"},
        instance=instance,
        config=config,
    )
    scores = payload["scores"]
    if len(scores) != len(inputs.sequences):
        raise ValueError(f"Expected {len(inputs.sequences)} RS3 scores, got {len(scores)}")
    return RS3ScoreOutput(
        results=[
            RS3ScoreResult(
                sequence=sequence,
                protospacer=sequence[4:24],
                pam=sequence[24:27],
                metrics=RS3Metrics(score=float(score)),
            )
            for sequence, score in zip(inputs.sequences, scores, strict=True)
        ],
        model_name=RS3_MODEL,
        model_version=RS3_VERSION,
        tracr_rna=config.tracr_rna,
    )
