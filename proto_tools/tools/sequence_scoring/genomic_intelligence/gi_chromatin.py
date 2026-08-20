"""Chromatin-state prediction via the Genomic Intelligence hosted API.

Scores each window against a large panel of chromatin assays — accessibility,
transcription-factor occupancy and histone marks — across many cell types, and
returns how many calls clear the threshold, per window and per assay category.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from proto_tools.tools.sequence_scoring.genomic_intelligence.shared_data_models import (
    GIConfig,
    GIRequestMeta,
    GISequence,
    build_request_meta,
    call_predict,
    coerce_gi_sequences,
    validate_gi_sequence,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField

CHROMATIN_MIN_BP = 200
"""Published ``minLength`` for the chromatin endpoint, in base pairs."""


# ============================================================================
# Data Models
# ============================================================================


class GIChromatinInput(BaseToolInput):
    """Input for chromatin-state prediction.

    Attributes:
        sequences (list[GISequence]): Sequences to score. A bare DNA string is
            accepted and coerced. Each must be at least 200 bp, the endpoint's
            published floor.
    """

    sequences: list[GISequence] = InputField(
        title="Sequences",
        description="DNA sequences to score for chromatin state (>=200 bp each)",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        return coerce_gi_sequences(value)


class GIChromatinConfig(GIConfig):
    """Configuration for chromatin prediction.

    Attributes:
        threshold (float): Probability above which an assay call is reported.
            The panel is large, so lowering this materially increases response
            size.
        gi_api_key (str | None): Bearer key for the hosted API. Defaults to the
            ``GI_API_KEY`` environment variable.
        base_url (str): API root. Override only to target a non-production
            deployment.
        model (str | None): Model identifier. Leave unset: the service resolves
            the current default for the task.
        respond_async (bool): Request ``202`` + polling instead of a synchronous
            ``200``. A per-request delivery choice available on every endpoint.
        poll_interval_seconds (float): Delay between job polls when
            ``respond_async`` is set.
        timeout_seconds (float): Wall-clock cap on the async wait.
    """

    threshold: float = ConfigField(
        title="Threshold",
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability above which an assay call is reported",
    )


class ChromatinWindow(BaseModel):
    """One scored window.

    Attributes:
        window_index (int): Position of this window in the slide.
        start (int): 0-based inclusive start in the submitted sequence.
        end (int): 0-based exclusive end in the submitted sequence.
        annotation_count (int): Number of assay calls above the threshold.
    """

    window_index: int = Field(title="Window Index", description="Position of this window in the slide", ge=0)
    start: int = Field(title="Start", description="0-based inclusive start in the submitted sequence", ge=0)
    end: int = Field(title="End", description="0-based exclusive end in the submitted sequence", ge=0)
    annotation_count: int = Field(
        title="Annotation Count", description="Number of assay calls above the threshold", ge=0
    )

    model_config = ConfigDict(frozen=True)


class GIChromatinResult(BaseModel):
    """Chromatin prediction for one submitted sequence.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        total_windows (int): Number of windows scored.
        total_annotations (int): Assay calls above the threshold, summed.
        category_counts (dict[str, int]): Calls per assay category.
        windows (list[ChromatinWindow]): Per-window call counts.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    total_windows: int = Field(title="Total Windows", description="Number of windows scored", ge=0)
    total_annotations: int = Field(
        title="Total Annotations", description="Assay calls above the threshold, summed", ge=0
    )
    category_counts: dict[str, int] = Field(
        title="Category Counts", default_factory=dict, description="Calls per assay category"
    )
    windows: list[ChromatinWindow] = Field(title="Windows", default_factory=list, description="Per-window call counts")
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GIChromatinOutput(BaseToolOutput):
    """Output from chromatin prediction.

    Attributes:
        results (list[GIChromatinResult]): One result per submitted sequence,
            in the order submitted.
    """

    results: list[GIChromatinResult] = Field(title="Results", description="One result per submitted sequence, in order")

    @property
    def output_format_options(self) -> list[str]:
        """Return the supported output format options."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Parsing
# ============================================================================


def parse_chromatin_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> GIChromatinResult:
    """Build a :class:`GIChromatinResult` from one response payload.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        GIChromatinResult: Parsed result.
    """
    summary = data.get("summary") or {}
    windows = [
        ChromatinWindow(
            window_index=int(window.get("window_index", index)),
            start=int(window.get("start", 0)),
            end=int(window.get("end", 0)),
            annotation_count=int(window.get("annotation_count", 0)),
        )
        for index, window in enumerate(data.get("windows") or [])
    ]
    raw_counts = summary.get("category_counts") or {}
    return GIChromatinResult(
        name=name,
        sequence_length=int((data.get("input") or {}).get("sequence_length", 0)),
        total_windows=int(summary.get("total_windows", len(windows))),
        total_annotations=int(summary.get("total_annotations", 0)),
        category_counts={str(key): int(value) for key, value in raw_counts.items()},
        windows=windows,
        meta=build_request_meta(payload, data),
    )


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GIChromatinInput(sequences=[GISequence(sequence="ATGC" * 75, name="example")])


@tool(
    key="gi-chromatin",
    label="GI Chromatin State",
    category="sequence_scoring",
    input_class=GIChromatinInput,
    config_class=GIChromatinConfig,
    output_class=GIChromatinOutput,
    description="Predict chromatin accessibility, TF occupancy and histone marks via the hosted "
    "Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-chromatin calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_chromatin(
    inputs: GIChromatinInput,
    config: GIChromatinConfig,
    instance: Any = None,
) -> GIChromatinOutput:
    """Predict chromatin state for each submitted sequence.

    Args:
        inputs (GIChromatinInput): Sequences to score.
        config (GIChromatinConfig): API credentials, model selection, threshold.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GIChromatinOutput: One result per submitted sequence, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    results: list[GIChromatinResult] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=CHROMATIN_MIN_BP, task="chromatin")
        data, payload = call_predict(
            config,
            "chromatin",
            sequence,
            item.name,
            options={"threshold": config.threshold},
        )
        results.append(parse_chromatin_data(data, payload, item.name))
    return GIChromatinOutput(results=results)
