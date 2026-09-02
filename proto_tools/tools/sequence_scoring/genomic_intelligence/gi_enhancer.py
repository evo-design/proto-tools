"""Enhancer-activity prediction via the Genomic Intelligence hosted API.

Reports two activity scores per window, following the STARR-seq split between
developmental and housekeeping enhancer programmes. That split is a *Drosophila*
assay definition; read the scores as relative activity within a comparison
rather than as calibrated cross-species values.
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
    as_object,
    as_object_list,
    build_request_meta,
    call_predict,
    coerce_gi_sequences,
    validate_gi_sequence,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, InputField

ENHANCER_MIN_BP = 50
"""Published ``minLength`` for the enhancer endpoint, in base pairs."""


# ============================================================================
# Data Models
# ============================================================================


class GIEnhancerInput(BaseToolInput):
    """Input for enhancer-activity prediction.

    Attributes:
        sequences (list[GISequence]): Sequences to score. A bare DNA string is
            accepted and coerced. Each must be at least 50 bp, the endpoint's
            published floor.
    """

    sequences: list[GISequence] = InputField(
        title="Sequences",
        description="DNA sequences to score for enhancer activity (>=50 bp each)",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        return coerce_gi_sequences(value)


class GIEnhancerConfig(GIConfig):
    """Configuration for enhancer prediction.

    The endpoint declares no task-specific options, so this adds nothing to the
    shared configuration.

    Attributes:
        gi_api_key (str | None): Bearer key for the hosted API. Defaults to the
            ``GI_API_KEY`` environment variable.
        model (str | None): Model identifier. Leave unset: the service resolves
            the current default for the task.
        respond_async (bool): Request ``202`` + polling instead of a synchronous
            ``200``. A per-request delivery choice available on every endpoint.
        timeout_seconds (float): Wall-clock cap on the async wait.
    """


class EnhancerWindow(BaseModel):
    """One scored window.

    Attributes:
        window_index (int): Position of this window in the slide.
        start (int): 0-based inclusive start in the submitted sequence.
        end (int): 0-based exclusive end in the submitted sequence.
        dev_score (float): Developmental-programme activity score.
        hk_score (float): Housekeeping-programme activity score.
    """

    window_index: int = Field(title="Window Index", description="Position of this window in the slide", ge=0)
    start: int = Field(title="Start", description="0-based inclusive start in the submitted sequence", ge=0)
    end: int = Field(title="End", description="0-based exclusive end in the submitted sequence", ge=0)
    dev_score: float = Field(title="Dev Score", description="Developmental-programme activity score")
    hk_score: float = Field(title="HK Score", description="Housekeeping-programme activity score")

    model_config = ConfigDict(frozen=True)


class GIEnhancerResult(BaseModel):
    """Enhancer prediction for one submitted sequence.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        total_windows (int): Number of windows scored.
        dev_score_max (float | None): Highest developmental score.
        hk_score_max (float | None): Highest housekeeping score.
        windows (list[EnhancerWindow]): Every scored window.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    total_windows: int = Field(title="Total Windows", description="Number of windows scored", ge=0)
    dev_score_max: float | None = Field(title="Dev Score Max", default=None, description="Highest developmental score")
    hk_score_max: float | None = Field(title="HK Score Max", default=None, description="Highest housekeeping score")
    windows: list[EnhancerWindow] = Field(title="Windows", default_factory=list, description="Every scored window")
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GIEnhancerOutput(BaseToolOutput):
    """Output from enhancer prediction.

    Attributes:
        results (list[GIEnhancerResult]): One result per submitted sequence, in
            the order submitted.
    """

    results: list[GIEnhancerResult] = Field(title="Results", description="One result per submitted sequence, in order")

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


def parse_enhancer_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> GIEnhancerResult:
    """Build a :class:`GIEnhancerResult` from one response payload.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        GIEnhancerResult: Parsed result.
    """
    summary = as_object(data.get("summary"), "data.summary")
    windows = [
        EnhancerWindow(
            window_index=int(window.get("window_index", index)),
            start=int(window.get("start", 0)),
            end=int(window.get("end", 0)),
            dev_score=float(window.get("dev_score", 0.0)),
            hk_score=float(window.get("hk_score", 0.0)),
        )
        for index, window in enumerate(as_object_list(data.get("windows"), "data.windows"))
    ]
    return GIEnhancerResult(
        name=name,
        sequence_length=int(as_object(data.get("input"), "data.input").get("sequence_length", 0)),
        total_windows=int(summary.get("total_windows", len(windows))),
        dev_score_max=summary.get("dev_score_max"),
        hk_score_max=summary.get("hk_score_max"),
        windows=windows,
        meta=build_request_meta(payload, data),
    )


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GIEnhancerInput(sequences=[GISequence(sequence="ATGC" * 25, name="example")])


@tool(
    key="gi-enhancer",
    label="GI Enhancer Activity",
    category="sequence_scoring",
    input_class=GIEnhancerInput,
    config_class=GIEnhancerConfig,
    output_class=GIEnhancerOutput,
    description="Predict developmental and housekeeping enhancer activity via the hosted Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-enhancer calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_enhancer(
    inputs: GIEnhancerInput,
    config: GIEnhancerConfig,
    instance: Any = None,
) -> GIEnhancerOutput:
    """Predict enhancer activity for each submitted sequence.

    Args:
        inputs (GIEnhancerInput): Sequences to score.
        config (GIEnhancerConfig): API credentials and model selection.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GIEnhancerOutput: One result per submitted sequence, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API, and on a 2xx whose
            body is not a ``{data, meta}`` envelope.
        GIResponseShapeError: If a field inside ``data`` documented as an
            object or an array arrives as something else.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    results: list[GIEnhancerResult] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=ENHANCER_MIN_BP, task="enhancer")
        data, payload = call_predict(config, "enhancer", sequence, item.name)
        results.append(parse_enhancer_data(data, payload, item.name))
    return GIEnhancerOutput(results=results)
