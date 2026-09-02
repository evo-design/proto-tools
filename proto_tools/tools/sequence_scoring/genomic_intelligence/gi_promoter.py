"""Promoter-region prediction via the Genomic Intelligence hosted API.

Slides a DNA language model across the submitted sequence and reports the
windows it calls as promoters, with the per-window probabilities behind them.
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
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField

PROMOTER_MIN_BP = 300
"""Published ``minLength`` for the promoter endpoint, in base pairs."""


# ============================================================================
# Data Models
# ============================================================================


class GIPromoterInput(BaseToolInput):
    """Input for promoter-region prediction.

    Attributes:
        sequences (list[GISequence]): Sequences to score. A bare DNA string is
            accepted and coerced. Each must be at least 300 bp, the endpoint's
            published floor.
    """

    sequences: list[GISequence] = InputField(
        title="Sequences",
        description="DNA sequences to score for promoter activity (>=300 bp each)",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        return coerce_gi_sequences(value)


class GIPromoterConfig(GIConfig):
    """Configuration for promoter prediction.

    Attributes:
        threshold (float): Probability above which a window is called a
            promoter. Applied server-side.
        gi_api_key (str | None): Bearer key for the hosted API. Defaults to the
            ``GI_API_KEY`` environment variable.
        model (str | None): Model identifier. Leave unset: the service resolves
            the current default for the task.
        respond_async (bool): Request ``202`` + polling instead of a synchronous
            ``200``. A per-request delivery choice available on every endpoint.
        timeout_seconds (float): Wall-clock cap on the async wait.
    """

    threshold: float = ConfigField(
        title="Threshold",
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability above which a window is reported as a promoter",
    )


class PromoterRegion(BaseModel):
    """One contiguous region called as a promoter.

    Attributes:
        start (int): 0-based inclusive start in the submitted sequence.
        end (int): 0-based exclusive end in the submitted sequence.
        score (float): Model probability for the region.
        name (str): Service-assigned region label.
    """

    start: int = Field(title="Start", description="0-based inclusive start in the submitted sequence", ge=0)
    end: int = Field(title="End", description="0-based exclusive end in the submitted sequence", ge=0)
    score: float = Field(title="Score", description="Model probability for the region")
    name: str = Field(title="Name", default="", description="Service-assigned region label")

    model_config = ConfigDict(frozen=True)


class PromoterWindow(BaseModel):
    """One scored sliding window, whether or not it passed the threshold.

    Attributes:
        window_index (int): Position of this window in the slide.
        start (int): 0-based inclusive start of the scored span.
        end (int): 0-based exclusive end of the scored span.
        probability (float): Model probability for the window.
        is_positive (bool): Whether the probability cleared the threshold.
    """

    window_index: int = Field(title="Window Index", description="Position of this window in the slide", ge=0)
    start: int = Field(title="Start", description="0-based inclusive start of the scored span", ge=0)
    end: int = Field(title="End", description="0-based exclusive end of the scored span", ge=0)
    probability: float = Field(title="Probability", description="Model probability for the window")
    is_positive: bool = Field(title="Is Positive", description="Whether the probability cleared the threshold")

    model_config = ConfigDict(frozen=True)


class GIPromoterResult(BaseModel):
    """Promoter prediction for one submitted sequence.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        promoter_windows (int): Number of windows that cleared the threshold.
        total_windows (int): Number of windows scored.
        max_probability (float | None): Highest window probability, or None
            when no window was scored.
        regions (list[PromoterRegion]): Regions called as promoters.
        windows (list[PromoterWindow]): Every scored window.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    promoter_windows: int = Field(
        title="Promoter Windows", description="Number of windows that cleared the threshold", ge=0
    )
    total_windows: int = Field(title="Total Windows", description="Number of windows scored", ge=0)
    max_probability: float | None = Field(
        title="Max Probability", default=None, description="Highest window probability"
    )
    regions: list[PromoterRegion] = Field(
        title="Regions", default_factory=list, description="Regions called as promoters"
    )
    windows: list[PromoterWindow] = Field(title="Windows", default_factory=list, description="Every scored window")
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GIPromoterOutput(BaseToolOutput):
    """Output from promoter prediction.

    Attributes:
        results (list[GIPromoterResult]): One result per submitted sequence,
            in the order submitted.
    """

    results: list[GIPromoterResult] = Field(title="Results", description="One result per submitted sequence, in order")

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


def parse_promoter_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> GIPromoterResult:
    """Build a :class:`GIPromoterResult` from one response payload.

    Split out from the request path so the shape can be tested without a
    network call.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        GIPromoterResult: Parsed result.
    """
    summary = as_object(data.get("summary"), "data.summary")
    raw_windows = as_object_list(data.get("window_details"), "data.window_details")
    windows = [
        PromoterWindow(
            window_index=int(window.get("window_index", index)),
            start=int(window.get("prediction_start", 0)),
            end=int(window.get("prediction_end", 0)),
            probability=float(window.get("probability", 0.0)),
            is_positive=bool(window.get("is_positive", False)),
        )
        for index, window in enumerate(raw_windows)
    ]
    regions = [
        PromoterRegion(
            start=int(region.get("start", 0)),
            end=int(region.get("end", 0)),
            score=float(region.get("score", 0.0)),
            name=str(region.get("name", "")),
        )
        for region in as_object_list(data.get("regions"), "data.regions")
    ]
    meta = as_object(payload.get("meta"), "meta")
    return GIPromoterResult(
        name=name,
        sequence_length=int(meta.get("sequence_length", 0)),
        promoter_windows=int(summary.get("promoter_windows", 0)),
        total_windows=int(summary.get("total_windows", len(windows))),
        max_probability=max((window.probability for window in windows), default=None),
        regions=regions,
        windows=windows,
        meta=build_request_meta(payload, data),
    )


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GIPromoterInput(sequences=[GISequence(sequence="ATGC" * 100, name="example")])


@tool(
    key="gi-promoter",
    label="GI Promoter",
    category="sequence_scoring",
    input_class=GIPromoterInput,
    config_class=GIPromoterConfig,
    output_class=GIPromoterOutput,
    description="Predict promoter regions in DNA via the hosted Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-promoter calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_promoter(
    inputs: GIPromoterInput,
    config: GIPromoterConfig,
    instance: Any = None,
) -> GIPromoterOutput:
    """Predict promoter regions for each submitted sequence.

    Args:
        inputs (GIPromoterInput): Sequences to score.
        config (GIPromoterConfig): API credentials, model selection, threshold.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GIPromoterOutput: One result per submitted sequence, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API, and on a 2xx whose
            body is not a ``{data, meta}`` envelope.
        GIResponseShapeError: If a field inside ``data`` documented as an
            object or an array arrives as something else.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    results: list[GIPromoterResult] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=PROMOTER_MIN_BP, task="promoter")
        data, payload = call_predict(
            config,
            "promoter",
            sequence,
            item.name,
            options={"threshold": config.threshold},
        )
        results.append(parse_promoter_data(data, payload, item.name))
    return GIPromoterOutput(results=results)
