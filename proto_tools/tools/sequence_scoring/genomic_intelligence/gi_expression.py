"""Gene-expression prediction via the Genomic Intelligence hosted API.

The model scores exactly one 9,198 bp window centred on a transcription start
site. Either submit that window directly, or submit a longer locus together with
``tss_index`` and let the service cut it; the window actually scored is echoed
back. Under-length input is rejected rather than padded.

Two properties are easy to get wrong:

* ``description`` is conditioning text, not a label. It is fed to the model, so
  its wording changes the predicted value — hold it fixed when comparing runs.
* The sequence is never reverse-complemented. Submit minus-strand genes in
  transcript orientation; the opposite strand returns a confident, wrong number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proto_tools.tools.sequence_scoring.genomic_intelligence.shared_data_models import (
    GIConfig,
    GIRequestMeta,
    build_request_meta,
    call_predict,
    validate_gi_sequence,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField

EXPRESSION_WINDOW_BP = 9_198
"""Width of the single window the model scores, and the endpoint's floor."""

EXPRESSION_TSS_RADIUS = EXPRESSION_WINDOW_BP // 2
"""Half-width of the scored window: 4,599 bp either side of the TSS."""


# ============================================================================
# Data Models
# ============================================================================


class ExpressionSequence(BaseToolInput):
    """One locus to score, with the TSS located inside it.

    Attributes:
        sequence (str): Either exactly 9,198 bp centred on the TSS, or a longer
            locus paired with ``tss_index``.
        name (str): Label echoed back in the response.
        tss_index (int | None): 0-based TSS offset into the whitespace-stripped
            sequence. Required unless the sequence is exactly 9,198 bp, where it
            defaults to the midpoint. Must satisfy
            ``4599 <= tss_index <= len(sequence) - 4599``.
    """

    sequence: str = InputField(title="Sequence", description="TSS-centred 9,198 bp window, or a longer locus")
    name: str = InputField(default="sequence", title="Name", description="Label echoed back in the response")
    tss_index: int | None = InputField(
        default=None,
        ge=0,
        title="TSS Index",
        description="0-based TSS offset; required unless the sequence is exactly 9,198 bp",
    )

    @model_validator(mode="after")
    def _tss_index_present_and_in_range(self) -> ExpressionSequence:
        """Require a TSS offset for any locus that is not exactly one window.

        A wrongly-placed window still scores and returns a confident number, so
        the offset is checked here rather than left to the service.
        """
        length = len(self.sequence.strip())
        if self.tss_index is None:
            if length != EXPRESSION_WINDOW_BP:
                raise ValueError(
                    f"tss_index is required unless the sequence is exactly {EXPRESSION_WINDOW_BP:,} bp "
                    f"(got {length:,} bp). It is the 0-based offset of the TSS into the sequence."
                )
            return self
        low = EXPRESSION_TSS_RADIUS
        high = length - EXPRESSION_TSS_RADIUS
        if not low <= self.tss_index <= high:
            raise ValueError(
                f"tss_index {self.tss_index:,} is outside [{low:,}, {high:,}] for a {length:,} bp sequence; "
                f"the model needs a full +/-{EXPRESSION_TSS_RADIUS:,} bp around the TSS. Submit more flanking sequence."
            )
        return self


class GIExpressionInput(BaseToolInput):
    """Input for expression prediction.

    Attributes:
        sequences (list[ExpressionSequence]): Loci to score, each carrying its
            own TSS offset where needed.
    """

    sequences: list[ExpressionSequence] = InputField(
        title="Sequences",
        description="Loci to score, each a 9,198 bp TSS window or a longer locus plus tss_index",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        if isinstance(value, (str, dict, ExpressionSequence)):
            value = [value]
        if not isinstance(value, list):
            return value
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, (ExpressionSequence, dict)):
                coerced.append(item)
            elif isinstance(item, str):
                coerced.append(ExpressionSequence(sequence=item))
            else:
                raise ValueError(f"each sequence must be a str, dict, or ExpressionSequence, got {type(item).__name__}")
        return coerced


class GIExpressionConfig(GIConfig):
    """Configuration for expression prediction.

    Attributes:
        description (str): Experimental context the model is conditioned on —
            cell type, assay, conditions. Required. This is model input, not a
            label: rewording it changes the prediction, so keep it identical
            across runs you intend to compare.
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

    description: str = ConfigField(
        title="Description",
        default="assay term name is polyA plus RNA-seq. biosample summary is Homo sapiens K562.",
        min_length=1,
        description="Experimental context fed to the model; wording changes the prediction",
    )


class ExpressionPrediction(BaseModel):
    """Predicted expression for one locus.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        expression_log_tpm (float | None): Predicted log(TPM+1).
        expression_tpm (float | None): Predicted TPM.
        tss_index (int | None): TSS offset the service applied.
        scored_window (list[int] | None): Window the service actually scored,
            as ``[start, end)`` in the submitted sequence.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    expression_log_tpm: float | None = Field(
        title="Expression Log TPM", default=None, description="Predicted log(TPM+1)"
    )
    expression_tpm: float | None = Field(title="Expression TPM", default=None, description="Predicted TPM")
    tss_index: int | None = Field(title="TSS Index", default=None, description="TSS offset the service applied")
    scored_window: list[int] | None = Field(
        title="Scored Window", default=None, description="Window actually scored, as [start, end)"
    )
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GIExpressionOutput(BaseToolOutput):
    """Output from expression prediction.

    Attributes:
        results (list[ExpressionPrediction]): One result per submitted locus,
            in the order submitted.
    """

    results: list[ExpressionPrediction] = Field(title="Results", description="One result per submitted locus, in order")

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


def parse_expression_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> ExpressionPrediction:
    """Build an :class:`ExpressionPrediction` from one response payload.

    The applied window is read back from ``meta.task_specific_counts`` rather
    than assumed from the request, because an in-range but wrong ``tss_index``
    scores a different window and still returns ``200``.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        ExpressionPrediction: Parsed result.
    """
    prediction = data.get("prediction") or {}
    counts = (payload.get("meta") or {}).get("task_specific_counts") or {}
    window = counts.get("scored_window")
    return ExpressionPrediction(
        name=name,
        sequence_length=int((data.get("input") or {}).get("sequence_length", 0)),
        expression_log_tpm=prediction.get("expression_log_tpm"),
        expression_tpm=prediction.get("expression_tpm"),
        tss_index=counts.get("tss_index"),
        scored_window=[int(value) for value in window] if isinstance(window, (list, tuple)) else None,
        meta=build_request_meta(payload, data),
    )


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GIExpressionInput(
        sequences=[ExpressionSequence(sequence="ATGC" * 2299 + "AC", name="example")],
    )


@tool(
    key="gi-expression",
    label="GI Gene Expression",
    category="sequence_scoring",
    input_class=GIExpressionInput,
    config_class=GIExpressionConfig,
    output_class=GIExpressionOutput,
    description="Predict gene expression from a TSS-centred window via the hosted Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-expression calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_expression(
    inputs: GIExpressionInput,
    config: GIExpressionConfig,
    instance: Any = None,
) -> GIExpressionOutput:
    """Predict expression for each submitted locus.

    Args:
        inputs (GIExpressionInput): Loci to score.
        config (GIExpressionConfig): API credentials, model selection, and the
            conditioning description.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GIExpressionOutput: One result per submitted locus, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    results: list[ExpressionPrediction] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=EXPRESSION_WINDOW_BP, task="expression")
        extra = {"tss_index": item.tss_index} if item.tss_index is not None else None
        data, payload = call_predict(
            config,
            "expression",
            sequence,
            item.name,
            options={"description": config.description},
            extra_body=extra,
        )
        results.append(parse_expression_data(data, payload, item.name))
    return GIExpressionOutput(results=results)
