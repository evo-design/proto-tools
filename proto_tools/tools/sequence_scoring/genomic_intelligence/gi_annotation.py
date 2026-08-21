"""De-novo gene and transcript annotation via the Genomic Intelligence hosted API.

Finds transcripts in raw sequence without a reference, returning each
transcript's bounds, strand, confidence score, and TSS and poly(A) positions.
Detection is strand-insensitive: genes on either strand are found from a single
submission, and the reported ``strand`` is relative to the sequence as
submitted.

Annotation is the slowest of the tasks. Setting ``respond_async`` on the config
returns a job id immediately and polls it, which avoids holding a long HTTP
request open. That is a delivery preference rather than a requirement — the
endpoint answers synchronously too.
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

ANNOTATION_MIN_BP = 1_000
"""Published ``minLength`` for the annotation endpoint, in base pairs."""


# ============================================================================
# Data Models
# ============================================================================


class GIAnnotationInput(BaseToolInput):
    """Input for de-novo transcript annotation.

    Attributes:
        sequences (list[GISequence]): Sequences to annotate. A bare DNA string
            is accepted and coerced. Each must be at least 1,000 bp, the
            endpoint's published floor.
    """

    sequences: list[GISequence] = InputField(
        title="Sequences",
        description="DNA sequences to annotate (>=1,000 bp each)",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        return coerce_gi_sequences(value)


class GIAnnotationConfig(GIConfig):
    """Configuration for annotation.

    Attributes:
        batch_size (int | None): Server-side batching hint. Leave unset for the
            service default.
        reverse_complement (bool | None): Also scan the reverse complement.
            Detection already finds genes on either strand, so this changes the
            reported orientation rather than whether genes are found.
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

    batch_size: int | None = ConfigField(
        title="Batch Size",
        default=None,
        ge=1,
        description="Server-side batching hint; unset uses the service default",
    )
    reverse_complement: bool | None = ConfigField(
        title="Reverse Complement",
        default=None,
        description="Also scan the reverse complement; unset uses the service default",
    )


class Transcript(BaseModel):
    """One predicted transcript.

    Attributes:
        name (str): Service-assigned transcript label.
        start (int): 0-based inclusive start in the submitted sequence.
        end (int): 0-based exclusive end in the submitted sequence.
        strand (str): Orientation relative to the submitted sequence.
        score (float): Model confidence for the transcript.
        tss_position (int | None): Transcription start site offset.
        polya_position (int | None): Poly(A) site offset.
        transcript_type (str | None): Biotype, when the model reports one.
    """

    name: str = Field(title="Name", default="", description="Service-assigned transcript label")
    start: int = Field(title="Start", description="0-based inclusive start in the submitted sequence", ge=0)
    end: int = Field(title="End", description="0-based exclusive end in the submitted sequence", ge=0)
    strand: str = Field(title="Strand", default="", description="Orientation relative to the submitted sequence")
    score: float = Field(title="Score", default=0.0, description="Model confidence for the transcript")
    tss_position: int | None = Field(title="TSS Position", default=None, description="Transcription start site offset")
    polya_position: int | None = Field(title="Polya Position", default=None, description="Poly(A) site offset")
    transcript_type: str | None = Field(
        title="Transcript Type", default=None, description="Biotype, when the model reports one"
    )

    model_config = ConfigDict(frozen=True)


class GIAnnotationResult(BaseModel):
    """Annotation for one submitted sequence.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        total_transcripts (int): Number of transcripts found.
        forward_strand (int): Transcripts on the submitted orientation.
        reverse_strand (int): Transcripts on the opposite orientation.
        transcripts (list[Transcript]): The transcripts found.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    total_transcripts: int = Field(title="Total Transcripts", description="Number of transcripts found", ge=0)
    forward_strand: int = Field(
        title="Forward Strand", default=0, description="Transcripts on the submitted orientation", ge=0
    )
    reverse_strand: int = Field(
        title="Reverse Strand", default=0, description="Transcripts on the opposite orientation", ge=0
    )
    transcripts: list[Transcript] = Field(
        title="Transcripts", default_factory=list, description="The transcripts found"
    )
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GIAnnotationOutput(BaseToolOutput):
    """Output from annotation.

    Attributes:
        results (list[GIAnnotationResult]): One result per submitted sequence,
            in the order submitted.
    """

    results: list[GIAnnotationResult] = Field(
        title="Results", description="One result per submitted sequence, in order"
    )

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


def parse_annotation_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> GIAnnotationResult:
    """Build a :class:`GIAnnotationResult` from one response payload.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        GIAnnotationResult: Parsed result.
    """
    summary = as_object(data.get("summary"), "data.summary")
    transcripts = [
        Transcript(
            name=str(record.get("name", "")),
            start=int(record.get("start", 0)),
            end=int(record.get("end", 0)),
            strand=str(record.get("strand", "")),
            score=float(record.get("score", 0.0)),
            tss_position=record.get("tss_position"),
            polya_position=record.get("polya_position"),
            transcript_type=record.get("transcript_type"),
        )
        for record in as_object_list(data.get("transcripts"), "data.transcripts")
    ]
    return GIAnnotationResult(
        name=name,
        sequence_length=int(as_object(data.get("input"), "data.input").get("sequence_length", 0)),
        total_transcripts=int(summary.get("total_transcripts", len(transcripts))),
        forward_strand=int(summary.get("forward_strand", 0)),
        reverse_strand=int(summary.get("reverse_strand", 0)),
        transcripts=transcripts,
        meta=build_request_meta(payload, data),
    )


def _annotation_options(config: GIAnnotationConfig) -> dict[str, Any]:
    """Build the task's closed options object, omitting unset members.

    Args:
        config (GIAnnotationConfig): Tool configuration.

    Returns:
        dict[str, Any]: Options to send with the request.
    """
    options: dict[str, Any] = {}
    if config.batch_size is not None:
        options["batch_size"] = config.batch_size
    if config.reverse_complement is not None:
        options["reverse_complement"] = config.reverse_complement
    return options


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GIAnnotationInput(sequences=[GISequence(sequence="ATGC" * 250, name="example")])


@tool(
    key="gi-annotation",
    label="GI Gene Annotation",
    category="sequence_scoring",
    input_class=GIAnnotationInput,
    config_class=GIAnnotationConfig,
    output_class=GIAnnotationOutput,
    description="Find genes and transcripts de novo in raw DNA via the hosted Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-annotation calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_annotation(
    inputs: GIAnnotationInput,
    config: GIAnnotationConfig,
    instance: Any = None,
) -> GIAnnotationOutput:
    """Annotate transcripts in each submitted sequence.

    Args:
        inputs (GIAnnotationInput): Sequences to annotate.
        config (GIAnnotationConfig): API credentials, model selection, options.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GIAnnotationOutput: One result per submitted sequence, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API, and on a 2xx whose
            body is not a ``{data, meta}`` envelope.
        GIResponseShapeError: If a field inside ``data`` documented as an
            object or an array arrives as something else.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    results: list[GIAnnotationResult] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=ANNOTATION_MIN_BP, task="annotation")
        data, payload = call_predict(
            config,
            "annotation",
            sequence,
            item.name,
            options=_annotation_options(config),
        )
        results.append(parse_annotation_data(data, payload, item.name))
    return GIAnnotationOutput(results=results)
