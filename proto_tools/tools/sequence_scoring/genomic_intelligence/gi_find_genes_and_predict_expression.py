"""Gene finding plus per-gene expression, in one call to the hosted API.

Runs annotation over the submitted locus, then scores expression for every gene
it finds, centring each window on that gene's own TSS. Use it when the TSS
positions are not known in advance; use ``gi-expression`` directly when they are.

This is the one endpoint with a delivery rule rather than a delivery preference:
a synchronous request above 50,000 bp is refused with ``413 sync_too_large``, so
long loci must set ``respond_async``. The tool does that automatically. Its
length ceiling is the endpoint's own 500,000 bp, which is not the same as the
expression model's window; gating on the model's number here rejects input the
service accepts.
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
    call_workflow,
    coerce_gi_sequences,
    validate_gi_sequence,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField, get_logger

logger = get_logger(__name__)

WORKFLOW_MIN_BP = 1_000
"""Published ``minLength`` for the workflow endpoint, in base pairs."""

WORKFLOW_SYNC_LIMIT_BP = 50_000
"""Above this length the endpoint refuses synchronous delivery with a 413."""


# ============================================================================
# Data Models
# ============================================================================


class GIFindGenesInput(BaseToolInput):
    """Input for the gene-finding plus expression workflow.

    Attributes:
        sequences (list[GISequence]): Loci to process. A bare DNA string is
            accepted and coerced. Each must be at least 1,000 bp, the
            endpoint's published floor.
    """

    sequences: list[GISequence] = InputField(
        title="Sequences",
        description="Genomic loci to annotate and score (>=1,000 bp each)",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        return coerce_gi_sequences(value)


class GIFindGenesConfig(GIConfig):
    """Configuration for the workflow.

    Attributes:
        description (str): Experimental context applied to every gene found.
            Conditioning text fed to the expression model, so its wording
            changes the predictions. The service requires it even though the
            published schema marks it optional.
        annotation_model (str | None): Override the annotation stage's model.
            Leave unset.
        expression_model (str | None): Override the expression stage's model.
            Leave unset.
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
        description="Experimental context applied to every gene found; wording changes the predictions",
    )
    annotation_model: str | None = ConfigField(
        title="Annotation Model",
        default=None,
        description="Annotation-stage model id; unset uses the service default",
    )
    expression_model: str | None = ConfigField(
        title="Expression Model",
        default=None,
        description="Expression-stage model id; unset uses the service default",
    )


class GenePrediction(BaseModel):
    """Expression predicted for one gene found by the annotation stage.

    Attributes:
        gene_index (int): Position of the gene in the annotation output.
        gene_name (str): Service-assigned gene label.
        strand (str): Orientation relative to the submitted sequence.
        tss_position (int): TSS offset in the submitted sequence.
        expression (float | None): Predicted log(TPM+1).
        expression_tpm (float | None): Predicted TPM.
        skipped (bool): Whether expression was skipped for this gene.
        skip_reason (str | None): Why it was skipped, when it was.
    """

    gene_index: int = Field(title="Gene Index", description="Position of the gene in the annotation output", ge=0)
    gene_name: str = Field(title="Gene Name", default="", description="Service-assigned gene label")
    strand: str = Field(title="Strand", default="", description="Orientation relative to the submitted sequence")
    tss_position: int = Field(title="TSS Position", default=0, description="TSS offset in the submitted sequence")
    expression: float | None = Field(title="Expression", default=None, description="Predicted log(TPM+1)")
    expression_tpm: float | None = Field(title="Expression TPM", default=None, description="Predicted TPM")
    skipped: bool = Field(title="Skipped", default=False, description="Whether expression was skipped for this gene")
    skip_reason: str | None = Field(title="Skip Reason", default=None, description="Why expression was skipped")

    model_config = ConfigDict(frozen=True)


class GIFindGenesResult(BaseModel):
    """Workflow result for one submitted locus.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        genes_found (int): Genes the annotation stage reported.
        genes_scored (int): Genes expression was predicted for.
        predictions (list[GenePrediction]): One record per gene found.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    genes_found: int = Field(title="Genes Found", description="Genes the annotation stage reported", ge=0)
    genes_scored: int = Field(title="Genes Scored", description="Genes expression was predicted for", ge=0)
    predictions: list[GenePrediction] = Field(
        title="Predictions", default_factory=list, description="One record per gene found"
    )
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GIFindGenesOutput(BaseToolOutput):
    """Output from the gene-finding plus expression workflow.

    Attributes:
        results (list[GIFindGenesResult]): One result per submitted locus, in
            the order submitted.
    """

    results: list[GIFindGenesResult] = Field(title="Results", description="One result per submitted locus, in order")

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


def parse_workflow_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> GIFindGenesResult:
    """Build a :class:`GIFindGenesResult` from one response payload.

    The workflow reports a model per stage rather than one top-level ``model``,
    so provenance is assembled here instead of through the shared helper.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        GIFindGenesResult: Parsed result.
    """
    summary = as_object(data.get("summary"), "data.summary")
    meta = as_object(payload.get("meta"), "meta")
    predictions = [
        GenePrediction(
            gene_index=int(record.get("gene_index", index)),
            gene_name=str(record.get("gene_name", "")),
            strand=str(record.get("strand", "")),
            tss_position=int(record.get("tss_position", 0)),
            expression=record.get("expression"),
            expression_tpm=record.get("expression_tpm"),
            skipped=bool(record.get("skipped", False)),
            skip_reason=record.get("skip_reason"),
        )
        for index, record in enumerate(
            as_object_list(data.get("expression_predictions"), "data.expression_predictions")
        )
    ]
    annotation_model = str(data.get("annotation_model") or "")
    expression_model = str(data.get("expression_model") or "")
    return GIFindGenesResult(
        name=name,
        sequence_length=int(as_object(data.get("input"), "data.input").get("sequence_length", 0)),
        genes_found=int(summary.get("genes_found", len(predictions))),
        genes_scored=int(summary.get("genes_predicted", sum(1 for record in predictions if not record.skipped))),
        predictions=predictions,
        meta=GIRequestMeta(
            model=f"{annotation_model} + {expression_model}".strip(" +"),
            request_id=meta.get("request_id"),
            job_id=meta.get("job_id"),
            inference_time_ms=meta.get("inference_time_ms"),
            cold_start=meta.get("cold_start"),
        ),
    )


def _workflow_options(config: GIFindGenesConfig) -> dict[str, Any]:
    """Build the workflow's closed options object, omitting unset members.

    The endpoint requires ``options`` to be present and, despite the published
    schema marking it optional, rejects a request without
    ``options.description``.

    Args:
        config (GIFindGenesConfig): Tool configuration.

    Returns:
        dict[str, Any]: Options to send with the request.
    """
    options: dict[str, Any] = {"description": config.description}
    if config.annotation_model is not None:
        options["annotation_model"] = config.annotation_model
    if config.expression_model is not None:
        options["expression_model"] = config.expression_model
    return options


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GIFindGenesInput(sequences=[GISequence(sequence="ATGC" * 250, name="example")])


@tool(
    key="gi-find-genes-and-predict-expression",
    label="GI Find Genes and Predict Expression",
    category="sequence_scoring",
    input_class=GIFindGenesInput,
    config_class=GIFindGenesConfig,
    output_class=GIFindGenesOutput,
    description="Annotate genes in a locus and predict expression for each, via the hosted Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-find-genes-and-predict-expression calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_find_genes_and_predict_expression(
    inputs: GIFindGenesInput,
    config: GIFindGenesConfig,
    instance: Any = None,
) -> GIFindGenesOutput:
    """Annotate each locus and predict expression for every gene found.

    Delivery is switched to asynchronous automatically for loci above the
    endpoint's 50,000 bp synchronous limit, which it would otherwise refuse.

    Args:
        inputs (GIFindGenesInput): Loci to process.
        config (GIFindGenesConfig): API credentials, stage models, conditioning.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GIFindGenesOutput: One result per submitted locus, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API, and on a 2xx whose
            body is not a ``{data, meta}`` envelope.
        GIResponseShapeError: If a field inside ``data`` documented as an
            object or an array arrives as something else.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    options = _workflow_options(config)
    results: list[GIFindGenesResult] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=WORKFLOW_MIN_BP, task="find-genes-and-predict-expression")
        call_config = config
        if len(sequence) > WORKFLOW_SYNC_LIMIT_BP and not config.respond_async:
            logger.info(
                "Sequence %s is %d bp, above the endpoint's %d bp synchronous limit; requesting async delivery.",
                item.name,
                len(sequence),
                WORKFLOW_SYNC_LIMIT_BP,
            )
            call_config = config.model_copy(update={"respond_async": True})
        data, payload = call_workflow(call_config, sequence, item.name, options)
        results.append(parse_workflow_data(data, payload, item.name))
    return GIFindGenesOutput(results=results)
