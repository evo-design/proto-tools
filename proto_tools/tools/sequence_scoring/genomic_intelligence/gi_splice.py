"""Splice donor/acceptor site prediction via the Genomic Intelligence hosted API.

The model is strand-specific: submit the sequence in transcript orientation.
Scoring the opposite strand does not fail loudly — it returns sites at different
positions, frequently at high confidence — so there is no score or count that
identifies a mis-oriented submission after the fact. Reverse-complement
minus-strand genes before calling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

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

SPLICE_MIN_BP = 100
"""Published ``minLength`` for the splice endpoint, in base pairs."""

SpliceSiteKind = Literal["donor", "acceptor"]


# ============================================================================
# Data Models
# ============================================================================


class GISpliceInput(BaseToolInput):
    """Input for splice-site prediction.

    Attributes:
        sequences (list[GISequence]): Sequences to score, in transcript
            orientation. A bare DNA string is accepted and coerced. Each must be
            at least 100 bp, the endpoint's published floor.
    """

    sequences: list[GISequence] = InputField(
        title="Sequences",
        description="DNA sequences in transcript orientation (>=100 bp each)",
        min_length=1,
    )

    @field_validator("sequences", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        """Accept a bare string or a single item in place of a list."""
        return coerce_gi_sequences(value)


class GISpliceConfig(GIConfig):
    """Configuration for splice-site prediction.

    Attributes:
        threshold (float): Score above which a position is reported as a site.
            Values at or near zero return every scored position and produce very
            large responses; the default is a good working value.
        site_types (list[SpliceSiteKind] | None): Restrict the reported site
            classes. Leave unset to report both.
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
        gt=0.0,
        le=1.0,
        description="Score above which a position is reported as a splice site",
    )
    site_types: list[SpliceSiteKind] | None = ConfigField(
        title="Site Types",
        default=None,
        description="Subset of ['donor', 'acceptor'] to report; unset reports both",
    )


class SpliceSite(BaseModel):
    """One predicted splice site.

    ``start`` and ``end`` bound a tokenizer span, not the exon/intron junction.
    The span is one variable-width token -- 4-10 bp across the sequences measured
    so far -- and the junction lies somewhere inside it, so neither endpoint is a
    base-resolution boundary coordinate.

    Attributes:
        name (str): Service-assigned site label.
        start (int): 0-based inclusive start of the site's token span.
        end (int): 0-based exclusive end of the site's token span.
        site_type (str): ``donor`` or ``acceptor``.
        score (float): Model score for the site.
    """

    name: str = Field(title="Name", default="", description="Service-assigned site label")
    start: int = Field(
        title="Start",
        description="0-based inclusive start of the site's token span in the submitted sequence",
        ge=0,
    )
    end: int = Field(
        title="End",
        description="0-based exclusive end of the site's token span in the submitted sequence",
        ge=0,
    )
    site_type: str = Field(title="Site Type", description="'donor' or 'acceptor'")
    score: float = Field(title="Score", description="Model score for the site")

    model_config = ConfigDict(frozen=True)


class GISpliceResult(BaseModel):
    """Splice-site prediction for one submitted sequence.

    Attributes:
        name (str): Label supplied with the sequence.
        sequence_length (int): Length of the submitted sequence in base pairs.
        total_sites (int): Number of sites reported.
        donor_sites (int): Number of donor sites reported.
        acceptor_sites (int): Number of acceptor sites reported.
        sites (list[SpliceSite]): The reported sites.
        meta (GIRequestMeta): Provenance for the call.
    """

    name: str = Field(title="Name", description="Label supplied with the sequence")
    sequence_length: int = Field(
        title="Sequence Length", description="Length of the submitted sequence in base pairs", ge=0
    )
    total_sites: int = Field(title="Total Sites", description="Number of sites reported", ge=0)
    donor_sites: int = Field(title="Donor Sites", description="Number of donor sites reported", ge=0)
    acceptor_sites: int = Field(title="Acceptor Sites", description="Number of acceptor sites reported", ge=0)
    sites: list[SpliceSite] = Field(title="Sites", default_factory=list, description="The reported sites")
    meta: GIRequestMeta = Field(title="Meta", description="Provenance for the call")

    model_config = ConfigDict(frozen=True)


class GISpliceOutput(BaseToolOutput):
    """Output from splice-site prediction.

    Attributes:
        results (list[GISpliceResult]): One result per submitted sequence, in
            the order submitted.
    """

    results: list[GISpliceResult] = Field(title="Results", description="One result per submitted sequence, in order")

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


def parse_splice_data(data: dict[str, Any], payload: dict[str, Any], name: str) -> GISpliceResult:
    """Build a :class:`GISpliceResult` from one response payload.

    Args:
        data (dict[str, Any]): The response's ``data`` member.
        payload (dict[str, Any]): The full ``{data, meta}`` response.
        name (str): Label supplied with the sequence.

    Returns:
        GISpliceResult: Parsed result.
    """
    summary = as_object(data.get("summary"), "data.summary")
    sites = [
        SpliceSite(
            name=str(site.get("name", "")),
            start=int(site.get("start", 0)),
            end=int(site.get("end", 0)),
            site_type=str(site.get("site_type", "")),
            score=float(site.get("score", 0.0)),
        )
        for site in as_object_list(data.get("sites"), "data.sites")
    ]
    meta = as_object(payload.get("meta"), "meta")
    return GISpliceResult(
        name=name,
        sequence_length=int(meta.get("sequence_length", 0)),
        total_sites=int(summary.get("total_sites", len(sites))),
        donor_sites=int(summary.get("donor_sites", 0)),
        acceptor_sites=int(summary.get("acceptor_sites", 0)),
        sites=sites,
        meta=build_request_meta(payload, data),
    )


def _splice_options(config: GISpliceConfig) -> dict[str, Any]:
    """Build the task's closed options object, omitting unset members.

    ``SpliceOptions`` is ``additionalProperties: false``, so only declared keys
    may be sent and ``None`` is not a valid value for ``site_types``.

    Args:
        config (GISpliceConfig): Tool configuration.

    Returns:
        dict[str, Any]: Options to send with the request.
    """
    options: dict[str, Any] = {"threshold": config.threshold}
    if config.site_types is not None:
        options["site_types"] = list(config.site_types)
    return options


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return GISpliceInput(sequences=[GISequence(sequence="ATGC" * 50, name="example")])


@tool(
    key="gi-splice",
    label="GI Splice Sites",
    category="sequence_scoring",
    input_class=GISpliceInput,
    config_class=GISpliceConfig,
    output_class=GISpliceOutput,
    description="Predict splice donor and acceptor sites via the hosted Genomic Intelligence API",
    uses_gpu=False,
    example_input=example_input,
    iterable_input_fields=["sequences"],
    iterable_output_field="results",
    max_chunk_size=1,
    cacheable=True,
    local_only="gi-splice calls a hosted HTTP API, so it neither uses a GPU nor needs an environment",
)
def run_gi_splice(
    inputs: GISpliceInput,
    config: GISpliceConfig,
    instance: Any = None,
) -> GISpliceOutput:
    """Predict splice sites for each submitted sequence.

    Submit sequences in transcript orientation. The model is strand-specific and
    the wrong strand yields plausible, often high-confidence sites at different
    positions, so the result cannot be checked for orientation after the fact.

    Args:
        inputs (GISpliceInput): Sequences to score.
        config (GISpliceConfig): API credentials, model selection, thresholds.
        instance (Any): Unused; the tool makes no subprocess dispatch.

    Returns:
        GISpliceOutput: One result per submitted sequence, in order.

    Raises:
        GIAPIError: On any non-2xx response from the API, and on a 2xx whose
            body is not a ``{data, meta}`` envelope.
        GIResponseShapeError: If a field inside ``data`` documented as an
            object or an array arrives as something else.
        OSError: If no API key is configured.
        ValueError: If a sequence falls outside the endpoint's published bounds.
    """
    del instance
    results: list[GISpliceResult] = []
    for item in inputs.sequences:
        sequence = validate_gi_sequence(item.sequence, min_bp=SPLICE_MIN_BP, task="splice")
        data, payload = call_predict(
            config,
            "splice",
            sequence,
            item.name,
            options=_splice_options(config),
        )
        results.append(parse_splice_data(data, payload, item.name))
    return GISpliceOutput(results=results)
