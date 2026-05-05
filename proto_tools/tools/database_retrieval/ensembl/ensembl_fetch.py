"""proto_tools/tools/database_retrieval/ensembl/ensembl_fetch.py.

Wraps Ensembl REST for gene/transcript/exon lookup, sequence retrieval,
region overlap, and cross-references via an ``endpoint:`` switch.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblGene,
    EnsemblOverlapFeatureRecord,
    EnsemblSequence,
    EnsemblSpecies,
    EnsemblXref,
    base_url_for,
    build_session,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
)

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30


EnsemblEndpoint = Literal["lookup_id", "lookup_symbol", "sequence", "overlap", "xrefs"]
EnsemblSequenceType = Literal["genomic", "cdna", "cds", "protein"]
EnsemblOverlapFeature = Literal[
    "gene",
    "transcript",
    "exon",
    "cds",
    "regulatory",
    "motif",
    "variation",
    "repeat",
    "constrained",
    "translation_exon",
]


# ============================================================================
# Data Models
# ============================================================================


# Discriminated by `EnsemblFetchOutput.endpoint`; the Output keeps `result: Any`
# since Pydantic doesn't accept a `BaseModel | list[...]` union as a field type.
EnsemblResult = EnsemblGene | EnsemblSequence | list[EnsemblXref] | list[EnsemblOverlapFeatureRecord]


class EnsemblFetchInput(BaseToolInput):
    """Input for Ensembl fetch.

    The required field depends on ``config.endpoint``:
    ``ensembl_id`` for ``lookup_id`` / ``sequence`` / ``overlap``;
    ``symbol`` for ``lookup_symbol``; either for ``xrefs``.

    Attributes:
        ensembl_id (str | None): Ensembl ID (e.g. ``ENSG00000012048``).
        symbol (str | None): Gene symbol (e.g. ``BRCA1``).
    """

    ensembl_id: str | None = InputField(default=None, description="Ensembl ID (ENSG..., ENST..., ENSP...)")
    symbol: str | None = InputField(default=None, description="Gene symbol (e.g. BRCA1)")

    @model_validator(mode="after")
    def validate_input(self) -> "EnsemblFetchInput":
        """Require at least one of ensembl_id / symbol after stripping whitespace."""
        eid = (self.ensembl_id or "").strip()
        sym = (self.symbol or "").strip()
        if not eid and not sym:
            raise ValueError("Provide either ensembl_id or symbol")
        return self


class EnsemblFetchConfig(BaseConfig):
    """Configuration for Ensembl fetch.

    Attributes:
        endpoint (EnsemblEndpoint): Which Ensembl REST endpoint to call.
        species (EnsemblSpecies): Species slug used by ``lookup_symbol`` and
            symbol-form ``xrefs``. Default ``homo_sapiens``.
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            calls ``rest.ensembl.org``; ``GRCh37`` calls
            ``grch37.rest.ensembl.org``.
        sequence_type (EnsemblSequenceType): Sequence flavor for the
            ``sequence`` endpoint (``genomic`` / ``cdna`` / ``cds`` /
            ``protein``).
        overlap_feature (EnsemblOverlapFeature): Feature class for the
            ``overlap`` endpoint.
        expand (bool): Expand transcripts/exons on ``lookup_*``.
    """

    endpoint: EnsemblEndpoint = ConfigField(
        title="Endpoint",
        default="lookup_id",
        description="Which Ensembl REST endpoint to call",
    )
    species: EnsemblSpecies = ConfigField(
        title="Species",
        default="homo_sapiens",
        description="Species (used by lookup_symbol and symbol-form xrefs)",
        depends_on={"endpoint": ["lookup_symbol", "xrefs"]},
    )
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly",
        default="GRCh38",
        description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org",
    )
    sequence_type: EnsemblSequenceType = ConfigField(
        title="Sequence Type",
        default="cdna",
        description="Sequence flavor (sequence endpoint only)",
        advanced=True,
        depends_on={"endpoint": ["sequence"]},
    )
    overlap_feature: EnsemblOverlapFeature = ConfigField(
        title="Overlap Feature",
        default="gene",
        description="Feature type (overlap endpoint only)",
        advanced=True,
        depends_on={"endpoint": ["overlap"]},
    )
    expand: bool = ConfigField(
        title="Expand Transcripts/Exons",
        default=True,
        description="Include transcripts/exons (lookup_* endpoints only)",
        advanced=True,
        depends_on={"endpoint": ["lookup_id", "lookup_symbol"]},
    )


class EnsemblFetchOutput(BaseToolOutput):
    """Output from Ensembl fetch.

    The ``result`` field's concrete type depends on ``endpoint``:
    ``EnsemblGene`` for ``lookup_*``; ``EnsemblSequence`` for ``sequence``;
    ``list[EnsemblXref]`` for ``xrefs``; ``list[EnsemblOverlapFeatureRecord]``
    for ``overlap``.

    Attributes:
        endpoint (EnsemblEndpoint): The endpoint that produced this result.
        result (Any): The parsed payload — see endpoint mapping above.
        source_url (str): Final URL that was hit (after redirects/params).
        raw_payload (dict[str, Any] | list[dict[str, Any]]): Raw API JSON.
    """

    endpoint: EnsemblEndpoint = Field(description="Which endpoint produced this result")
    result: Any = Field(description="Parsed payload; concrete type depends on endpoint")
    source_url: str = Field(description="Final Ensembl REST URL that was hit")
    raw_payload: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict, description="Raw API JSON")

    @property
    def output_format_options(self) -> list[str]:
        """Return supported output formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format == "json":
            path = Path(export_path).with_suffix(".json")
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EnsemblFetchInput(ensembl_id="ENSG00000012048")


@tool(
    key="ensembl-fetch",
    label="Ensembl Fetch",
    category="database_retrieval",
    input_class=EnsemblFetchInput,
    config_class=EnsemblFetchConfig,
    output_class=EnsemblFetchOutput,
    description=(
        "Fetch gene / transcript / exon hierarchy, sequences, region overlaps, or cross-references "
        "from the Ensembl REST API (GRCh38 default; GRCh37 supported)"
    ),
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_fetch(
    inputs: EnsemblFetchInput,
    config: EnsemblFetchConfig,
    instance: Any = None,
) -> EnsemblFetchOutput:
    """Dispatch to one of Ensembl REST's five fetch endpoints.

    Args:
        inputs (EnsemblFetchInput): Ensembl ID or gene symbol.
        config (EnsemblFetchConfig): Endpoint + species + assembly + per-endpoint knobs.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblFetchOutput: Parsed result whose concrete type matches
            ``config.endpoint``.
    """
    del instance

    base = base_url_for(config.assembly)
    url, params = _build_url_and_params(inputs, config, base)

    session = build_session("ensembl-fetch")
    try:
        response = session.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Ensembl returned non-JSON for endpoint={config.endpoint}, url={response.url}; "
                f"body[:200]={response.text[:200]!r}"
            ) from exc
        result = _parse_payload(payload, config.endpoint)
        return EnsemblFetchOutput(
            endpoint=config.endpoint,
            result=result,
            source_url=response.url,
            raw_payload=payload,
        )
    finally:
        session.close()


# ============================================================================
# Private Helpers
# ============================================================================


def _build_url_and_params(
    inputs: EnsemblFetchInput,
    config: EnsemblFetchConfig,
    base: str,
) -> "tuple[str, dict[str, Any]]":
    """Build the per-endpoint URL + query params; raise if input is missing for the endpoint."""
    eid = (inputs.ensembl_id or "").strip()
    sym = (inputs.symbol or "").strip()

    if config.endpoint == "lookup_id":
        if not eid:
            raise ValueError("endpoint='lookup_id' requires inputs.ensembl_id")
        return f"{base}/lookup/id/{eid}", ({"expand": "1"} if config.expand else {})
    if config.endpoint == "lookup_symbol":
        if not sym:
            raise ValueError("endpoint='lookup_symbol' requires inputs.symbol")
        return (
            f"{base}/lookup/symbol/{config.species}/{sym}",
            ({"expand": "1"} if config.expand else {}),
        )
    if config.endpoint == "sequence":
        if not eid:
            raise ValueError("endpoint='sequence' requires inputs.ensembl_id")
        return f"{base}/sequence/id/{eid}", {"type": config.sequence_type}
    if config.endpoint == "overlap":
        if not eid:
            raise ValueError("endpoint='overlap' requires inputs.ensembl_id")
        return f"{base}/overlap/id/{eid}", {"feature": config.overlap_feature}
    if config.endpoint == "xrefs":
        if eid:
            return f"{base}/xrefs/id/{eid}", {}
        return f"{base}/xrefs/symbol/{config.species}/{sym}", {}
    raise ValueError(f"Unknown endpoint: {config.endpoint!r}")


def _parse_payload(payload: Any, endpoint: EnsemblEndpoint) -> EnsemblResult:
    """Parse the API JSON into the typed result for the given endpoint."""
    if endpoint in ("lookup_id", "lookup_symbol"):
        if not isinstance(payload, dict):
            raise ValueError(f"Ensembl {endpoint} returned non-dict payload: {type(payload).__name__}")
        return EnsemblGene.model_validate(payload)
    if endpoint == "sequence":
        if not isinstance(payload, dict):
            raise ValueError(f"Ensembl sequence returned non-dict payload: {type(payload).__name__}")
        return EnsemblSequence.model_validate(payload)
    if endpoint == "xrefs":
        if not isinstance(payload, list):
            raise ValueError(f"Ensembl xrefs returned non-list payload: {type(payload).__name__}")
        return [EnsemblXref.model_validate(x) for x in payload]
    if endpoint == "overlap":
        if not isinstance(payload, list):
            raise ValueError(f"Ensembl overlap returned non-list payload: {type(payload).__name__}")
        records: list[EnsemblOverlapFeatureRecord] = []
        for i, f in enumerate(payload):
            if not isinstance(f, dict):
                raise ValueError(f"Ensembl overlap payload[{i}] is non-dict: {type(f).__name__}")
            records.append(EnsemblOverlapFeatureRecord.model_validate({**f, "raw": f}))
        return records
    raise ValueError(f"Unknown endpoint: {endpoint!r}")
