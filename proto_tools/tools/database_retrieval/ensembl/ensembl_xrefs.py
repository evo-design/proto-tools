"""proto_tools/tools/database_retrieval/ensembl/ensembl_xrefs.py.

Wraps Ensembl REST ``/xrefs/id/{id}`` — cross-references from an Ensembl ID
to external databases (UniProt, EntrezGene, RefSeq, ...).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblExternalDB,
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


# ============================================================================
# Data Models
# ============================================================================


class EnsemblXrefsInput(BaseToolInput):
    """Input for Ensembl cross-reference lookup.

    Attributes:
        ensembl_id (str): Ensembl ID for direct cross-reference lookup.
    """

    ensembl_id: str = InputField(title="Ensembl ID", description="Ensembl ID (ENSG..., ENST..., ENSP...)")

    @field_validator("ensembl_id")
    @classmethod
    def validate_ensembl_id(cls, value: str) -> str:
        """Reject blank stable IDs before constructing an Ensembl URL."""
        if not value.strip():
            raise ValueError("ensembl_id cannot be blank")
        return value


class EnsemblXrefsConfig(BaseConfig):
    """Configuration for Ensembl xrefs query.

    Attributes:
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            or ``GRCh37``.
        all_levels (bool): Fan out to transcripts and translations.
            On a gene query this also returns xrefs from each child
            transcript and protein.
        external_db (EnsemblExternalDB | None): Restrict to one external
            database (e.g. ``UniProtKB/Swiss-Prot``, ``HGNC``).
        object_type (Literal['gene','transcript','translation'] | None):
            Restrict to one feature type when the stable ID resolves
            ambiguously.
    """

    assembly: EnsemblAssembly = ConfigField(
        title="Assembly", default="GRCh38", description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org"
    )
    all_levels: bool = ConfigField(
        title="All Levels",
        default=False,
        description="Fan out xrefs to child transcripts and translations (gene queries)",
    )
    external_db: EnsemblExternalDB | None = ConfigField(
        title="External DB Filter",
        default=None,
        description="Restrict to one external DB (e.g. 'UniProtKB/Swiss-Prot', 'HGNC')",
    )
    object_type: Literal["gene", "transcript", "translation"] | None = ConfigField(
        title="Object Type Filter",
        default=None,
        description="Restrict to one feature type when the stable ID resolves ambiguously",
    )


class EnsemblXrefsOutput(BaseToolOutput):
    """Output from Ensembl xrefs query.

    Attributes:
        result (list[EnsemblXref]): Cross-reference records to external
            databases (UniProt, EntrezGene, RefSeq, ...).
        source_url (str): Final Ensembl REST URL that was hit.
        raw_payload (list[dict[str, Any]]): Raw API JSON.
    """

    result: list[EnsemblXref] = Field(
        default_factory=list, title="Cross-References", description="Cross-reference records to external databases"
    )
    source_url: str = Field(title="Source URL", description="Final Ensembl REST URL that was hit")
    raw_payload: list[dict[str, Any]] = Field(
        default_factory=list, title="Raw Payload", description="Raw API JSON returned by Ensembl"
    )

    @property
    def output_format_options(self) -> list[str]:
        """Return supported output formats."""
        return ["json", "csv"]

    @property
    def output_format_default(self) -> str:
        """Return the default output format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        path = Path(export_path).with_suffix(f".{file_format}")
        if file_format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2)
            return
        if file_format == "csv":
            rows = [r.model_dump() for r in self.result]
            with path.open("w", encoding="utf-8", newline="") as f:
                if not rows:
                    return
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EnsemblXrefsInput(ensembl_id="ENSG00000012048")


@tool(
    key="ensembl-xrefs",
    label="Ensembl Xrefs",
    category="database_retrieval",
    input_class=EnsemblXrefsInput,
    config_class=EnsemblXrefsConfig,
    output_class=EnsemblXrefsOutput,
    description="Fetch cross-references from an Ensembl ID to external databases",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_xrefs(
    inputs: EnsemblXrefsInput,
    config: EnsemblXrefsConfig,
    instance: Any = None,
) -> EnsemblXrefsOutput:
    """Fetch cross-references from Ensembl REST.

    Args:
        inputs (EnsemblXrefsInput): Ensembl ID.
        config (EnsemblXrefsConfig): Assembly.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblXrefsOutput: List of ``EnsemblXref`` records.
    """
    del instance

    base = base_url_for(config.assembly)
    eid = inputs.ensembl_id.strip()
    url = f"{base}/xrefs/id/{eid}"
    params: dict[str, Any] = {}
    if config.all_levels:
        params["all_levels"] = "1"
    if config.external_db:
        params["external_db"] = config.external_db
    if config.object_type:
        params["object_type"] = config.object_type

    session = build_session("ensembl-xrefs")
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
                f"Ensembl returned non-JSON for xrefs at {response.url}; body[:200]={response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"Ensembl xrefs returned non-list payload: {type(payload).__name__}")
        return EnsemblXrefsOutput(
            result=[EnsemblXref.model_validate(x) for x in payload],
            source_url=response.url,
            raw_payload=payload,
        )
    finally:
        session.close()
