"""proto_tools/tools/database_retrieval/ensembl/ensembl_xrefs.py.

Wraps Ensembl REST ``/xrefs/id/{id}`` — cross-references from an Ensembl ID
to external databases (UniProt, EntrezGene, RefSeq, ...).
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
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

    ensembl_id: str = InputField(description="Ensembl ID (ENSG..., ENST..., ENSP...)")

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
    """

    assembly: EnsemblAssembly = ConfigField(
        title="Assembly",
        default="GRCh38",
        description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org",
    )


class EnsemblXrefsOutput(BaseToolOutput):
    """Output from Ensembl xrefs query.

    Attributes:
        result (list[EnsemblXref]): Cross-reference records to external
            databases (UniProt, EntrezGene, RefSeq, ...).
        source_url (str): Final Ensembl REST URL that was hit.
        raw_payload (list[dict[str, Any]]): Raw API JSON.
    """

    result: list[EnsemblXref] = Field(default_factory=list, description="Cross-reference records to external databases")
    source_url: str = Field(description="Final Ensembl REST URL that was hit")
    raw_payload: list[dict[str, Any]] = Field(default_factory=list, description="Raw API JSON")

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

    session = build_session("ensembl-xrefs")
    try:
        response = session.get(
            url,
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
