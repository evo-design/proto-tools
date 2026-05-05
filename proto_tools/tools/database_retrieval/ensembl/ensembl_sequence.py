"""proto_tools/tools/database_retrieval/ensembl/ensembl_sequence.py.

Wraps Ensembl REST ``/sequence/id/{id}`` — DNA / cDNA / CDS / protein
sequence retrieval keyed by an Ensembl gene, transcript, or protein ID.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblSequence,
    EnsemblSequenceType,
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


class EnsemblSequenceInput(BaseToolInput):
    """Input for Ensembl sequence fetch.

    Attributes:
        ensembl_id (str): Ensembl ID (``ENSG...``, ``ENST...``, or ``ENSP...``).
    """

    ensembl_id: str = InputField(description="Ensembl ID (ENSG..., ENST..., ENSP...)")

    @field_validator("ensembl_id")
    @classmethod
    def validate_ensembl_id(cls, value: str) -> str:
        """Reject blank stable IDs before constructing an Ensembl URL."""
        if not value.strip():
            raise ValueError("ensembl_id cannot be blank")
        return value


class EnsemblSequenceConfig(BaseConfig):
    """Configuration for Ensembl sequence fetch.

    Attributes:
        sequence_type (EnsemblSequenceType): Sequence flavor —
            ``genomic`` (default) / ``cdna`` / ``cds`` / ``protein``.
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            or ``GRCh37``.
    """

    sequence_type: EnsemblSequenceType = ConfigField(
        title="Sequence Type",
        default="genomic",
        description="Sequence flavor",
    )
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly",
        default="GRCh38",
        description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org",
    )


class EnsemblSequenceOutput(BaseToolOutput):
    """Output from Ensembl sequence fetch.

    Attributes:
        result (EnsemblSequence): The fetched sequence record.
        source_url (str): Final Ensembl REST URL that was hit.
        raw_payload (dict[str, Any]): Raw API JSON.
    """

    result: EnsemblSequence = Field(description="The fetched sequence record")
    source_url: str = Field(description="Final Ensembl REST URL that was hit")
    raw_payload: dict[str, Any] = Field(default_factory=dict, description="Raw API JSON")

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
    return EnsemblSequenceInput(ensembl_id="ENST00000357654")


@tool(
    key="ensembl-sequence",
    label="Ensembl Sequence",
    category="database_retrieval",
    input_class=EnsemblSequenceInput,
    config_class=EnsemblSequenceConfig,
    output_class=EnsemblSequenceOutput,
    description="Fetch DNA / cDNA / CDS / protein sequence for an Ensembl ID",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_sequence(
    inputs: EnsemblSequenceInput,
    config: EnsemblSequenceConfig,
    instance: Any = None,
) -> EnsemblSequenceOutput:
    """Fetch a sequence record from Ensembl REST.

    Args:
        inputs (EnsemblSequenceInput): Ensembl ID.
        config (EnsemblSequenceConfig): Sequence type + assembly.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblSequenceOutput: Parsed ``EnsemblSequence`` plus metadata.
    """
    del instance

    base = base_url_for(config.assembly)
    url = f"{base}/sequence/id/{inputs.ensembl_id.strip()}"
    params = {"type": config.sequence_type}

    session = build_session("ensembl-sequence")
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
                f"Ensembl returned non-JSON for sequence at {response.url}; body[:200]={response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Ensembl sequence returned non-dict payload: {type(payload).__name__}")
        return EnsemblSequenceOutput(
            result=EnsemblSequence.model_validate(payload),
            source_url=response.url,
            raw_payload=payload,
        )
    finally:
        session.close()
