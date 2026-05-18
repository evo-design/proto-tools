"""proto_tools/tools/database_retrieval/ensembl/ensembl_lookup.py.

Wraps Ensembl REST ``/lookup/`` — gene lookup by Ensembl gene ID or by
gene symbol. ``expand=True`` includes the upstream transcript/exon hierarchy.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblGene,
    EnsemblSpecies,
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


class EnsemblLookupInput(BaseToolInput):
    """Input for Ensembl lookup.

    Provide exactly one of ``ensembl_id`` (direct lookup) or ``symbol`` (gene-name
    lookup, requires ``config.species``).

    Attributes:
        ensembl_id (str | None): Ensembl gene ID (e.g. ``ENSG...``).
        symbol (str | None): Gene symbol (e.g. ``BRCA1``).
    """

    ensembl_id: str | None = InputField(default=None, description="Ensembl gene ID (e.g. ENSG00000012048)")
    symbol: str | None = InputField(default=None, description="Gene symbol (e.g. BRCA1)")

    @model_validator(mode="after")
    def validate_input(self) -> "EnsemblLookupInput":
        """Require exactly one of ensembl_id / symbol after stripping whitespace."""
        has_ensembl_id = bool((self.ensembl_id or "").strip())
        has_symbol = bool((self.symbol or "").strip())
        if has_ensembl_id == has_symbol:
            raise ValueError("Provide exactly one of ensembl_id or symbol")
        return self


class EnsemblLookupConfig(BaseConfig):
    """Configuration for Ensembl lookup.

    Attributes:
        species (EnsemblSpecies): Species slug used when ``symbol`` is the
            input. Default ``homo_sapiens``.
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            calls ``rest.ensembl.org``; ``GRCh37`` calls
            ``grch37.rest.ensembl.org``.
        expand (bool): Include transcripts, translations, and exons in the
            response. Default ``False`` matches Ensembl REST.
        mane (bool): Include MANE Select annotations (``/lookup/id`` only;
            requires ``expand=True``).
        phenotypes (bool): Include phenotype annotations on gene records
            (``/lookup/id`` only).
        utr (bool): Include UTR coordinates per transcript (``/lookup/id``
            only; requires ``expand=True``).
    """

    species: EnsemblSpecies = ConfigField(
        title="Species", default="homo_sapiens", description="Species (used when symbol is provided)"
    )
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly", default="GRCh38", description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org"
    )
    expand: bool = ConfigField(
        title="Expand Transcripts/Exons",
        default=False,
        description="Include transcripts, translations, and exons in the response",
    )
    mane: bool = ConfigField(
        title="MANE Annotations",
        default=False,
        description="Include MANE Select annotations (lookup-by-id only; requires expand=True)",
    )
    phenotypes: bool = ConfigField(
        title="Phenotype Annotations",
        default=False,
        description="Include phenotype annotations on gene records (lookup-by-id only)",
    )
    utr: bool = ConfigField(
        title="UTR Coordinates",
        default=False,
        description="Include UTR coordinates per transcript (lookup-by-id only; requires expand=True)",
    )


class EnsemblLookupOutput(BaseToolOutput):
    """Output from Ensembl lookup.

    Attributes:
        result (EnsemblGene): The looked-up gene record.
        source_url (str): Final Ensembl REST URL that was hit.
        raw_payload (dict[str, Any]): Raw API JSON.
    """

    result: EnsemblGene = Field(description="The looked-up gene record")
    source_url: str = Field(description="Final Ensembl REST URL that was hit")
    raw_payload: dict[str, Any] = Field(default_factory=dict, description="Raw API JSON")

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
            row = self.result.model_dump(exclude={"Transcript"})
            row["Transcript"] = json.dumps([t.model_dump() for t in self.result.Transcript], separators=(",", ":"))
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return
        raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# Tool Implementation
# ============================================================================


def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return EnsemblLookupInput(ensembl_id="ENSG00000012048")


@tool(
    key="ensembl-lookup",
    label="Ensembl Lookup",
    category="database_retrieval",
    input_class=EnsemblLookupInput,
    config_class=EnsemblLookupConfig,
    output_class=EnsemblLookupOutput,
    description="Look up an Ensembl gene record by Ensembl gene ID or gene symbol",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_lookup(
    inputs: EnsemblLookupInput,
    config: EnsemblLookupConfig,
    instance: Any = None,
) -> EnsemblLookupOutput:
    """Fetch a gene record via Ensembl REST.

    Args:
        inputs (EnsemblLookupInput): Ensembl ID or gene symbol.
        config (EnsemblLookupConfig): Species + assembly + expand toggle.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblLookupOutput: Parsed ``EnsemblGene`` plus metadata.
    """
    del instance

    base = base_url_for(config.assembly)
    eid = (inputs.ensembl_id or "").strip()
    sym = (inputs.symbol or "").strip()
    if eid:
        url = f"{base}/lookup/id/{eid}"
        params: dict[str, Any] = {"object_type": "gene"}
        # mane / phenotypes / utr are accepted only by /lookup/id
        if config.mane:
            params["mane"] = "1"
        if config.phenotypes:
            params["phenotypes"] = "1"
        if config.utr:
            params["utr"] = "1"
    else:
        url = f"{base}/lookup/symbol/{config.species}/{sym}"
        params = {}
    if config.expand:
        params["expand"] = "1"

    session = build_session("ensembl-lookup")
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
                f"Ensembl returned non-JSON for lookup at {response.url}; body[:200]={response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Ensembl lookup returned non-dict payload: {type(payload).__name__}")
        return EnsemblLookupOutput(
            result=EnsemblGene.model_validate(payload),
            source_url=response.url,
            raw_payload=payload,
        )
    finally:
        session.close()
