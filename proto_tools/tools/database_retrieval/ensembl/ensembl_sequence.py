"""proto_tools/tools/database_retrieval/ensembl/ensembl_sequence.py.

Wraps Ensembl REST ``/sequence/id/{id}`` — DNA / cDNA / CDS / protein
sequence retrieval keyed by an Ensembl gene, transcript, or protein ID.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

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

    ensembl_id: str = InputField(title="Ensembl ID", description="Ensembl ID (ENSG..., ENST..., ENSP...)")

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
        sequence_type (EnsemblSequenceType): What to return — ``genomic``
            (DNA + UTRs + introns), ``cdna`` (spliced mRNA + UTRs),
            ``cds`` (spliced coding only), ``protein`` (translation).
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            or ``GRCh37``.
        multiple_sequences (bool): Return all sequences when an ID maps to
            multiple records (e.g. patches, alternative haplotypes).
        mask (Literal['hard', 'soft'] | None): Mask repeats in the returned
            sequence. ``hard`` replaces with ``N``; ``soft`` lowercases.
            Genomic sequence_type only; mutually exclusive with mask_feature.
        mask_feature (bool): Mask introns (when ``sequence_type='genomic'``)
            or UTRs (when ``sequence_type='cdna'``) so the primary feature
            stands out. Mutually exclusive with ``mask``.
        expand_3prime (int | None): Bases to add to the 3' end (genomic only,
            incompatible with ``end``).
        expand_5prime (int | None): Bases to add to the 5' end (genomic only,
            incompatible with ``start``).
        start (int | None): 1-indexed start trim coordinate (incompatible
            with ``expand_5prime``).
        end (int | None): 1-indexed end trim coordinate (incompatible with
            ``expand_3prime``).
    """

    sequence_type: EnsemblSequenceType = ConfigField(
        title="Sequence Type",
        default="genomic",
        description="genomic = DNA+UTR+introns; cdna = mRNA+UTR; cds = coding only; protein = translation",
    )
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly", default="GRCh38", description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org"
    )
    multiple_sequences: bool = ConfigField(
        title="Allow Multiple Sequences",
        default=False,
        description="Return all sequences for IDs that map to multiple records",
    )
    mask: Literal["hard", "soft"] | None = ConfigField(
        title="Repeat Mask", default=None, description="Mask repeats: 'hard' (N) or 'soft' (lowercase); genomic only"
    )
    mask_feature: bool = ConfigField(
        title="Mask Features",
        default=False,
        description="Mask introns (genomic) or UTRs (cdna) so the primary feature stands out",
    )
    expand_3prime: int | None = ConfigField(
        title="Expand 3' End",
        default=None,
        ge=0,
        description="Add bases to the 3' end (genomic only; incompatible with 'end')",
    )
    expand_5prime: int | None = ConfigField(
        title="Expand 5' End",
        default=None,
        ge=0,
        description="Add bases to the 5' end (genomic only; incompatible with 'start')",
    )
    start: int | None = ConfigField(
        title="Trim Start",
        default=None,
        ge=1,
        description="1-indexed start trim coordinate (incompatible with expand_5prime)",
    )
    end: int | None = ConfigField(
        title="Trim End",
        default=None,
        ge=1,
        description="1-indexed end trim coordinate (incompatible with expand_3prime)",
    )

    @model_validator(mode="after")
    def _check_mutually_exclusive(self) -> "EnsemblSequenceConfig":
        """Reject incompatible combinations (mask vs mask_feature; expand_X vs start/end)."""
        if self.mask is not None and self.mask_feature:
            raise ValueError("Set either 'mask' or 'mask_feature', not both")
        if self.expand_5prime is not None and self.start is not None:
            raise ValueError("'expand_5prime' and 'start' are mutually exclusive")
        if self.expand_3prime is not None and self.end is not None:
            raise ValueError("'expand_3prime' and 'end' are mutually exclusive")
        return self


class EnsemblSequenceOutput(BaseToolOutput):
    """Output from Ensembl sequence fetch.

    Attributes:
        results (list[EnsemblSequence]): Fetched sequence record(s). Length 1
            unless ``multiple_sequences=True`` and the ID maps to more than one.
        source_url (str): Final Ensembl REST URL that was hit.
        raw_payload (list[dict[str, Any]]): Raw API JSON, always wrapped in a list.
    """

    results: list[EnsemblSequence] = Field(
        default_factory=list, title="Sequence Records", description="Fetched sequence record(s)"
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
            rows = [r.model_dump() for r in self.results]
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
    params: dict[str, Any] = {"type": config.sequence_type}
    if config.multiple_sequences:
        params["multiple_sequences"] = "1"
    if config.mask is not None:
        params["mask"] = config.mask
    if config.mask_feature:
        params["mask_feature"] = "1"
    if config.expand_3prime is not None:
        params["expand_3prime"] = str(config.expand_3prime)
    if config.expand_5prime is not None:
        params["expand_5prime"] = str(config.expand_5prime)
    if config.start is not None:
        params["start"] = str(config.start)
    if config.end is not None:
        params["end"] = str(config.end)

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

        if isinstance(payload, list):
            records = [EnsemblSequence.model_validate(item) for item in payload]
            raw_list = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            records = [EnsemblSequence.model_validate(payload)]
            raw_list = [payload]
        else:
            raise ValueError(f"Ensembl sequence returned non-dict/list payload: {type(payload).__name__}")

        return EnsemblSequenceOutput(
            results=records,
            source_url=response.url,
            raw_payload=raw_list,
        )
    finally:
        session.close()
