"""proto_tools/tools/database_retrieval/ensembl/ensembl_overlap.py.

Wraps Ensembl REST ``/overlap/id/{id}`` — features overlapping the given
ID's region. Returns a list of typed records with the full upstream dict
preserved in ``raw`` since per-feature shapes diverge.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
    EnsemblBiotype,
    EnsemblOverlapFeature,
    EnsemblOverlapFeatureRecord,
    EnsemblSOTerm,
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


class EnsemblOverlapInput(BaseToolInput):
    """Input for Ensembl region-overlap query.

    Attributes:
        ensembl_id (str): Ensembl ID whose region to query for overlapping features.
    """

    ensembl_id: str = InputField(title="Ensembl ID", description="Ensembl ID (ENSG..., ENST..., ENSP...)")

    @field_validator("ensembl_id")
    @classmethod
    def validate_ensembl_id(cls, value: str) -> str:
        """Reject blank stable IDs before constructing an Ensembl URL."""
        if not value.strip():
            raise ValueError("ensembl_id cannot be blank")
        return value


class EnsemblOverlapConfig(BaseConfig):
    """Configuration for Ensembl overlap query.

    Attributes:
        overlap_feature (EnsemblOverlapFeature): Type of feature to retrieve
            (e.g. gene, transcript, exon, regulatory, variation).
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            or ``GRCh37``.
        biotype (EnsemblBiotype | None): Restrict to a biotype (e.g.
            ``protein_coding``); most useful when ``overlap_feature`` is
            ``gene`` or ``transcript``.
        so_term (EnsemblSOTerm | None): Restrict variation features by
            Sequence Ontology consequence (e.g. ``missense_variant``).
        variant_set (str | None): Restrict variation features to a named
            variant set (e.g. ``ClinVar``).
    """

    overlap_feature: EnsemblOverlapFeature = ConfigField(
        title="Overlap Feature",
        default="gene",
        description="Type of feature to retrieve (e.g. gene, transcript, exon, regulatory, variation)",
    )
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly", default="GRCh38", description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org"
    )
    biotype: EnsemblBiotype | None = ConfigField(
        title="Biotype Filter",
        default=None,
        description="Restrict to a biotype (e.g. 'protein_coding'); most useful for gene/transcript features",
    )
    so_term: EnsemblSOTerm | None = ConfigField(
        title="SO Term Filter",
        default=None,
        description="Restrict variation features by SO consequence (e.g. 'missense_variant')",
    )
    variant_set: str | None = ConfigField(
        title="Variant Set Filter",
        default=None,
        description="Restrict variation features to a named variant set (e.g. 'ClinVar')",
    )


class EnsemblOverlapOutput(BaseToolOutput):
    """Output from Ensembl overlap query.

    Attributes:
        result (list[EnsemblOverlapFeatureRecord]): Features overlapping
            the queried region; each carries the full upstream dict in
            ``raw`` for feature-specific keys.
        source_url (str): Final Ensembl REST URL that was hit.
        raw_payload (list[dict[str, Any]]): Raw API JSON.
    """

    result: list[EnsemblOverlapFeatureRecord] = Field(
        default_factory=list,
        title="Overlapping Features",
        description="Overlapping features (typed common fields + raw dict)",
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
            rows = [
                {**r.model_dump(exclude={"raw"}), "raw": json.dumps(r.raw, separators=(",", ":"))} for r in self.result
            ]
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
    return EnsemblOverlapInput(ensembl_id="ENSG00000012048")


@tool(
    key="ensembl-overlap",
    label="Ensembl Overlap",
    category="database_retrieval",
    input_class=EnsemblOverlapInput,
    config_class=EnsemblOverlapConfig,
    output_class=EnsemblOverlapOutput,
    description="Fetch features overlapping an Ensembl region (default: gene; supports exon, regulatory, motif, variation, …)",
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_overlap(
    inputs: EnsemblOverlapInput,
    config: EnsemblOverlapConfig,
    instance: Any = None,
) -> EnsemblOverlapOutput:
    """Fetch overlapping features for a region from Ensembl REST.

    Args:
        inputs (EnsemblOverlapInput): Ensembl ID whose region to query.
        config (EnsemblOverlapConfig): Feature class + assembly.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblOverlapOutput: List of typed records + raw upstream dicts.
    """
    del instance

    base = base_url_for(config.assembly)
    url = f"{base}/overlap/id/{inputs.ensembl_id.strip()}"
    params: dict[str, Any] = {"feature": config.overlap_feature}
    if config.biotype:
        params["biotype"] = config.biotype
    if config.so_term:
        params["so_term"] = config.so_term
    if config.variant_set:
        params["variant_set"] = config.variant_set

    session = build_session("ensembl-overlap")
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
                f"Ensembl returned non-JSON for overlap at {response.url}; body[:200]={response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"Ensembl overlap returned non-list payload: {type(payload).__name__}")
        records: list[EnsemblOverlapFeatureRecord] = []
        for i, f in enumerate(payload):
            if not isinstance(f, dict):
                raise ValueError(f"Ensembl overlap payload[{i}] is non-dict: {type(f).__name__}")
            records.append(EnsemblOverlapFeatureRecord.model_validate({**f, "raw": f}))
        return EnsemblOverlapOutput(result=records, source_url=response.url, raw_payload=payload)
    finally:
        session.close()
