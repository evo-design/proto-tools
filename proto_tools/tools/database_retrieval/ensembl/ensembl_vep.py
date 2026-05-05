"""proto_tools/tools/database_retrieval/ensembl/ensembl_vep.py.

Wraps Ensembl's Variant Effect Predictor REST endpoint (HGVS form):
submits a notation, returns per-transcript consequence predictions.
"""

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from proto_tools.tools.database_retrieval.ensembl.shared_data_models import (
    EnsemblAssembly,
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


class EnsemblVEPConsequence(BaseModel):
    """One variant-effect prediction record.

    Attributes:
        input (str): HGVS string the API echoed back.
        most_severe_consequence (str): Highest-severity consequence term
            (Sequence Ontology: ``intron_variant``, ``missense_variant``,
            ``stop_gained``, …).
        seq_region_name (str | None): Chromosome / contig.
        start (int | None): 1-indexed inclusive genomic start.
        end (int | None): 1-indexed inclusive genomic end.
        strand (int | None): +1 or -1.
        allele_string (str | None): Reference/alternate alleles (e.g. 'G/C').
        transcript_consequences (list[dict[str, Any]]): Per-transcript
            consequence records — kept as raw dicts because the field set
            (sift_prediction, polyphen_prediction, codons, amino_acids, …)
            varies by consequence type and plugin configuration.
        colocated_variants (list[dict[str, Any]]): Co-located known variants
            (rsIDs, frequencies, clinical significance) — also kept raw.
    """

    model_config = ConfigDict(extra="ignore")

    input: str
    most_severe_consequence: str
    seq_region_name: str | None = None
    start: int | None = None
    end: int | None = None
    strand: int | None = None
    allele_string: str | None = None
    transcript_consequences: list[dict[str, Any]] = Field(default_factory=list)
    colocated_variants: list[dict[str, Any]] = Field(default_factory=list)


class EnsemblVEPInput(BaseToolInput):
    """Input for Ensembl VEP.

    Attributes:
        hgvs (str): HGVS notation. Genomic (e.g. ``9:g.22125504G>C``),
            coding (``ENST00000357654:c.5074G>A``), or protein
            (``ENSP00000418960:p.Tyr124Cys``) forms all work.
    """

    hgvs: str = InputField(description="HGVS notation (genomic / coding / protein)")

    @model_validator(mode="after")
    def validate_hgvs(self) -> "EnsemblVEPInput":
        """Reject empty / whitespace-only HGVS."""
        if not self.hgvs.strip():
            raise ValueError("hgvs must be a non-empty HGVS notation")
        return self


class EnsemblVEPConfig(BaseConfig):
    """Configuration for Ensembl VEP.

    Attributes:
        species (EnsemblSpecies): Species slug. Default ``homo_sapiens``.
        assembly (EnsemblAssembly): Genome assembly. ``GRCh38`` (default)
            or ``GRCh37``.
    """

    species: EnsemblSpecies = ConfigField(title="Species", default="homo_sapiens", description="Species slug for VEP")
    assembly: EnsemblAssembly = ConfigField(
        title="Assembly",
        default="GRCh38",
        description="Genome assembly; GRCh37 routes to grch37.rest.ensembl.org",
    )


class EnsemblVEPOutput(BaseToolOutput):
    """Output from Ensembl VEP.

    Attributes:
        consequences (list[EnsemblVEPConsequence]): One record per VEP
            input (Ensembl returns a list even for a single HGVS).
        num_consequences (int): ``len(consequences)``.
        source_url (str): Final URL hit.
        raw_payload (list[dict[str, Any]]): Raw API JSON.
    """

    consequences: list[EnsemblVEPConsequence] = Field(
        default_factory=list, description="One record per VEP input (Ensembl returns a list even for a single HGVS)"
    )
    source_url: str = Field(description="Final Ensembl REST URL that was hit")
    raw_payload: list[dict[str, Any]] = Field(default_factory=list, description="Raw API JSON")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_consequences(self) -> int:
        """``len(self.consequences)`` — derived so the count can't drift."""
        return len(self.consequences)

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
    return EnsemblVEPInput(hgvs="9:g.22125504G>C")


@tool(
    key="ensembl-vep",
    label="Ensembl VEP",
    category="database_retrieval",
    input_class=EnsemblVEPInput,
    config_class=EnsemblVEPConfig,
    output_class=EnsemblVEPOutput,
    description=(
        "Predict variant consequences from an HGVS notation via Ensembl's Variant Effect Predictor REST endpoint"
    ),
    uses_gpu=False,
    example_input=example_input,
    cacheable=True,
)
def run_ensembl_vep(
    inputs: EnsemblVEPInput,
    config: EnsemblVEPConfig,
    instance: Any = None,
) -> EnsemblVEPOutput:
    """Submit an HGVS notation to Ensembl VEP and parse the consequence list.

    Args:
        inputs (EnsemblVEPInput): HGVS notation (genomic / coding / protein).
        config (EnsemblVEPConfig): Species + assembly + optional contact.
        instance (Any): Optional ToolInstance; unused for HTTP-only tools.

    Returns:
        EnsemblVEPOutput: One ``EnsemblVEPConsequence`` per record returned
            by the API (typically one for a single-HGVS query).
    """
    del instance

    base = base_url_for(config.assembly)
    url = f"{base}/vep/{config.species}/hgvs/{quote(inputs.hgvs.strip(), safe='')}"

    session = build_session("ensembl-vep")
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
                f"Ensembl VEP returned non-JSON for {inputs.hgvs!r} at {response.url}; "
                f"body[:200]={response.text[:200]!r}"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"Ensembl VEP returned non-list payload: {type(payload).__name__}")
        consequences: list[EnsemblVEPConsequence] = []
        for i, c in enumerate(payload):
            if not isinstance(c, dict):
                raise ValueError(f"Ensembl VEP payload[{i}] is non-dict: {type(c).__name__}")
            consequences.append(EnsemblVEPConsequence.model_validate(c))
        return EnsemblVEPOutput(
            consequences=consequences,
            source_url=response.url,
            raw_payload=payload,
        )
    finally:
        session.close()
