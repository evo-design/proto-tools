"""Version-pinned NCBI Datasets genome reference bundle retrieval."""

import hashlib
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.entities.provenance import AccessLevel, RetrievedArtifact, SourceProvenance
from proto_tools.tools.database_retrieval.ncbi.shared_data_models import NCBIFetchConfig
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField, build_http_session

AnnotationType = Literal["GENOME_FASTA", "GENOME_GFF", "RNA_FASTA", "PROT_FASTA"]
_ACCESSION = re.compile(r"^GC[AF]_\d+\.\d+$")
_MEDIA = {".fna": "text/x-fasta", ".faa": "text/x-fasta", ".gff": "text/x-gff3", ".jsonl": "application/jsonl"}


class NCBIReferenceBundleInput(BaseToolInput):
    """Pinned assembly accession and artifact destination."""

    accession: str = InputField(title="Assembly Accession", description="Versioned GCA/GCF assembly accession")
    destination_dir: str = InputField(
        title="Destination Directory", description="Empty or new directory for fetched artifacts"
    )

    @field_validator("accession")
    @classmethod
    def _versioned(cls, value: str) -> str:
        if not _ACCESSION.fullmatch(value):
            raise ValueError("accession must be a versioned GCA_ or GCF_ accession")
        return value


class NCBIReferenceBundleConfig(NCBIFetchConfig):
    """NCBI Datasets bundle selection.

    Attributes:
        ncbi_api_key: Optional NCBI API key.
        ncbi_email: Optional NCBI contact email.
        annotation_types: Requested reference representations.
        max_download_bytes: Maximum accepted ZIP response size.
    """

    annotation_types: list[AnnotationType] = ConfigField(
        default_factory=lambda: ["GENOME_FASTA", "GENOME_GFF", "RNA_FASTA", "PROT_FASTA"],
        title="Annotation Types",
        description="NCBI Datasets files to retrieve",
    )
    max_download_bytes: int = ConfigField(
        default=2_000_000_000,
        gt=0,
        title="Maximum Download Bytes",
        description="Maximum compressed bundle size",
    )


class NCBIReferenceBundleOutput(BaseToolOutput):
    """Materialized, checksummed NCBI reference artifacts."""

    accession: str
    source_url: str
    destination_dir: str
    artifacts: list[RetrievedArtifact] = Field(default_factory=list)
    dataset_catalog: dict[str, Any] = Field(default_factory=dict)

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Return the default export format."""
        return "json"

    def _export_output(self, export_path: Any, file_format: str) -> None:
        if file_format != "json":
            raise ValueError(f"Unsupported format: {file_format}")
        Path(export_path).with_suffix(".json").write_text(json.dumps(self.model_dump(mode="json"), indent=2))


def example_input() -> Any:
    """Return a small pinned bacterial reference example."""
    return NCBIReferenceBundleInput(accession="GCF_000005845.2", destination_dir="ncbi-reference")


@tool(
    key="ncbi-reference-bundle",
    local_only="HTTP-only artifact retrieval needs no isolated environment",
    label="NCBI Reference Bundle",
    category="database_retrieval",
    input_class=NCBIReferenceBundleInput,
    config_class=NCBIReferenceBundleConfig,
    output_class=NCBIReferenceBundleOutput,
    description="Fetch a version-pinned NCBI Datasets genome reference bundle",
    uses_gpu=False,
    example_input=example_input,
    cacheable=False,
)
def run_ncbi_reference_bundle(
    inputs: NCBIReferenceBundleInput, config: NCBIReferenceBundleConfig, instance: Any = None
) -> NCBIReferenceBundleOutput:
    """Download, safely extract, and checksum a pinned NCBI Datasets bundle."""
    del instance
    destination = Path(inputs.destination_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination_dir must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{inputs.accession}/download"
    headers = {"Accept": "application/zip", "User-Agent": "proto-tools/ncbi-reference-bundle-v1"}
    if config.ncbi_api_key:
        headers["api-key"] = config.ncbi_api_key
    session = build_http_session(http_retries=2, backoff_seconds=1.0, user_agent=headers["User-Agent"])
    try:
        response = session.get(
            url, params={"include_annotation_type": config.annotation_types}, headers=headers, timeout=120
        )
        response.raise_for_status()
        payload = response.content
    finally:
        session.close()
    if len(payload) > config.max_download_bytes:
        raise ValueError("NCBI bundle exceeds max_download_bytes")
    provenance = SourceProvenance(
        provider="NCBI Datasets",
        source_url=url,
        identifier=inputs.accession,
        version=inputs.accession.rsplit(".", 1)[1],
        access_level=AccessLevel.PUBLIC,
        request_parameters={"include_annotation_type": list(config.annotation_types)},
    )
    artifacts: list[RetrievedArtifact] = []
    catalog: dict[str, Any] = {}
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path in NCBI bundle")
            data = archive.read(info)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            item_provenance = provenance.model_copy(
                update={"content_sha256": digest, "media_type": _MEDIA.get(target.suffix, "application/octet-stream")}
            )
            artifacts.append(
                RetrievedArtifact(
                    name=relative.name,
                    path=str(target),
                    content_sha256=digest,
                    media_type=item_provenance.media_type or "application/octet-stream",
                    size_bytes=len(data),
                    provenance=item_provenance,
                )
            )
            if relative.name == "dataset_catalog.json":
                catalog = json.loads(data)
    if not artifacts:
        raise ValueError("NCBI returned an empty reference bundle")
    return NCBIReferenceBundleOutput(
        accession=inputs.accession,
        source_url=url,
        destination_dir=str(destination),
        artifacts=artifacts,
        dataset_catalog=catalog,
    )
