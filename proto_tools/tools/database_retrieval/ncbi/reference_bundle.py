"""Version-pinned NCBI Datasets genome reference bundle retrieval."""

import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.entities.provenance import (
    AccessLevel,
    ArtifactDerivation,
    RetrievedArtifact,
    SourceProvenance,
)
from proto_tools.tools.database_retrieval.ncbi.shared_data_models import NCBIFetchConfig
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField, build_http_session

AnnotationType = Literal["GENOME_FASTA", "GENOME_GFF", "RNA_FASTA", "PROT_FASTA"]
_ACCESSION = re.compile(r"^GC[AF]_\d+\.\d+$")
_MEDIA = {".fna": "text/x-fasta", ".faa": "text/x-fasta", ".gff": "text/x-gff3", ".jsonl": "application/jsonl"}


class NCBIReferenceBundleInput(BaseToolInput):
    """Pinned assembly accession and artifact destination.

    Attributes:
        accession (str): Versioned GCA or GCF assembly accession.
        destination_dir (str): Empty or new destination for fetched artifacts.
    """

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
        ncbi_api_key (str | None): Optional NCBI API key.
        ncbi_email (str | None): Optional NCBI contact email.
        annotation_types (list[AnnotationType]): Requested reference representations.
        max_download_bytes (int): Maximum accepted compressed and extracted size.
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
    """Materialized, checksummed NCBI reference artifacts.

    Attributes:
        accession (str): Versioned assembly accession.
        source_url (str): NCBI Datasets download endpoint.
        destination_dir (str): Materialized artifact directory.
        archive_artifact (RetrievedArtifact | None): Exact downloaded ZIP descriptor.
        artifacts (list[RetrievedArtifact]): Checksummed extracted files.
        dataset_catalog (dict[str, Any]): NCBI dataset catalog when present.
    """

    accession: str = Field(title="Assembly Accession", description="Versioned assembly accession")
    source_url: str = Field(title="Source URL", description="NCBI Datasets download endpoint")
    destination_dir: str = Field(title="Destination Directory", description="Materialized artifact directory")
    archive_artifact: RetrievedArtifact | None = Field(
        default=None,
        title="Archive Artifact",
        description="Checksum and size descriptor for the exact downloaded ZIP bytes",
    )
    artifacts: list[RetrievedArtifact] = Field(
        default_factory=list, title="Artifacts", description="Checksummed extracted files"
    )
    dataset_catalog: dict[str, Any] = Field(
        default_factory=dict, title="Dataset Catalog", description="NCBI dataset catalog"
    )

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
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"destination_dir must be an empty directory or not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{inputs.accession}/download"
    headers = {"Accept": "application/zip", "User-Agent": "proto-tools/ncbi-reference-bundle-v1"}
    if config.ncbi_api_key:
        headers["api-key"] = config.ncbi_api_key
    session = build_http_session(http_retries=2, backoff_seconds=1.0, user_agent=headers["User-Agent"])
    try:
        response = session.get(
            url,
            params={"include_annotation_type": config.annotation_types},
            headers=headers,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > config.max_download_bytes:
            raise ValueError("NCBI bundle exceeds max_download_bytes")
        # Python 3.10's SpooledTemporaryFile lacks the seekable attribute that
        # zipfile requires when opening members. A disk-backed temporary file
        # preserves bounded-memory streaming across the supported interpreters.
        download = tempfile.TemporaryFile()  # noqa: SIM115
        downloaded = 0
        archive_hasher = hashlib.sha256()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > config.max_download_bytes:
                raise ValueError("NCBI bundle exceeds max_download_bytes")
            archive_hasher.update(chunk)
            download.write(chunk)
        download.seek(0)
    finally:
        session.close()
    provenance = SourceProvenance(
        provider="NCBI Datasets",
        source_url=url,
        identifier=inputs.accession,
        version=inputs.accession.rsplit(".", 1)[1],
        content_sha256=archive_hasher.hexdigest(),
        media_type="application/zip",
        access_level=AccessLevel.PUBLIC,
        request_parameters={"include_annotation_type": list(config.annotation_types)},
    )
    archive_artifact = RetrievedArtifact(
        name=f"NCBI Datasets {inputs.accession} reference bundle",
        content_sha256=archive_hasher.hexdigest(),
        media_type="application/zip",
        size_bytes=downloaded,
        provenance=provenance,
    )
    artifacts: list[RetrievedArtifact] = []
    catalog: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ncbi-reference-", dir=destination.parent) as staging_name:
        staging = Path(staging_name)
        with zipfile.ZipFile(download) as archive:
            extracted_bytes = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe path in NCBI bundle")
                extracted_bytes += info.file_size
                if extracted_bytes > config.max_download_bytes:
                    raise ValueError("extracted NCBI bundle exceeds max_download_bytes")
                data = archive.read(info)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                media_type = _MEDIA.get(target.suffix, "application/octet-stream")
                artifacts.append(
                    RetrievedArtifact(
                        name=relative.name,
                        path=str(destination / relative),
                        content_sha256=digest,
                        media_type=media_type,
                        size_bytes=len(data),
                        provenance=provenance,
                        derivation=ArtifactDerivation(
                            source_sha256=[archive_artifact.content_sha256],
                            tool="zipfile",
                            parameters={"operation": "extract-member", "member": info.filename},
                        ),
                    )
                )
                if relative.name == "dataset_catalog.json":
                    catalog = json.loads(data)
        if not artifacts:
            raise ValueError("NCBI returned an empty reference bundle")
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    download.close()
    return NCBIReferenceBundleOutput(
        accession=inputs.accession,
        source_url=url,
        destination_dir=str(destination),
        archive_artifact=archive_artifact,
        artifacts=artifacts,
        dataset_catalog=catalog,
    )
