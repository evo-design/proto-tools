"""Version-pinned NCBI Datasets genome reference bundle retrieval."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import IO, Any, Literal

import requests
from pydantic import Field, field_validator

from proto_tools.databases.registry import get_databases_root
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import (
    BaseConfig,
    BaseToolInput,
    BaseToolOutput,
    ConfigField,
    InputField,
    build_http_session,
    request_with_retry,
)

logger = logging.getLogger(__name__)

AnnotationType = Literal["GENOME_FASTA", "GENOME_GFF", "RNA_FASTA", "PROT_FASTA"]

_DATASETS_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession"
_USER_AGENT = "proto-tools/ncbi-reference-bundle-v1"
_REQUEST_TIMEOUT_SECONDS = 120
_HTTP_RETRIES = 2
_BACKOFF_SECONDS = 1.0
_CHUNK_BYTES = 1024 * 1024

# The annotation types this bundle was downloaded with, so a later call can reuse it.
_REQUEST_RECORD = "proto_request.json"
_CATALOG = "ncbi_dataset/data/dataset_catalog.json"

# dataset_catalog.json file types for each output path field.
_CATALOG_FIELDS = {
    "GENOMIC_NUCLEOTIDE_FASTA": "genome_fasta",
    "GFF3": "genome_gff",
    "RNA_NUCLEOTIDE_FASTA": "rna_fasta",
    "PROTEIN_FASTA": "protein_fasta",
}

# A versioned assembly accession, bare or inside an NCBI page link.
_ACCESSION = re.compile(r"GC[AF]_\d{9}\.\d+")
_UNVERSIONED = re.compile(r"GC[AF]_\d{9}")


class NCBIReferenceBundleInput(BaseToolInput):
    """Pinned assembly accession and optional destination.

    Attributes:
        accession (str): Versioned GCA or GCF assembly accession, bare or as an NCBI page link.
        destination_dir (str | None): Directory to extract into; defaults to
            ``$PROTO_DATABASES_DIR/ncbi/<accession>``.
    """

    accession: str = InputField(
        title="Assembly Accession",
        description="Versioned GCA_/GCF_ assembly accession, or an NCBI page link containing one",
    )
    destination_dir: str | None = InputField(
        default=None,
        title="Destination Directory",
        description="Directory to extract into; defaults to the shared databases directory",
    )

    @field_validator("accession")
    @classmethod
    def _parse_accession(cls, value: str) -> str:
        """Read a versioned accession out of a bare ID or an NCBI link."""
        text = value.strip().upper()
        match = _ACCESSION.search(text)
        if match is not None:
            return match.group(0)
        unversioned = _UNVERSIONED.search(text)
        if unversioned is not None:
            raise ValueError(
                f"{unversioned.group(0)} has no version; pin one, e.g. {unversioned.group(0)}.1, "
                "so the bundle cannot change underneath you"
            )
        raise ValueError(f"{value!r} is not a GCA_ or GCF_ assembly accession (e.g. GCF_000005845.2)")


class NCBIReferenceBundleConfig(BaseConfig):
    """NCBI Datasets bundle selection.

    Attributes:
        annotation_types (list[AnnotationType]): Reference files to download.
        max_download_bytes (int): Largest compressed bundle to download.
        ncbi_api_key (str | None): Optional NCBI API key; defaults to ``NCBI_API_KEY``.
    """

    annotation_types: list[AnnotationType] = ConfigField(
        default_factory=lambda: ["GENOME_FASTA", "GENOME_GFF", "RNA_FASTA", "PROT_FASTA"],
        min_length=1,
        title="Annotation Types",
        description="NCBI Datasets reference files to download",
    )
    max_download_bytes: int = ConfigField(
        default=5_000_000_000,
        gt=0,
        title="Maximum Download Bytes",
        description="Largest compressed bundle to download; the human assembly is about 1.2 GB",
    )
    ncbi_api_key: str | None = ConfigField(
        title="NCBI API Key",
        description="Optional NCBI API key; raises the request rate limit.",
        credential="NCBI_API_KEY",
    )


class NCBIReferenceBundleOutput(BaseToolOutput):
    """Paths to an extracted NCBI reference bundle.

    Attributes:
        accession (str): Versioned assembly accession.
        source_url (str): NCBI Datasets download endpoint.
        directory (str): Directory holding the extracted bundle.
        genome_fasta (str | None): Genomic nucleotide FASTA.
        genome_gff (str | None): GFF3 genome annotation.
        rna_fasta (str | None): Transcript nucleotide FASTA.
        protein_fasta (str | None): Protein FASTA.
    """

    accession: str = Field(title="Assembly Accession", description="Versioned assembly accession")
    source_url: str = Field(title="Source URL", description="NCBI Datasets download endpoint")
    directory: str = Field(title="Directory", description="Directory holding the extracted bundle")
    genome_fasta: str | None = Field(default=None, title="Genome FASTA", description="Genomic nucleotide FASTA")
    genome_gff: str | None = Field(default=None, title="Genome GFF", description="GFF3 genome annotation")
    rna_fasta: str | None = Field(default=None, title="RNA FASTA", description="Transcript nucleotide FASTA")
    protein_fasta: str | None = Field(default=None, title="Protein FASTA", description="Protein FASTA")

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


def _is_complete(directory: Path, annotation_types: list[AnnotationType]) -> bool:
    """Whether ``directory`` holds a finished bundle downloaded with at least these annotation types."""
    record = directory / _REQUEST_RECORD
    if not record.is_file():
        return False
    return set(annotation_types) <= set(json.loads(record.read_text())["annotation_types"])


def _check_available(
    session: requests.Session,
    accession: str,
    params: dict[str, Any],
    headers: dict[str, str],
    max_bytes: int,
) -> None:
    """Fail before downloading when NCBI has no such assembly or the bundle is too large."""
    response = request_with_retry(
        lambda: session.get(
            f"{_DATASETS_BASE}/{accession}/download_summary",
            params=params,
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        ),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    response.raise_for_status()
    summary = response.json()
    # NCBI answers an unknown accession with a summary that has no record_count.
    if not summary.get("record_count"):
        raise ValueError(f"NCBI Datasets has no assembly {accession}")
    estimated_bytes = summary["hydrated"]["estimated_file_size_mb"] * 1_000_000
    if estimated_bytes > max_bytes:
        raise ValueError(
            f"{accession} bundle is about {estimated_bytes / 1e9:.1f} GB, over max_download_bytes ({max_bytes:,})"
        )


def _download(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    max_bytes: int,
    sink: IO[bytes],
) -> None:
    """Stream the bundle into ``sink``, stopping once it passes ``max_bytes``."""
    response = request_with_retry(
        lambda: session.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS, stream=True),
        retries=_HTTP_RETRIES,
        backoff_seconds=_BACKOFF_SECONDS,
    )
    response.raise_for_status()
    downloaded = 0
    for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
        downloaded += len(chunk)
        if downloaded > max_bytes:
            raise ValueError(f"NCBI bundle exceeds max_download_bytes ({max_bytes:,})")
        sink.write(chunk)


def _extract(archive: zipfile.ZipFile, staging: Path) -> None:
    """Extract every member into ``staging``, refusing paths that would escape it."""
    for info in archive.infolist():
        relative = Path(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path in NCBI bundle: {info.filename}")
        if info.is_dir():
            continue
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, _CHUNK_BYTES)


def _output(accession: str, source_url: str, directory: Path) -> NCBIReferenceBundleOutput:
    """Locate each reference file through the bundle's dataset catalog."""
    catalog = json.loads((directory / _CATALOG).read_text())
    paths = {
        _CATALOG_FIELDS[entry["fileType"]]: str(directory / "ncbi_dataset" / "data" / entry["filePath"])
        for assembly in catalog["assemblies"]
        for entry in assembly["files"]
        if entry["fileType"] in _CATALOG_FIELDS
    }
    return NCBIReferenceBundleOutput.model_validate(
        {"accession": accession, "source_url": source_url, "directory": str(directory), **paths}
    )


def example_input() -> NCBIReferenceBundleInput:
    """Return the E. coli K-12 MG1655 reference (about 3 MB compressed)."""
    return NCBIReferenceBundleInput(accession="GCF_000005845.2")


@tool(
    key="ncbi-reference-bundle",
    local_only="ncbi-reference-bundle writes reference files to local disk and needs no isolated environment",
    label="NCBI Reference Bundle",
    category="database_retrieval",
    input_class=NCBIReferenceBundleInput,
    config_class=NCBIReferenceBundleConfig,
    output_class=NCBIReferenceBundleOutput,
    description="Download a version-pinned NCBI Datasets genome reference bundle",
    uses_gpu=False,
    example_input=example_input,
    cacheable=False,
)
def run_ncbi_reference_bundle(
    inputs: NCBIReferenceBundleInput,
    config: NCBIReferenceBundleConfig,
    instance: Any = None,
) -> NCBIReferenceBundleOutput:
    """Download and extract a pinned NCBI Datasets bundle, reusing one already on disk.

    Args:
        inputs (NCBIReferenceBundleInput): Versioned accession and optional destination.
        config (NCBIReferenceBundleConfig): Annotation types, size limit, and API key.
        instance (Any): Unused; kept for tool API consistency.

    Returns:
        NCBIReferenceBundleOutput: Paths to the extracted reference files.
    """
    del instance
    accession = inputs.accession
    source_url = f"{_DATASETS_BASE}/{accession}/download"
    destination = (
        Path(inputs.destination_dir).expanduser().resolve()
        if inputs.destination_dir is not None
        else get_databases_root() / "ncbi" / accession
    )
    if _is_complete(destination, config.annotation_types):
        logger.info("Reusing the %s bundle in %s", accession, destination)
        return _output(accession, source_url, destination)
    occupied = destination.exists() and (not destination.is_dir() or any(destination.iterdir()))
    if occupied and not (destination / _REQUEST_RECORD).is_file():
        raise FileExistsError(f"destination_dir must be empty, new, or an earlier bundle: {destination}")

    params = {"include_annotation_type": config.annotation_types}
    headers = {"Accept": "application/zip"}
    if config.ncbi_api_key:
        headers["api-key"] = config.ncbi_api_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = build_http_session(http_retries=_HTTP_RETRIES, backoff_seconds=_BACKOFF_SECONDS, user_agent=_USER_AGENT)
    with (
        tempfile.TemporaryFile() as download,
        tempfile.TemporaryDirectory(prefix=f"{accession}-", dir=destination.parent) as staging_name,
    ):
        try:
            _check_available(session, accession, params, headers, config.max_download_bytes)
            _download(session, source_url, params, headers, config.max_download_bytes, download)
        finally:
            session.close()
        download.seek(0)
        staging = Path(staging_name) / "bundle"
        staging.mkdir()
        with zipfile.ZipFile(download) as archive:
            _extract(archive, staging)
        (staging / _REQUEST_RECORD).write_text(json.dumps({"annotation_types": config.annotation_types}))
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    return _output(accession, source_url, destination)
