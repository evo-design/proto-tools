"""Version-pinned NCBI Datasets genome reference bundle retrieval."""

import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from proto_tools.tools.database_retrieval.ncbi.shared_data_models import NCBIFetchConfig
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import BaseToolInput, BaseToolOutput, ConfigField, InputField, build_http_session

AnnotationType = Literal["GENOME_FASTA", "GENOME_GFF", "RNA_FASTA", "PROT_FASTA"]
_ACCESSION = re.compile(r"^GC[AF]_\d+\.\d+$")


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
    """NCBI reference files extracted into ``destination_dir``.

    Attributes:
        accession (str): Versioned assembly accession.
        source_url (str): NCBI Datasets download endpoint.
        destination_dir (str): Directory the bundle was extracted into.
        files (list[str]): Paths of the extracted files.
        dataset_catalog (dict[str, Any]): NCBI dataset catalog when present.
    """

    accession: str = Field(title="Assembly Accession", description="Versioned assembly accession")
    source_url: str = Field(title="Source URL", description="NCBI Datasets download endpoint")
    destination_dir: str = Field(title="Destination Directory", description="Directory the bundle was extracted into")
    files: list[str] = Field(default_factory=list, title="Files", description="Paths of the extracted files")
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
    """Download and safely extract a pinned NCBI Datasets bundle."""
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
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > config.max_download_bytes:
                raise ValueError("NCBI bundle exceeds max_download_bytes")
            download.write(chunk)
        download.seek(0)
    finally:
        session.close()
    files: list[str] = []
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
                files.append(str(destination / relative))
                if relative.name == "dataset_catalog.json":
                    catalog = json.loads(data)
        if not files:
            raise ValueError("NCBI returned an empty reference bundle")
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    download.close()
    return NCBIReferenceBundleOutput(
        accession=inputs.accession,
        source_url=url,
        destination_dir=str(destination),
        files=files,
        dataset_catalog=catalog,
    )
