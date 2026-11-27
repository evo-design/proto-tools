"""Tests for staged NCBI reference-bundle retrieval."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from urllib3 import HTTPResponse

from proto_tools.tools.database_retrieval.ncbi import reference_bundle as module
from proto_tools.tools.database_retrieval.ncbi.reference_bundle import (
    NCBIReferenceBundleConfig,
    NCBIReferenceBundleInput,
    run_ncbi_reference_bundle,
)

_ACCESSION = "GCF_000005845.2"
_CATALOG = {
    "apiVersion": "V2",
    "assemblies": [
        {"files": [{"filePath": "assembly_data_report.jsonl", "fileType": "DATA_REPORT"}]},
        {
            "accession": _ACCESSION,
            "files": [
                {"filePath": f"{_ACCESSION}/genomic.fna", "fileType": "GENOMIC_NUCLEOTIDE_FASTA"},
                {"filePath": f"{_ACCESSION}/genomic.gff", "fileType": "GFF3"},
                {"filePath": f"{_ACCESSION}/genomic.gbff", "fileType": "GENBANK_FLAT_FILE"},
            ],
        },
    ],
}
_REPORT = {
    "accession": _ACCESSION,
    "organism": {"taxId": 511145, "organismName": "Escherichia coli str. K-12 substr. MG1655"},
    "assemblyInfo": {
        "assemblyName": "ASM584v2",
        "assemblyLevel": "Complete Genome",
        "assemblyStatus": "current",
        "refseqCategory": "reference genome",
    },
    "assemblyStats": {"totalSequenceLength": "4641652", "numberOfContigs": 1, "gcPercent": 51},
}


def _archive(extra: dict[str, str] | None = None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("README.md", "NCBI Datasets\n")
        archive.writestr("ncbi_dataset/data/dataset_catalog.json", json.dumps(_CATALOG))
        archive.writestr(f"ncbi_dataset/data/{_ACCESSION}/genomic.fna", ">NC_000913.3\nACGT\n")
        archive.writestr(f"ncbi_dataset/data/{_ACCESSION}/genomic.gff", "##gff-version 3\n")
        archive.writestr(f"ncbi_dataset/data/{_ACCESSION}/genomic.gbff", "LOCUS       NC_000913\n")
        archive.writestr("ncbi_dataset/data/assembly_data_report.jsonl", json.dumps(_REPORT) + "\n")
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return payload.getvalue()


class _Response:
    def __init__(self, url: str, *, summary: dict[str, Any] | None = None, payload: bytes = b"") -> None:
        self.url = url
        self.summary = summary
        # A call under a deadline streams the body through ``raw``; without one, ``iter_content``.
        self.raw = HTTPResponse(body=io.BytesIO(payload), preload_content=False)
        self.payload = payload

    def iter_content(self, chunk_size: int) -> Any:
        return (self.payload[i : i + chunk_size] for i in range(0, len(self.payload), chunk_size))

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any] | None:
        return self.summary


class _Session:
    """Fake NCBI Datasets: answers download_summary and the bundle download, recording each URL."""

    def __init__(self, payload: bytes, summary: dict[str, Any] | None = None) -> None:
        self.payload = payload
        self.summary = summary or {"record_count": 1, "hydrated": {"estimated_file_size_mb": 1}}
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _Response:
        self.urls.append(url)
        if url.endswith("/download_summary"):
            return _Response(url, summary=self.summary)
        return _Response(url, payload=self.payload)

    def close(self) -> None:
        pass


@pytest.fixture
def databases_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "databases"
    monkeypatch.setenv("PROTO_DATABASES_DIR", str(root))
    return root


def _serve(monkeypatch: pytest.MonkeyPatch, session: _Session) -> _Session:
    monkeypatch.setattr(module, "build_http_session", lambda **_kwargs: session)
    return session


@pytest.mark.parametrize(
    "copied",
    [_ACCESSION, " gcf_000005845.2 ", f"https://www.ncbi.nlm.nih.gov/datasets/genome/{_ACCESSION}/"],
)
def test_accession_accepts_ids_as_users_copy_them(copied: str) -> None:
    assert NCBIReferenceBundleInput(accession=copied).accession == _ACCESSION


def test_unversioned_accession_asks_for_a_version() -> None:
    with pytest.raises(ValidationError, match=r"GCF_000005845 has no version"):
        NCBIReferenceBundleInput(accession="GCF_000005845")


def test_bundle_extracts_into_databases_root_with_typed_paths_and_summary(
    monkeypatch: pytest.MonkeyPatch, databases_root: Path
) -> None:
    _serve(monkeypatch, _Session(_archive()))

    output = run_ncbi_reference_bundle(NCBIReferenceBundleInput(accession=_ACCESSION), NCBIReferenceBundleConfig())

    directory = databases_root / "ncbi" / _ACCESSION
    assert output.directory == str(directory)
    assert output.genome_fasta == str(directory / "ncbi_dataset" / "data" / _ACCESSION / "genomic.fna")
    assert Path(output.genome_fasta).read_text() == ">NC_000913.3\nACGT\n"
    assert output.genome_gff is not None and Path(output.genome_gff).is_file()
    assert output.genome_gbff is not None and Path(output.genome_gbff).read_text().startswith("LOCUS")
    assert output.rna_fasta is None  # not in this assembly's catalog
    assert output.source_url.endswith(f"/{_ACCESSION}/download")
    assert output.assembly.organism_name == "Escherichia coli str. K-12 substr. MG1655"
    assert output.assembly.total_sequence_length == 4641652


def test_existing_bundle_is_reused_without_downloading(monkeypatch: pytest.MonkeyPatch, databases_root: Path) -> None:
    _serve(monkeypatch, _Session(_archive()))
    first = run_ncbi_reference_bundle(NCBIReferenceBundleInput(accession=_ACCESSION), NCBIReferenceBundleConfig())
    second_session = _serve(monkeypatch, _Session(_archive()))

    second = run_ncbi_reference_bundle(
        NCBIReferenceBundleInput(accession=_ACCESSION),
        NCBIReferenceBundleConfig(annotation_types=["GENOME_FASTA"]),
    )

    assert second_session.urls == []
    assert second.genome_fasta == first.genome_fasta


def test_unknown_accession_fails_before_downloading(monkeypatch: pytest.MonkeyPatch, databases_root: Path) -> None:
    session = _serve(monkeypatch, _Session(_archive(), summary={"hydrated": {"url": "..."}}))

    with pytest.raises(ValueError, match=r"has no assembly GCF_999999999\.1"):
        run_ncbi_reference_bundle(NCBIReferenceBundleInput(accession="GCF_999999999.1"), NCBIReferenceBundleConfig())
    assert len(session.urls) == 1


@pytest.mark.parametrize("estimated_mb", [1, 5000])
def test_oversized_bundle_is_refused_and_leaves_nothing(
    monkeypatch: pytest.MonkeyPatch, databases_root: Path, estimated_mb: int
) -> None:
    """Refused from NCBI's size estimate up front, or mid-stream when the estimate was low."""
    summary = {"record_count": 1, "hydrated": {"estimated_file_size_mb": estimated_mb}}
    _serve(monkeypatch, _Session(_archive(), summary=summary))

    with pytest.raises(ValueError, match="max_download_bytes"):
        run_ncbi_reference_bundle(
            NCBIReferenceBundleInput(accession=_ACCESSION),
            NCBIReferenceBundleConfig(max_download_bytes=100),
        )
    assert not (databases_root / "ncbi" / _ACCESSION).exists()


def test_archive_paths_outside_the_bundle_are_refused(monkeypatch: pytest.MonkeyPatch, databases_root: Path) -> None:
    _serve(monkeypatch, _Session(_archive({"../escaped.txt": "x"})))

    with pytest.raises(ValueError, match="unsafe path"):
        run_ncbi_reference_bundle(NCBIReferenceBundleInput(accession=_ACCESSION), NCBIReferenceBundleConfig())
    assert not (databases_root / "ncbi" / "escaped.txt").exists()


def test_unrelated_nonempty_destination_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "mine.txt").write_text("keep me")
    session = _serve(monkeypatch, _Session(_archive()))

    with pytest.raises(FileExistsError, match="must be empty"):
        run_ncbi_reference_bundle(
            NCBIReferenceBundleInput(accession=_ACCESSION, destination_dir=str(tmp_path)),
            NCBIReferenceBundleConfig(),
        )
    assert session.urls == []
    assert (tmp_path / "mine.txt").read_text() == "keep me"


def test_remote_devices_are_refused() -> None:
    """The bundle lands on the running machine's disk, so a hosted run would return unreachable paths."""
    reason = NCBIReferenceBundleConfig().remote_unsupported_reason("modal")

    assert reason is not None and "local proto-tools MCP server or CLI" in reason


@pytest.mark.network
@pytest.mark.uses_env
def test_reference_bundle_live(tmp_path: Path) -> None:
    output = run_ncbi_reference_bundle(
        NCBIReferenceBundleInput(accession=_ACCESSION, destination_dir=str(tmp_path / "ecoli")),
        NCBIReferenceBundleConfig(annotation_types=["GENOME_FASTA"]),
    )

    assert output.genome_fasta is not None
    assert Path(output.genome_fasta).read_text().startswith(">NC_000913.3")
    assert output.assembly.tax_id == 511145
