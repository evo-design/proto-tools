"""Tests for staged NCBI reference-bundle retrieval."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from proto_tools.tools.database_retrieval.ncbi.reference_bundle import (
    NCBIReferenceBundleConfig,
    NCBIReferenceBundleInput,
    run_ncbi_reference_bundle,
)


def _archive() -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("ncbi_dataset/data/genome.fna", ">ref\nACGT\n")
        archive.writestr("ncbi_dataset/data/dataset_catalog.json", json.dumps({"assemblies": []}))
    return payload.getvalue()


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        yield from (self.payload[index : index + chunk_size] for index in range(0, len(self.payload), chunk_size))


class _Session:
    def __init__(self, response: _Response):
        self.response = response

    def get(self, *args, **kwargs):
        assert kwargs["stream"] is True
        return self.response

    def close(self) -> None:
        pass


def test_reference_bundle_extracts_files_into_destination(monkeypatch, tmp_path):
    payload = _archive()
    monkeypatch.setattr(
        "proto_tools.tools.database_retrieval.ncbi.reference_bundle.build_http_session",
        lambda **kwargs: _Session(_Response(payload)),
    )
    destination = tmp_path / "reference"
    output = run_ncbi_reference_bundle(
        NCBIReferenceBundleInput(accession="GCF_000005845.2", destination_dir=str(destination)),
        NCBIReferenceBundleConfig(),
    )
    assert output.destination_dir == str(destination)
    assert len(output.files) == 2
    assert all(destination in Path(path).parents and Path(path).is_file() for path in output.files)
    assert output.dataset_catalog == {"assemblies": []}


def test_reference_bundle_rejects_oversized_response_before_writing(monkeypatch, tmp_path):
    payload = _archive()
    monkeypatch.setattr(
        "proto_tools.tools.database_retrieval.ncbi.reference_bundle.build_http_session",
        lambda **kwargs: _Session(_Response(payload)),
    )
    destination = tmp_path / "reference"
    with pytest.raises(ValueError, match="max_download_bytes"):
        run_ncbi_reference_bundle(
            NCBIReferenceBundleInput(accession="GCF_000005845.2", destination_dir=str(destination)),
            NCBIReferenceBundleConfig(max_download_bytes=1),
        )
    assert not destination.exists()
