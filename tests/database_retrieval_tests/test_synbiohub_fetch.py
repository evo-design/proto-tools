"""Tests for standards-native SynBioHub retrieval."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    SynBioHubFetchConfig,
    SynBioHubFetchInput,
    run_synbiohub_fetch,
)
from proto_tools.tools.database_retrieval.synbiohub import synbiohub_fetch


def test_object_uri_rejects_query_parameters():
    with pytest.raises(ValidationError, match="query parameters"):
        SynBioHubFetchInput(object_uri="https://synbiohub.org/public/igem/BBa_E0040/1?token=secret")


def test_fetch_rejects_cross_origin_object(monkeypatch):
    with pytest.raises(ValueError, match="same scheme and host"):
        run_synbiohub_fetch(
            SynBioHubFetchInput(object_uri="https://example.org/public/igem/BBa_E0040/1"),
            SynBioHubFetchConfig(registry_url="https://synbiohub.org"),
        )


def test_fetches_exact_documents_without_exposing_token(monkeypatch):
    sbol = MagicMock(content=b"<rdf:RDF>\n</rdf:RDF>\n", encoding="utf-8", status_code=200)
    fasta = MagicMock(content=b">BBa_E0040\nATGC\n", encoding="utf-8", status_code=200)
    sbol.raise_for_status.return_value = None
    fasta.raise_for_status.return_value = None
    session = MagicMock()
    session.get.side_effect = [sbol, fasta]
    monkeypatch.setattr(synbiohub_fetch, "build_http_session", lambda **kwargs: session)

    output = run_synbiohub_fetch(
        SynBioHubFetchInput(object_uri="https://api.synbiohub.org/public/bsu/BO_9757/1"),
        SynBioHubFetchConfig(synbiohub_token="private-token"),
    )

    assert [document.format for document in output.documents] == ["sbol", "fasta"]
    assert output.documents[0].content.endswith("\n")
    for document in output.documents:
        assert document.artifact.matches_content(document.content.encode("utf-8"))
    assert "private-token" not in output.model_dump_json()
    assert session.get.call_args_list[0].kwargs["headers"]["X-authorization"] == "private-token"


def test_nonrecursive_sbol_uses_sbolnr(monkeypatch):
    response = MagicMock(content=b"<rdf:RDF/>\n", encoding="utf-8", status_code=200)
    response.raise_for_status.return_value = None
    session = MagicMock()
    session.get.return_value = response
    monkeypatch.setattr(synbiohub_fetch, "build_http_session", lambda **kwargs: session)

    run_synbiohub_fetch(
        SynBioHubFetchInput(object_uri="https://api.synbiohub.org/public/bsu/BO_9757/1"),
        SynBioHubFetchConfig(formats=["sbol"], recursive_sbol=False),
    )

    assert session.get.call_args.args[0].endswith("/sbolnr")
