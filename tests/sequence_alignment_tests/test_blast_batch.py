"""Tests for complete online BLAST FASTA submission and query coverage."""

from types import SimpleNamespace

import pytest
from Bio.Blast import NCBIXML

from proto_tools.tools.sequence_alignment.blast import _online_client, blast_search


@pytest.mark.parametrize("returned", [["b", "a"], ["a"], ["a", "a"], ["a", "other"], []])
def test_batch_query_coverage_including_no_hits(monkeypatch, tmp_path, returned):
    path = tmp_path / "queries.fasta"
    path.write_text(">a description\nACGT\n>b\nTGCA\n")
    requests = []

    def submit(params, **_kwargs):
        requests.append(params)
        return "mock XML"

    monkeypatch.setattr(_online_client, "search_xml", submit)
    monkeypatch.setattr(
        NCBIXML, "parse", lambda _: iter([SimpleNamespace(query=query, alignments=[]) for query in returned])
    )
    inputs = blast_search.BlastSearchInput(query=str(path))
    config = blast_search.BlastSearchConfig()
    if returned == ["b", "a"]:
        output = blast_search._online_search(inputs, config)
        assert output.hits == []
        assert output.metadata["query_ids"] == returned
        assert output.metadata["query_length"] == 8
    else:
        with pytest.raises(ValueError, match="identifiers do not match"):
            blast_search._online_search(inputs, config)
    assert requests[0]["QUERY"] == ">a\nACGT\n>b\nTGCA\n"


@pytest.mark.parametrize("fasta", ["", ">a\n", ">a\nACGT\n>a\nTGCA\n"])
def test_invalid_batch_rejected_before_submission(monkeypatch, tmp_path, fasta):
    path = tmp_path / "queries.fasta"
    path.write_text(fasta)
    calls = []
    monkeypatch.setattr(_online_client, "search_xml", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="requires"):
        blast_search._online_search(blast_search.BlastSearchInput(query=str(path)), blast_search.BlastSearchConfig())
    assert calls == []
