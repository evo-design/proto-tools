"""tests/database_retrieval_tests/test_interproscan_fetch.py.

Tests for the InterPro fetch tool — direct UniProt-accession lookup and
sequence submit-and-poll paths against EBI's iprscan5 endpoint.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    InterProScanFetchConfig,
    InterProScanFetchInput,
    UniProtFetchInput,
    run_interproscan_fetch,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.interproscan.interproscan_fetch import (
    _DIRECT_LOOKUP_MAX_PAGES,
    _direct_lookup,
    _extract_xref_ids,
    _parse_direct_entry,
    _parse_iprscan_payload,
    _submit_iprscan,
)

# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


def test_validator_rejects_empty_input():
    """No uniprot_id and no sequence is rejected at parse time."""
    with pytest.raises(ValidationError, match="Provide either uniprot_id or sequence"):
        InterProScanFetchInput()


def test_validator_rejects_both_modes():
    """Providing both uniprot_id and sequence is rejected at parse time."""
    with pytest.raises(ValidationError, match="exactly one"):
        InterProScanFetchInput(uniprot_id="P04637", sequence="MKT")


def test_validator_rejects_whitespace_only_input():
    """Whitespace-only `uniprot_id` (or `sequence`) is treated as empty by the validator."""
    with pytest.raises(ValidationError, match="Provide either uniprot_id or sequence"):
        InterProScanFetchInput(uniprot_id="   ")


def test_run_surfaces_missing_email_for_sequence_path():
    """Sequence path with no config.email surfaces as output.success=False (decorator wraps the ValueError)."""
    output = run_interproscan_fetch(
        InterProScanFetchInput(sequence="MKTILV"),
        InterProScanFetchConfig(),
    )
    assert output.success is False
    assert any("email" in err.lower() for err in output.errors)


# ---------------------------------------------------------------------------
# Direct-lookup parser tests
# ---------------------------------------------------------------------------


_TP53_DBD_ENTRY = {
    "metadata": {
        "accession": "PF00870",
        "name": "P53 DNA-binding domain",
        "source_database": "pfam",
        "type": "domain",
        "integrated": "IPR011615",
        "go_terms": [{"identifier": "GO:0003677"}, {"identifier": "GO:0006355"}],
    },
    "proteins": [
        {
            "accession": "p04637",
            "protein_length": 393,
            "entry_protein_locations": [
                {
                    "fragments": [{"start": 100, "end": 288, "dc-status": "CONTINUOUS"}],
                    "score": 1.1e-59,
                    "model": "PF00870",
                    "representative": True,
                }
            ],
        }
    ],
}


def test_parse_direct_entry_parses_tp53_dbd():
    """The TP53 P53 DNA-binding-domain Pfam hit anchors the direct-lookup parser."""
    rows, sequence_length = _parse_direct_entry(_TP53_DBD_ENTRY, include_go_terms=True, include_pathways=True)
    assert sequence_length == 393
    assert len(rows) == 1
    row = rows[0]
    assert row.accession == "PF00870"
    assert row.name == "P53 DNA-binding domain"
    assert row.member_database == "pfam"
    assert row.integrated_ipr == "IPR011615"
    assert row.start == 100
    assert row.end == 288
    assert row.type == "domain"
    assert row.score == pytest.approx(1.1e-59)
    assert row.representative is True
    assert "GO:0003677" in row.go_terms


def test_parse_direct_entry_multi_fragment():
    """A location with multiple discontinuous fragments produces one row per fragment."""
    entry = {
        "metadata": {
            "accession": "PF00001",
            "name": "Ex Domain",
            "source_database": "pfam",
            "type": "family",
        },
        "proteins": [
            {
                "accession": "p99999",
                "protein_length": 200,
                "entry_protein_locations": [
                    {
                        "fragments": [
                            {"start": 10, "end": 50},
                            {"start": 80, "end": 120},
                        ],
                        "score": 1e-20,
                        "model": "PF00001",
                    }
                ],
            }
        ],
    }
    rows, _ = _parse_direct_entry(entry, include_go_terms=False, include_pathways=False)
    assert [(r.start, r.end) for r in rows] == [(10, 50), (80, 120)]
    assert all(r.go_terms == [] for r in rows)


def test_parse_direct_entry_unknown_type_falls_back():
    """An unrecognized 'type' value maps to 'unknown' rather than crashing."""
    entry = {
        "metadata": {
            "accession": "PF99999",
            "name": "Mystery",
            "source_database": "pfam",
            "type": "freshly_invented_category",
        },
        "proteins": [
            {
                "accession": "p99999",
                "protein_length": 100,
                "entry_protein_locations": [{"fragments": [{"start": 1, "end": 50}]}],
            }
        ],
    }
    rows, _ = _parse_direct_entry(entry, include_go_terms=False, include_pathways=False)
    assert rows[0].type == "unknown"


# ---------------------------------------------------------------------------
# Direct-lookup pagination test
# ---------------------------------------------------------------------------


def _mock_paginated_session(pages):
    session = MagicMock()
    responses = []
    for page in pages:
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = page
        responses.append(response)
    session.get.side_effect = responses
    return session


def test_direct_lookup_walks_next_cursor():
    """The paginated direct path follows `next` until None and accumulates results."""
    pages = [
        {
            "results": [_TP53_DBD_ENTRY],
            "next": "https://example/page2",
        },
        {
            "results": [
                {
                    "metadata": {
                        "accession": "PF00097",
                        "name": "Zinc finger",
                        "source_database": "pfam",
                        "type": "domain",
                    },
                    "proteins": [
                        {
                            "accession": "p04637",
                            "protein_length": 393,
                            "entry_protein_locations": [{"fragments": [{"start": 320, "end": 356}]}],
                        }
                    ],
                }
            ],
            "next": None,
        },
    ]
    session = _mock_paginated_session(pages)
    config = InterProScanFetchConfig()
    output = _direct_lookup("P04637", config, session)
    assert output.num_domains == 2
    assert output.sequence_length == 393
    assert {d.accession for d in output.domains} == {"PF00870", "PF00097"}
    assert output.job_id == ""
    assert "P04637" in output.source_url
    assert session.get.call_count == 2


@pytest.mark.parametrize(
    "status_code, body",
    [
        (404, ""),
        (204, ""),
    ],
    ids=["404", "204-empty"],
)
def test_direct_lookup_raises_on_unknown_accession(status_code, body):
    """InterPro returns 204 No Content (observed live) or 404 for unknown accessions; both surface as 'no entries'."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    session.get.return_value = response
    with pytest.raises(ValueError, match="no entries"):
        _direct_lookup("Q99999999", InterProScanFetchConfig(), session)


def test_direct_lookup_caps_pagination_depth():
    """A corrupted upstream `next` cursor that never terminates is bounded by _DIRECT_LOOKUP_MAX_PAGES."""
    pages = [{"results": [], "next": f"https://example/page{i}"} for i in range(_DIRECT_LOOKUP_MAX_PAGES + 5)]
    session = _mock_paginated_session(pages)
    with pytest.raises(ValueError, match="pagination exceeded"):
        _direct_lookup("P04637", InterProScanFetchConfig(), session)


def test_direct_lookup_detects_pagination_loop():
    """If `next` echoes a previously-seen URL, raise rather than poll forever."""
    page = {"results": [], "next": "https://example/page1"}  # `next` points back at itself
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.text = '{"results": [], "next": "https://example/page1"}'
    response.json.return_value = page
    session.get.return_value = response
    with pytest.raises(ValueError, match="revisited URL"):
        _direct_lookup("P04637", InterProScanFetchConfig(), session)


def test_direct_lookup_wraps_corrupt_json_with_context():
    """A 200 response with invalid JSON surfaces a tight error mentioning the accession + body excerpt."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.text = "<html>maintenance page</html>"
    response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    session.get.return_value = response
    with pytest.raises(ValueError, match=r"non-JSON for 'P04637'.*maintenance"):
        _direct_lookup("P04637", InterProScanFetchConfig(), session)


# ---------------------------------------------------------------------------
# iprscan5 submit + parse tests
# ---------------------------------------------------------------------------


def test_submit_iprscan_returns_plain_text_job_id():
    """iprscan5 returns the job ID in the response body as plain text (not JSON)."""
    session = MagicMock()
    response = MagicMock()
    response.text = "iprscan5-R20260504-123456-7890-12345-p1m"
    response.raise_for_status.return_value = None
    session.post.return_value = response
    job_id = _submit_iprscan(
        "MKTILVAA",
        "dev@example.org",
        InterProScanFetchConfig(email="dev@example.org"),
        session,
    )
    assert job_id == "iprscan5-R20260504-123456-7890-12345-p1m"
    args, kwargs = session.post.call_args
    assert args[0].endswith("/iprscan5/run/")
    posted_data = dict(kwargs["data"]) if isinstance(kwargs["data"], list) else kwargs["data"]
    assert posted_data["email"] == "dev@example.org"
    assert posted_data["sequence"] == "MKTILVAA"


def test_submit_iprscan_rejects_empty_body():
    """An empty response body from iprscan5 means the submission failed."""
    session = MagicMock()
    response = MagicMock()
    response.text = "   "
    response.raise_for_status.return_value = None
    session.post.return_value = response
    with pytest.raises(ValueError, match="empty job ID"):
        _submit_iprscan("MKT", "dev@example.org", InterProScanFetchConfig(email="dev@example.org"), session)


_IPRSCAN5_RESULT_PAYLOAD = {
    "results": [
        {
            "sequence": "MKTILVAA",
            "sequenceLength": 8,
            "xref": [{"id": "P04637", "name": "TP53"}],
            "matches": [
                {
                    "model-ac": "PF00870",
                    "signature": {
                        "accession": "PF00870",
                        "name": "P53",
                        "description": "P53 DNA-binding domain",
                        "signatureLibraryRelease": {"library": "pfam", "version": "37.0"},
                        "entry": {
                            "accession": "IPR011615",
                            "type": "domain",
                            "goXRefs": [{"id": "GO:0003677"}],
                            "pathwayXRefs": [{"id": "R-HSA-9663199"}],
                        },
                    },
                    "locations": [{"start": 1, "end": 8, "evalue": 1.5e-12}],
                }
            ],
        }
    ]
}


def test_parse_iprscan_payload_flattens_match_locations():
    """The iprscan5 result parser flattens match.locations[] into InterProDomain rows."""
    output = _parse_iprscan_payload(
        _IPRSCAN5_RESULT_PAYLOAD,
        job_id="iprscan5-test",
        result_url="https://example/iprscan5/result/iprscan5-test/json",
        config=InterProScanFetchConfig(email="dev@example.org"),
    )
    assert output.accession == "P04637"
    assert output.sequence_length == 8
    assert output.num_domains == 1
    row = output.domains[0]
    assert row.accession == "PF00870"
    assert row.member_database == "pfam"
    assert row.integrated_ipr == "IPR011615"
    assert row.start == 1
    assert row.end == 8
    assert row.score == pytest.approx(1.5e-12)
    assert row.go_terms == ["GO:0003677"]
    assert row.pathways == ["R-HSA-9663199"]


def test_parse_iprscan_payload_raises_on_empty_results():
    """An iprscan5 payload with no results indicates a server-side regression; surface it."""
    with pytest.raises(ValueError, match="no results"):
        _parse_iprscan_payload(
            {"results": []},
            job_id="iprscan5-test",
            result_url="https://example/url",
            config=InterProScanFetchConfig(email="dev@example.org"),
        )


# ---------------------------------------------------------------------------
# Submit-and-poll end-to-end dispatch (driven through real session mocks)
# ---------------------------------------------------------------------------


def test_submit_path_dispatches_through_full_iprscan_cycle(monkeypatch):
    """End-to-end: POST submit → poll RUNNING/FINISHED → fetch JSON → parsed Output.

    Drives the whole sequence path through a single mocked session, asserting
    the user-visible Output rather than internal call args. This replaces an
    earlier version that was overfit to the literal frozenset args passed to
    poll_until_complete.
    """
    monkeypatch.setattr("proto_tools.utils.polling.time.sleep", lambda _s: None)
    session = MagicMock()

    submit_response = MagicMock()
    submit_response.text = "iprscan5-XYZ"
    submit_response.raise_for_status.return_value = None
    session.post.return_value = submit_response

    status_running = MagicMock()
    status_running.status_code = 200
    status_running.text = "RUNNING"
    status_running.raise_for_status.return_value = None
    status_finished = MagicMock()
    status_finished.status_code = 200
    status_finished.text = "FINISHED"
    status_finished.raise_for_status.return_value = None
    result_response = MagicMock()
    result_response.status_code = 200
    result_response.raise_for_status.return_value = None
    result_response.json.return_value = _IPRSCAN5_RESULT_PAYLOAD
    session.get.side_effect = [status_running, status_finished, result_response]

    with patch(
        "proto_tools.tools.database_retrieval.interproscan.interproscan_fetch.build_http_session",
        return_value=session,
    ):
        output = run_interproscan_fetch(
            InterProScanFetchInput(sequence="MKTILVAA"),
            InterProScanFetchConfig(email="dev@example.org"),
        )

    assert output.success
    assert output.job_id == "iprscan5-XYZ"
    assert output.accession == "P04637"
    assert output.num_domains == 1
    assert output.domains[0].accession == "PF00870"
    # POST hit /run/, GETs hit /status/ then /result/.
    assert session.post.call_args.args[0].endswith("/iprscan5/run/")
    polled_urls = [call.args[0] for call in session.get.call_args_list]
    assert polled_urls[0].endswith("/status/iprscan5-XYZ")
    assert polled_urls[1].endswith("/status/iprscan5-XYZ")
    assert polled_urls[2].endswith("/result/iprscan5-XYZ/json")


# ---------------------------------------------------------------------------
# Cross-reference extractor (single helper used by both paths)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([{"id": "GO:1"}, {"id": "GO:2"}], ["GO:1", "GO:2"]),
        ([{"identifier": "GO:1"}], ["GO:1"]),
        (["GO:1", "GO:2"], ["GO:1", "GO:2"]),
        ([{"id": ""}, {"id": "GO:1"}, {"id": "   "}], ["GO:1"]),
        ([{"name": "no id key"}], []),
        ([], []),
        (None, []),
        ("not a list", []),
    ],
    ids=["dict-id", "dict-identifier", "plain-strings", "drops-empty", "no-id-key", "empty-list", "none", "non-list"],
)
def test_extract_xref_ids_handles_diverse_shapes(raw, expected):
    """One helper covers both paths' xref shapes; missing/empty/malformed values are dropped silently."""
    assert _extract_xref_ids(raw) == expected


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_direct_lookup_unknown_accession_surfaces_failure():
    """An unknown UniProt accession yields output.success=False with a descriptive error."""
    output = run_interproscan_fetch(InterProScanFetchInput(uniprot_id="Q99999999"))
    assert output.success is False
    assert any("no entries" in err.lower() or "not found" in err.lower() for err in output.errors)


@pytest.mark.integration
def test_workflow_with_uniprot_fetch():
    """Cross-tool chain: uniprot-fetch length matches interproscan-fetch sequence_length, all domain coords in bounds."""
    uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id="P04637"))
    ipr = run_interproscan_fetch(InterProScanFetchInput(uniprot_id="P04637"))
    assert uniprot.success and ipr.success
    assert ipr.sequence_length == uniprot.length
    assert all(1 <= d.start <= d.end <= uniprot.length for d in ipr.domains)


@pytest.mark.integration
@pytest.mark.parametrize(
    "uniprot_id, expected_length, expected_pfam_accession, expected_anchor",
    [
        # Each row anchors a stable Pfam hit by exact (start, end) coords. Pfam family
        # boundaries are decade-stable; if any of these drift, InterPro/Pfam moved and
        # we want to know.
        ("P04637", 393, "PF00870", (100, 288)),  # TP53 — Pfam P53 DBD
        ("P38398", 1863, "PF00533", (1645, 1723)),  # BRCA1 — first Pfam BRCT
        ("P01116", 189, "PF00071", (5, 164)),  # KRAS — Pfam Ras family
        ("P00533", 1210, "PF07714", (714, 966)),  # EGFR — Pfam tyrosine kinase
    ],
    ids=["TP53", "BRCA1", "KRAS", "EGFR"],
)
def test_direct_lookup_known_oncogenes_workload(uniprot_id, expected_length, expected_pfam_accession, expected_anchor):
    """Realistic workload: 4 cancer-driver proteins each return their canonical Pfam anchor at exact coords."""
    output = run_interproscan_fetch(InterProScanFetchInput(uniprot_id=uniprot_id))
    assert output.success, output.errors
    assert output.accession == uniprot_id
    assert output.sequence_length == expected_length
    start, end = expected_anchor
    assert any(
        d.member_database == "pfam" and d.accession == expected_pfam_accession and d.start == start and d.end == end
        for d in output.domains
    ), (
        f"missing canonical {expected_pfam_accession} at {start}-{end}; "
        f"got Pfam: {sorted((d.accession, d.start, d.end) for d in output.domains if d.member_database == 'pfam')[:5]}"
    )
    assert all(1 <= d.start <= d.end <= expected_length for d in output.domains)


@pytest.mark.integration
@pytest.mark.slow
def test_submit_path_real_short_sequence():
    """End-to-end iprscan5 submit path against the first 200 residues of TP53.

    Wall-clock 2-30 min depending on EBI queue depth. Validates that
    _submit_iprscan + poll_until_complete + _parse_iprscan_payload all
    work against the live service, which the mocked dispatch test cannot.
    """
    # First 200 residues of P04637 — covers the P53 DNA-binding domain (residues 100-288 truncated to 100-200).
    sequence = (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQK"
        "TYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGN"
    )
    assert len(sequence) == 200
    output = run_interproscan_fetch(
        InterProScanFetchInput(sequence=sequence),
        InterProScanFetchConfig(email="noreply@example.org"),
    )
    assert output.success, output.errors
    assert output.job_id.startswith("iprscan5-")
    assert output.num_domains > 0
    # P53 DBD start ≥ 100 in TP53; submission was the first 200 residues, so any
    # Pfam P53 hit must start at or after residue 100 within the submitted segment.
    assert any(d.member_database == "pfam" and "P53" in d.name for d in output.domains)
