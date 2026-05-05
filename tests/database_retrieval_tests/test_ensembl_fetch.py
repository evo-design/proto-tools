"""tests/database_retrieval_tests/test_ensembl_fetch.py.

Tests for the Ensembl REST fetch wrapper (lookup/sequence/overlap/xrefs).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblFetchConfig,
    EnsemblFetchInput,
    EnsemblGene,
    UniProtFetchInput,
    run_ensembl_fetch,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.ensembl.ensembl_fetch import (
    _build_url_and_params,
    _parse_payload,
)

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_rejects_empty_input():
    """Both ensembl_id and symbol blank → rejected at parse time."""
    with pytest.raises(ValidationError, match="Provide either ensembl_id or symbol"):
        EnsemblFetchInput()


def test_validator_rejects_whitespace_only():
    """Whitespace-only is treated as empty by the validator."""
    with pytest.raises(ValidationError, match="Provide either ensembl_id or symbol"):
        EnsemblFetchInput(ensembl_id="   ", symbol="  ")


# ---------------------------------------------------------------------------
# URL builder (per-endpoint coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint, inputs_kwargs, config_overrides, expected_url_suffix, expected_params",
    [
        (
            "lookup_id",
            {"ensembl_id": "ENSG00000012048"},
            {},
            "/lookup/id/ENSG00000012048",
            {"expand": "1"},
        ),
        (
            "lookup_id",
            {"ensembl_id": "ENSG00000012048"},
            {"expand": False},
            "/lookup/id/ENSG00000012048",
            {},
        ),
        (
            "lookup_symbol",
            {"symbol": "BRCA1"},
            {},
            "/lookup/symbol/homo_sapiens/BRCA1",
            {"expand": "1"},
        ),
        (
            "sequence",
            {"ensembl_id": "ENST00000357654"},
            {"sequence_type": "protein"},
            "/sequence/id/ENST00000357654",
            {"type": "protein"},
        ),
        (
            "overlap",
            {"ensembl_id": "ENSG00000012048"},
            {"overlap_feature": "regulatory"},
            "/overlap/id/ENSG00000012048",
            {"feature": "regulatory"},
        ),
        (
            "xrefs",
            {"ensembl_id": "ENSG00000012048"},
            {},
            "/xrefs/id/ENSG00000012048",
            {},
        ),
        (
            "xrefs",
            {"symbol": "BRCA1"},
            {},
            "/xrefs/symbol/homo_sapiens/BRCA1",
            {},
        ),
    ],
    ids=[
        "lookup_id-expand-on",
        "lookup_id-expand-off",
        "lookup_symbol",
        "sequence-protein",
        "overlap-regulatory",
        "xrefs-by-id",
        "xrefs-by-symbol",
    ],
)
def test_build_url_and_params(endpoint, inputs_kwargs, config_overrides, expected_url_suffix, expected_params):
    """Each endpoint produces the right URL + query params."""
    inputs = EnsemblFetchInput(**inputs_kwargs)
    config = EnsemblFetchConfig(endpoint=endpoint, **config_overrides)
    url, params = _build_url_and_params(inputs, config, base="https://rest.ensembl.org")
    assert url.endswith(expected_url_suffix)
    assert params == expected_params


@pytest.mark.parametrize(
    "endpoint, inputs_kwargs, error_match",
    [
        ("lookup_id", {"symbol": "BRCA1"}, "lookup_id.*requires.*ensembl_id"),
        ("lookup_symbol", {"ensembl_id": "ENSG00000012048"}, "lookup_symbol.*requires.*symbol"),
        ("sequence", {"symbol": "BRCA1"}, "sequence.*requires.*ensembl_id"),
        ("overlap", {"symbol": "BRCA1"}, "overlap.*requires.*ensembl_id"),
    ],
    ids=["lookup_id-no-id", "lookup_symbol-no-symbol", "sequence-no-id", "overlap-no-id"],
)
def test_build_url_rejects_missing_input_for_endpoint(endpoint, inputs_kwargs, error_match):
    """Each endpoint that requires a specific input field rejects the wrong one."""
    inputs = EnsemblFetchInput(**inputs_kwargs)
    config = EnsemblFetchConfig(endpoint=endpoint)
    with pytest.raises(ValueError, match=error_match):
        _build_url_and_params(inputs, config, base="https://rest.ensembl.org")


# ---------------------------------------------------------------------------
# Payload parser
# ---------------------------------------------------------------------------


_BRCA1_LOOKUP_PAYLOAD = {
    "id": "ENSG00000012048",
    "display_name": "BRCA1",
    "description": "BRCA1 DNA repair associated",
    "biotype": "protein_coding",
    "species": "homo_sapiens",
    "seq_region_name": "17",
    "start": 43044292,
    "end": 43170245,
    "strand": -1,
    "assembly_name": "GRCh38",
    "canonical_transcript": "ENST00000357654.9",
    "Transcript": [
        {
            "id": "ENST00000357654",
            "display_name": "BRCA1-201",
            "biotype": "protein_coding",
            "is_canonical": True,
            "start": 43044295,
            "end": 43125483,
            "strand": -1,
            "seq_region_name": "17",
            "assembly_name": "GRCh38",
            "Exon": [
                {
                    "id": "ENSE00001871077",
                    "seq_region_name": "17",
                    "start": 43044295,
                    "end": 43045802,
                    "strand": -1,
                    "assembly_name": "GRCh38",
                    "version": 1,
                }
            ],
            "Translation": {
                "id": "ENSP00000350283",
                "start": 43045802,
                "end": 43124096,
                "length": 1863,
                "Parent": "ENST00000357654",
            },
        }
    ],
}


def test_parse_lookup_payload_into_ensembl_gene():
    """A real-shape lookup_id payload round-trips into EnsemblGene with nested submodels."""
    result = _parse_payload(_BRCA1_LOOKUP_PAYLOAD, "lookup_id")
    assert isinstance(result, EnsemblGene)
    assert result.display_name == "BRCA1"
    assert result.canonical_transcript == "ENST00000357654.9"
    assert len(result.Transcript) == 1
    transcript = result.Transcript[0]
    assert transcript.is_canonical is True
    assert len(transcript.Exon) == 1
    assert transcript.Translation is not None
    assert transcript.Translation.length == 1863


def test_parse_sequence_payload():
    """Sequence endpoint returns {id, desc, mol_type, seq}."""
    result = _parse_payload(
        {"id": "ENST00000357654", "desc": "...", "mol_type": "protein", "seq": "MDLSAL"}, "sequence"
    )
    assert result.id == "ENST00000357654"  # type: ignore[union-attr]
    assert result.seq == "MDLSAL"  # type: ignore[union-attr]


def test_parse_xrefs_payload():
    """Xrefs returns a list; we keep Uniprot_gn rows accessible by dbname."""
    payload = [
        {"dbname": "Uniprot_gn", "display_id": "BRCA1", "primary_id": "P38398", "info_type": "DIRECT"},
        {"dbname": "EntrezGene", "display_id": "BRCA1", "primary_id": "672", "info_type": "DEPENDENT"},
    ]
    result = _parse_payload(payload, "xrefs")
    assert isinstance(result, list)
    assert len(result) == 2
    uniprot = next(x for x in result if x.dbname == "Uniprot_gn")
    assert uniprot.primary_id == "P38398"


def test_parse_overlap_payload_keeps_raw_for_feature_specific_keys():
    """Overlap features differ in shape; parser keeps the raw dict alongside the typed common fields."""
    payload = [
        {
            "feature_type": "gene",
            "id": "ENSG00000012048",
            "biotype": "protein_coding",
            "start": 43044292,
            "end": 43170245,
            "strand": -1,
            "seq_region_name": "17",
            "external_name": "BRCA1",
        },
        {
            "feature_type": "regulatory",
            "start": 43044300,
            "end": 43044400,
            "strand": 0,
            "seq_region_name": "17",
            "feature_name": "Open chromatin",
        },
    ]
    result = _parse_payload(payload, "overlap")
    assert len(result) == 2
    gene_record, reg_record = result
    assert gene_record.feature_type == "gene"  # type: ignore[union-attr]
    assert gene_record.raw["external_name"] == "BRCA1"  # type: ignore[index, union-attr]
    assert reg_record.feature_type == "regulatory"  # type: ignore[union-attr]
    assert reg_record.id is None  # type: ignore[union-attr]


def test_parse_overlap_raw_override_wins_if_api_adds_raw_key():
    """If Ensembl ever returns a feature with a 'raw' key, our injected raw must override it."""
    payload = [
        {
            "feature_type": "gene",
            "start": 1,
            "end": 100,
            "strand": 1,
            "seq_region_name": "17",
            "raw": "SHOULD_BE_OVERRIDDEN",
        },
    ]
    result = _parse_payload(payload, "overlap")
    assert result[0].raw == payload[0]  # type: ignore[union-attr]


def test_parse_overlap_rejects_non_dict_element():
    """A non-dict element raises a clear ValueError mentioning the index, not a cryptic TypeError."""
    with pytest.raises(ValueError, match=r"overlap payload\[0\] is non-dict"):
        _parse_payload(["not a dict"], "overlap")


def test_parse_payload_rejects_wrong_shape():
    """Type-mismatch (list expected, dict received) raises a clear error."""
    with pytest.raises(ValueError, match="non-list"):
        _parse_payload({"oops": "dict"}, "xrefs")
    with pytest.raises(ValueError, match="non-dict"):
        _parse_payload([{"id": "x"}], "lookup_id")


# ---------------------------------------------------------------------------
# Run-time (mocked session)
# ---------------------------------------------------------------------------


def _stub_session(json_payload):
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/lookup/id/ENSG00000012048?expand=1"
    response.raise_for_status.return_value = None
    response.json.return_value = json_payload
    session.get.return_value = response
    return session


def test_run_ensembl_fetch_dispatches_lookup_id():
    """Full dispatch: build URL, GET, parse, return Output with discriminated result."""
    session = _stub_session(_BRCA1_LOOKUP_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_fetch.build_session",
        return_value=session,
    ):
        out = run_ensembl_fetch(
            EnsemblFetchInput(ensembl_id="ENSG00000012048"),
            EnsemblFetchConfig(endpoint="lookup_id"),
        )
    assert out.success
    assert out.endpoint == "lookup_id"
    assert isinstance(out.result, EnsemblGene)
    assert out.result.display_name == "BRCA1"
    # Confirms the GET was made with the Accept header + expand param.
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Accept"] == "application/json"
    assert kwargs["params"] == {"expand": "1"}


def test_run_ensembl_fetch_wraps_corrupt_json_with_context():
    """A 200 with non-JSON body surfaces a tight error mentioning both 'non-JSON' and the URL."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/lookup/id/ENSG00000012048"
    response.text = "<html>maintenance</html>"
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_fetch.build_session",
        return_value=session,
    ):
        out = run_ensembl_fetch(
            EnsemblFetchInput(ensembl_id="ENSG00000012048"),
            EnsemblFetchConfig(endpoint="lookup_id"),
        )
    assert out.success is False
    assert any("non-JSON" in err and "ENSG00000012048" in err for err in out.errors)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_lookup_symbol_brca1_live():
    """BRCA1 lookup_symbol resolves to ENSG00000012048 with the canonical transcript present."""
    out = run_ensembl_fetch(EnsemblFetchInput(symbol="BRCA1"), EnsemblFetchConfig(endpoint="lookup_symbol"))
    assert out.success, out.errors
    assert isinstance(out.result, EnsemblGene)
    assert out.result.id == "ENSG00000012048"
    assert out.result.display_name == "BRCA1"
    assert out.result.canonical_transcript is not None
    assert out.result.canonical_transcript.startswith("ENST00000357654")
    # Live BRCA1 returns 47 transcripts; floor at 20 catches "expand=False regression" /
    # "only canonical returned" without being drift-fragile.
    assert len(out.result.Transcript) >= 20


@pytest.mark.integration
def test_lookup_id_grch37_routes_to_alt_host():
    """assembly='GRCh37' returns coordinates from the GRCh37 mirror (different from GRCh38)."""
    out_grch38 = run_ensembl_fetch(
        EnsemblFetchInput(ensembl_id="ENSG00000012048"),
        EnsemblFetchConfig(endpoint="lookup_id", assembly="GRCh38"),
    )
    out_grch37 = run_ensembl_fetch(
        EnsemblFetchInput(ensembl_id="ENSG00000012048"),
        EnsemblFetchConfig(endpoint="lookup_id", assembly="GRCh37"),
    )
    assert out_grch38.success and out_grch37.success
    assert out_grch38.result.assembly_name == "GRCh38"  # type: ignore[union-attr]
    assert out_grch37.result.assembly_name == "GRCh37"  # type: ignore[union-attr]
    assert out_grch38.result.start != out_grch37.result.start  # type: ignore[union-attr]


@pytest.mark.integration
def test_xrefs_brca1_includes_uniprot_mapping():
    """Xrefs surfaces the Uniprot_gn cross-reference enabling chain-into-uniprot-fetch."""
    out = run_ensembl_fetch(
        EnsemblFetchInput(ensembl_id="ENSG00000012048"),
        EnsemblFetchConfig(endpoint="xrefs"),
    )
    assert out.success, out.errors
    assert any(x.dbname == "Uniprot_gn" for x in out.result)  # type: ignore[union-attr]


@pytest.mark.integration
def test_unknown_id_surfaces_failure():
    """An unknown Ensembl ID propagates the upstream HTTPError as output.success=False."""
    out = run_ensembl_fetch(
        EnsemblFetchInput(ensembl_id="ENSGBOGUS00000000"),
        EnsemblFetchConfig(endpoint="lookup_id"),
    )
    assert out.success is False


@pytest.mark.integration
def test_workflow_with_uniprot_fetch():
    """Cross-tool: ensembl xrefs → uniprot-fetch on the resolved accession returns the same gene name."""
    xrefs = run_ensembl_fetch(
        EnsemblFetchInput(ensembl_id="ENSG00000012048"),
        EnsemblFetchConfig(endpoint="xrefs"),
    )
    assert xrefs.success
    uniprot_id = next(x.primary_id for x in xrefs.result if x.dbname == "Uniprot_gn")  # type: ignore[union-attr]
    uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id=uniprot_id))
    assert uniprot.success
    assert "brca1" in {n.lower() for n in uniprot.gene_names}
