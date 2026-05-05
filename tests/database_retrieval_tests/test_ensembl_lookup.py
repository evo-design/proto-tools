"""tests/database_retrieval_tests/test_ensembl_lookup.py.

Tests for the Ensembl REST lookup wrapper (id-or-symbol → EnsemblGene).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblGene,
    EnsemblLookupConfig,
    EnsemblLookupInput,
    run_ensembl_lookup,
)

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"ensembl_id": "   ", "symbol": "  "}, {"ensembl_id": "ENSG00000012048", "symbol": "BRCA1"}],
    ids=["both-empty", "both-whitespace", "both-set"],
)
def test_validator_requires_exactly_one_of_id_or_symbol(kwargs):
    """The wrapper dispatches to exactly one upstream lookup route per call."""
    with pytest.raises(ValidationError, match="Provide exactly one of ensembl_id or symbol"):
        EnsemblLookupInput(**kwargs)


# ---------------------------------------------------------------------------
# Mocked dispatch — URL + params
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


def _stub_session(json_payload):
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/lookup/id/ENSG00000012048?expand=1"
    response.raise_for_status.return_value = None
    response.json.return_value = json_payload
    session.get.return_value = response
    return session


@pytest.mark.parametrize(
    "inputs_kwargs, config_overrides, expected_url_suffix, expected_params",
    [
        ({"ensembl_id": "ENSG00000012048"}, {}, "/lookup/id/ENSG00000012048", {"object_type": "gene"}),
        (
            {"ensembl_id": "ENSG00000012048"},
            {"expand": True},
            "/lookup/id/ENSG00000012048",
            {"object_type": "gene", "expand": "1"},
        ),
        ({"symbol": "BRCA1"}, {}, "/lookup/symbol/homo_sapiens/BRCA1", {}),
        ({"symbol": "BRCA1"}, {"expand": True}, "/lookup/symbol/homo_sapiens/BRCA1", {"expand": "1"}),
    ],
    ids=["lookup-by-id-default", "lookup-by-id-expand-on", "lookup-by-symbol-default", "lookup-by-symbol-expand-on"],
)
def test_url_and_params(inputs_kwargs, config_overrides, expected_url_suffix, expected_params):
    """Dispatch builds the right URL + query params for id vs symbol form."""
    session = _stub_session(_BRCA1_LOOKUP_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_lookup.build_session",
        return_value=session,
    ):
        run_ensembl_lookup(EnsemblLookupInput(**inputs_kwargs), EnsemblLookupConfig(**config_overrides))
    args, kwargs = session.get.call_args
    assert args[0].endswith(expected_url_suffix)
    assert kwargs["params"] == expected_params
    assert kwargs["headers"]["Accept"] == "application/json"


def test_dispatches_and_parses_into_ensembl_gene():
    """Real-shape BRCA1 payload round-trips into EnsemblGene with nested submodels."""
    session = _stub_session(_BRCA1_LOOKUP_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_lookup.build_session",
        return_value=session,
    ):
        out = run_ensembl_lookup(EnsemblLookupInput(ensembl_id="ENSG00000012048"))
    assert out.success
    assert isinstance(out.result, EnsemblGene)
    assert out.result.display_name == "BRCA1"
    assert out.result.canonical_transcript == "ENST00000357654.9"
    assert len(out.result.Transcript) == 1
    transcript = out.result.Transcript[0]
    assert transcript.is_canonical is True
    assert len(transcript.Exon) == 1
    assert transcript.Translation is not None
    assert transcript.Translation.length == 1863


def test_wraps_corrupt_json_with_context():
    """A 200 with non-JSON body surfaces a tight error mentioning the URL."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/lookup/id/ENSG00000012048"
    response.text = "<html>maintenance</html>"
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_lookup.build_session",
        return_value=session,
    ):
        out = run_ensembl_lookup(EnsemblLookupInput(ensembl_id="ENSG00000012048"))
    assert out.success is False
    assert any("non-JSON" in err and "ENSG00000012048" in err for err in out.errors)


def test_rejects_non_dict_payload():
    """A list where a dict is expected is a server-shape regression — surface it."""
    session = _stub_session([{"oops": "list"}])
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_lookup.build_session",
        return_value=session,
    ):
        out = run_ensembl_lookup(EnsemblLookupInput(ensembl_id="ENSG00000012048"))
    assert out.success is False
    assert any("non-dict" in err for err in out.errors)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_lookup_symbol_brca1_live():
    """BRCA1 lookup_symbol resolves to ENSG00000012048 with the canonical transcript present."""
    out = run_ensembl_lookup(EnsemblLookupInput(symbol="BRCA1"), EnsemblLookupConfig(expand=True))
    assert out.success, out.errors
    assert out.result.id == "ENSG00000012048"
    assert out.result.display_name == "BRCA1"
    assert out.result.canonical_transcript is not None
    assert out.result.canonical_transcript.startswith("ENST00000357654")
    # Live BRCA1 returns 47 transcripts; floor at 20 catches "expand=False regression"
    # / "only canonical returned" without being drift-fragile.
    assert len(out.result.Transcript) >= 20


@pytest.mark.integration
def test_lookup_id_grch37_routes_to_alt_host():
    """assembly='GRCh37' returns coordinates from the GRCh37 mirror (different from GRCh38)."""
    out_grch38 = run_ensembl_lookup(
        EnsemblLookupInput(ensembl_id="ENSG00000012048"),
        EnsemblLookupConfig(assembly="GRCh38"),
    )
    out_grch37 = run_ensembl_lookup(
        EnsemblLookupInput(ensembl_id="ENSG00000012048"),
        EnsemblLookupConfig(assembly="GRCh37"),
    )
    assert out_grch38.success and out_grch37.success
    assert out_grch38.result.assembly_name == "GRCh38"
    assert out_grch37.result.assembly_name == "GRCh37"
    assert out_grch38.result.start != out_grch37.result.start


@pytest.mark.integration
def test_unknown_id_surfaces_failure():
    """An unknown Ensembl ID propagates the upstream HTTPError as output.success=False."""
    out = run_ensembl_lookup(EnsemblLookupInput(ensembl_id="ENSGBOGUS00000000"))
    assert out.success is False
