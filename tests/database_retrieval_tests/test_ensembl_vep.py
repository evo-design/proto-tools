"""tests/database_retrieval_tests/test_ensembl_vep.py.

Tests for the Ensembl VEP wrapper (HGVS → consequence predictions).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblVEPInput,
    run_ensembl_vep,
)

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_rejects_empty_hgvs():
    """Empty / whitespace-only HGVS rejected at parse time."""
    with pytest.raises(ValidationError, match="non-empty HGVS"):
        EnsemblVEPInput(hgvs="   ")


# ---------------------------------------------------------------------------
# Run-time (mocked session)
# ---------------------------------------------------------------------------


_VEP_PAYLOAD = [
    {
        "input": "9:g.22125504G>C",
        "id": "rs7044859",
        "most_severe_consequence": "intron_variant",
        "seq_region_name": "9",
        "start": 22125504,
        "end": 22125504,
        "strand": 1,
        "allele_string": "G/C",
        "transcript_consequences": [
            {"transcript_id": "ENST00000380152", "consequence_terms": ["intron_variant"], "impact": "MODIFIER"}
        ],
        "colocated_variants": [{"id": "rs7044859", "frequencies": {}}],
    }
]


def test_run_ensembl_vep_dispatches_and_parses():
    """End-to-end: build URL, GET, parse list of consequences, return Output."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/vep/homo_sapiens/hgvs/9%3Ag.22125504G%3EC"
    response.raise_for_status.return_value = None
    response.json.return_value = _VEP_PAYLOAD
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_vep.build_session",
        return_value=session,
    ):
        out = run_ensembl_vep(EnsemblVEPInput(hgvs="9:g.22125504G>C"))
    assert out.success
    assert out.num_consequences == 1
    cons = out.consequences[0]
    assert cons.most_severe_consequence == "intron_variant"
    assert cons.allele_string == "G/C"
    assert len(cons.transcript_consequences) == 1
    # HGVS encoded in URL — '>' becomes '%3E'.
    args, _ = session.get.call_args
    assert "9%3Ag.22125504G%3EC" in args[0]


def test_run_ensembl_vep_wraps_corrupt_json():
    """Non-JSON body surfaces a tight error mentioning the HGVS input."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/vep/homo_sapiens/hgvs/bad"
    response.text = "<html>err</html>"
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_vep.build_session",
        return_value=session,
    ):
        with pytest.raises(Exception, match="non-JSON"):
            run_ensembl_vep(EnsemblVEPInput(hgvs="9:g.22125504G>C"))


def test_run_ensembl_vep_rejects_non_list_payload():
    """A dict where a list is expected indicates a server-shape regression — surface it."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/vep/homo_sapiens/hgvs/9%3Ag.22125504G%3EC"
    response.raise_for_status.return_value = None
    response.json.return_value = {"oops": "dict"}
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_vep.build_session",
        return_value=session,
    ):
        with pytest.raises(Exception, match="non-list"):
            run_ensembl_vep(EnsemblVEPInput(hgvs="9:g.22125504G>C"))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_vep_hgvs_genomic_live():
    """Genomic HGVS form returns at least one consequence with most_severe_consequence populated."""
    out = run_ensembl_vep(EnsemblVEPInput(hgvs="9:g.22125504G>C"))
    assert out.success, out.errors
    assert out.num_consequences >= 1
    cons = out.consequences[0]
    assert cons.most_severe_consequence
    assert cons.allele_string == "G/C"
    assert len(cons.transcript_consequences) > 0


@pytest.mark.integration
def test_vep_invalid_hgvs_surfaces_failure():
    """Malformed HGVS yields a 400 from the server, surfaced as a raised exception."""
    with pytest.raises(Exception):
        run_ensembl_vep(EnsemblVEPInput(hgvs="not-a-real-hgvs"))
