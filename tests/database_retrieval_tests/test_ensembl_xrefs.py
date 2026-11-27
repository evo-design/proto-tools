"""tests/database_retrieval_tests/test_ensembl_xrefs.py.

Tests for the Ensembl REST xrefs wrapper (id → list[EnsemblXref]).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblXrefsInput,
    UniProtFetchInput,
    run_ensembl_xrefs,
    run_uniprot_fetch,
)

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{}, {"ensembl_id": "   "}], ids=["missing", "whitespace"])
def test_validator_rejects_blank_ensembl_id(kwargs):
    """Missing or whitespace-only ID rejected at parse time."""
    with pytest.raises(ValidationError):
        EnsemblXrefsInput(**kwargs)


# ---------------------------------------------------------------------------
# Mocked dispatch — URL + parser
# ---------------------------------------------------------------------------


_XREFS_PAYLOAD = [
    {"dbname": "Uniprot_gn", "display_id": "BRCA1", "primary_id": "P38398", "info_type": "DIRECT"},
    {"dbname": "EntrezGene", "display_id": "BRCA1", "primary_id": "672", "info_type": "DEPENDENT"},
]


def _stub_session(json_payload):
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/xrefs/id/ENSG00000012048"
    response.raise_for_status.return_value = None
    response.json.return_value = json_payload
    session.get.return_value = response
    return session


def test_dispatches_and_parses():
    """End-to-end: build URL, GET, parse list of typed xrefs."""
    session = _stub_session(_XREFS_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_xrefs.build_session",
        return_value=session,
    ):
        out = run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id="ENSG00000012048"))
    assert out.success
    args, _ = session.get.call_args
    assert args[0].endswith("/xrefs/id/ENSG00000012048")
    uniprot = next(x for x in out.result if x.dbname == "Uniprot_gn")
    assert uniprot.primary_id == "P38398"


def test_rejects_non_list_payload():
    """A dict where a list is expected indicates a server-shape regression — surface it."""
    session = _stub_session({"oops": "dict"})
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_xrefs.build_session",
        return_value=session,
    ):
        with pytest.raises(Exception, match="non-list"):
            run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id="ENSG00000012048"))


def test_wraps_corrupt_json_with_context():
    """Non-JSON body surfaces a tight error mentioning the URL."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/xrefs/id/ENSG00000012048"
    response.text = "<html>err</html>"
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_xrefs.build_session",
        return_value=session,
    ):
        with pytest.raises(Exception, match="non-JSON"):
            run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id="ENSG00000012048"))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_workflow_xrefs_then_uniprot_fetch():
    """Cross-tool: ensembl xrefs → uniprot-fetch on the resolved accession returns the same gene name."""
    xrefs = run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id="ENSG00000012048"))
    assert xrefs.success, xrefs.errors
    # Resolve to the canonical reviewed entry; TrEMBL accessions carry no gene names.
    uniprot_ids = {x.primary_id for x in xrefs.result if x.dbname == "Uniprot_gn"}
    assert "P38398" in uniprot_ids
    uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id="P38398"))
    assert uniprot.success
    assert "brca1" in {n.lower() for n in uniprot.gene_names}
