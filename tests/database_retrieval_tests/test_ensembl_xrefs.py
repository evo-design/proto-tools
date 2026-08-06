"""tests/database_retrieval_tests/test_ensembl_xrefs.py.

Tests for the Ensembl REST xrefs wrapper (id → list[EnsemblXref]).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblXrefsInput,
    run_ensembl_xrefs,
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
