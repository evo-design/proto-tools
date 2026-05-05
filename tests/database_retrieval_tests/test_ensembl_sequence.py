"""tests/database_retrieval_tests/test_ensembl_sequence.py.

Tests for the Ensembl REST sequence wrapper (id → EnsemblSequence).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblSequenceConfig,
    EnsemblSequenceInput,
    run_ensembl_sequence,
)

# ---------------------------------------------------------------------------
# Mocked dispatch — URL + params + parser
# ---------------------------------------------------------------------------


def test_validator_rejects_blank_ensembl_id():
    """Whitespace-only IDs are rejected before URL construction."""
    with pytest.raises(ValidationError, match="ensembl_id cannot be blank"):
        EnsemblSequenceInput(ensembl_id="   ")


_PROTEIN_PAYLOAD = {
    "id": "ENSP00000350283",  # type=protein on ENST input resolves to the protein ID
    "desc": "BRCA1 protein",
    "molecule": "protein",
    "seq": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKND",
}


def _stub_session(json_payload):
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/sequence/id/ENST00000357654?type=protein"
    response.raise_for_status.return_value = None
    response.json.return_value = json_payload
    session.get.return_value = response
    return session


@pytest.mark.parametrize("sequence_type", ["genomic", "protein"])
def test_url_carries_sequence_type_param(sequence_type):
    """``type`` query param echoes the configured ``sequence_type``."""
    session = _stub_session(_PROTEIN_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_sequence.build_session",
        return_value=session,
    ):
        run_ensembl_sequence(
            EnsemblSequenceInput(ensembl_id="ENST00000357654"),
            EnsemblSequenceConfig(sequence_type=sequence_type),
        )
    _, kwargs = session.get.call_args
    assert kwargs["params"] == {"type": sequence_type}


def test_dispatches_and_parses():
    """Sequence payload round-trips into EnsemblSequence with the seq preserved."""
    session = _stub_session(_PROTEIN_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_sequence.build_session",
        return_value=session,
    ):
        out = run_ensembl_sequence(
            EnsemblSequenceInput(ensembl_id="ENST00000357654"),
            EnsemblSequenceConfig(sequence_type="protein"),
        )
    assert out.success
    # type=protein on an ENST input resolves to the corresponding ENSP server-side.
    assert out.results[0].id == "ENSP00000350283"
    assert out.results[0].mol_type == "protein"
    assert out.results[0].seq.startswith("MDLSAL")


def test_wraps_corrupt_json_with_context():
    """Non-JSON body surfaces a tight error mentioning the URL."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/sequence/id/ENST00000357654"
    response.text = "<html>err</html>"
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    session.get.return_value = response
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_sequence.build_session",
        return_value=session,
    ):
        out = run_ensembl_sequence(EnsemblSequenceInput(ensembl_id="ENST00000357654"))
    assert out.success is False
    assert any("non-JSON" in err for err in out.errors)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_protein_sequence_for_brca1_canonical_live():
    """Protein sequence fetch for BRCA1 canonical transcript returns a non-empty seq."""
    out = run_ensembl_sequence(
        EnsemblSequenceInput(ensembl_id="ENST00000357654"),
        EnsemblSequenceConfig(sequence_type="protein"),
    )
    assert out.success, out.errors
    assert out.results[0].seq
    assert out.results[0].seq.startswith("M")
    # BRCA1 canonical isoform is 1863 aa.
    assert len(out.results[0].seq) == 1863
