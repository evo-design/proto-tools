"""tests/database_retrieval_tests/test_ensembl_overlap.py.

Tests for the Ensembl REST overlap wrapper (id → list[EnsemblOverlapFeatureRecord]).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.database_retrieval import (
    EnsemblOverlapConfig,
    EnsemblOverlapInput,
    run_ensembl_overlap,
)

# ---------------------------------------------------------------------------
# Mocked dispatch — URL + params + parser
# ---------------------------------------------------------------------------


def test_validator_rejects_blank_ensembl_id():
    """Whitespace-only IDs are rejected before URL construction."""
    with pytest.raises(ValidationError, match="ensembl_id cannot be blank"):
        EnsemblOverlapInput(ensembl_id="   ")


def test_overlap_feature_rejects_translation_exon_endpoint_value():
    """translation_exon belongs to /overlap/translation, not /overlap/id."""
    with pytest.raises(ValidationError):
        EnsemblOverlapConfig(overlap_feature="translation_exon")


_OVERLAP_PAYLOAD = [
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


def _stub_session(json_payload):
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.url = "https://rest.ensembl.org/overlap/id/ENSG00000012048?feature=gene"
    response.raise_for_status.return_value = None
    response.json.return_value = json_payload
    session.get.return_value = response
    return session


@pytest.mark.parametrize("overlap_feature", ["gene", "regulatory", "variation"])
def test_url_carries_overlap_feature_param(overlap_feature):
    """``feature`` query param echoes the configured ``overlap_feature``."""
    session = _stub_session(_OVERLAP_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_overlap.build_session",
        return_value=session,
    ):
        run_ensembl_overlap(
            EnsemblOverlapInput(ensembl_id="ENSG00000012048"),
            EnsemblOverlapConfig(overlap_feature=overlap_feature),
        )
    args, kwargs = session.get.call_args
    assert args[0].endswith("/overlap/id/ENSG00000012048")
    assert kwargs["params"] == {"feature": overlap_feature}


def test_dispatches_and_keeps_raw_for_feature_specific_keys():
    """Overlap features differ in shape; parser keeps the raw dict alongside the typed common fields."""
    session = _stub_session(_OVERLAP_PAYLOAD)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_overlap.build_session",
        return_value=session,
    ):
        out = run_ensembl_overlap(EnsemblOverlapInput(ensembl_id="ENSG00000012048"))
    assert out.success
    assert len(out.result) == 2
    gene_record, reg_record = out.result
    assert gene_record.feature_type == "gene"
    assert gene_record.raw["external_name"] == "BRCA1"
    assert reg_record.feature_type == "regulatory"
    assert reg_record.id is None


def test_raw_override_wins_if_api_adds_raw_key():
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
    session = _stub_session(payload)
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_overlap.build_session",
        return_value=session,
    ):
        out = run_ensembl_overlap(EnsemblOverlapInput(ensembl_id="ENSG00000012048"))
    assert out.success
    assert out.result[0].raw == payload[0]


def test_rejects_non_dict_element():
    """A non-dict element raises a clear ValueError mentioning the index."""
    session = _stub_session(["not a dict"])
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_overlap.build_session",
        return_value=session,
    ):
        out = run_ensembl_overlap(EnsemblOverlapInput(ensembl_id="ENSG00000012048"))
    assert out.success is False
    assert any("payload[0] is non-dict" in err for err in out.errors)


def test_rejects_non_list_payload():
    """Type-mismatch (list expected, dict received) raises a clear error."""
    session = _stub_session({"oops": "dict"})
    with patch(
        "proto_tools.tools.database_retrieval.ensembl.ensembl_overlap.build_session",
        return_value=session,
    ):
        out = run_ensembl_overlap(EnsemblOverlapInput(ensembl_id="ENSG00000012048"))
    assert out.success is False
    assert any("non-list" in err for err in out.errors)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_overlap_brca1_exons_live():
    """Overlap with feature=exon for BRCA1 returns at least 10 records."""
    out = run_ensembl_overlap(
        EnsemblOverlapInput(ensembl_id="ENSG00000012048"),
        EnsemblOverlapConfig(overlap_feature="exon"),
    )
    assert out.success, out.errors
    assert len(out.result) >= 10
    assert all(r.feature_type == "exon" for r in out.result)
