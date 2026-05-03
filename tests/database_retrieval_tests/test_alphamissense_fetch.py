"""tests/database_retrieval_tests/test_alphamissense_fetch.py.

Tests for the AlphaMissense fetch tool.
"""

from unittest.mock import MagicMock

import pytest

from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig,
    AlphaMissenseFetchInput,
    UniProtFetchConfig,
    UniProtFetchInput,
    run_alphamissense_fetch,
    run_uniprot_fetch,
)
from proto_tools.tools.database_retrieval.alphamissense.alphamissense_fetch import (
    _fetch_csv,
    _parse_row,
)


@pytest.mark.parametrize(
    "variant, score, am_class, expected_position, expected_wt, expected_alt, expected_classification",
    [
        ("M1A", "0.4065", "Amb", 1, "M", "A", "ambiguous"),
        ("E2A", "0.1089", "LBen", 2, "E", "A", "likely_benign"),
        ("L175P", "0.95", "LPath", 175, "L", "P", "likely_pathogenic"),
    ],
    ids=["ambiguous", "benign", "pathogenic"],
)
def test_parse_row_decodes_variant_and_class(
    variant, score, am_class, expected_position, expected_wt, expected_alt, expected_classification
):
    """Variant string parsing covers single-, double-, and triple-digit positions and all 3 class codes."""
    row = {"protein_variant": variant, "am_pathogenicity": score, "am_class": am_class}
    prediction = _parse_row(row, "url")
    assert prediction.position == expected_position
    assert prediction.wild_type_aa == expected_wt
    assert prediction.alt_aa == expected_alt
    assert prediction.pathogenicity_score == pytest.approx(float(score))
    assert prediction.classification == expected_classification


@pytest.mark.parametrize(
    "row, error_pattern",
    [
        ({"protein_variant": "X", "am_pathogenicity": "0.5", "am_class": "Amb"}, "Malformed protein_variant"),
        ({"protein_variant": "M1A", "am_pathogenicity": "0.5", "am_class": "Unknown"}, "Unknown am_class"),
    ],
    ids=["malformed-variant", "unknown-class"],
)
def test_parse_row_rejects_malformed_input(row, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        _parse_row(row, "url")


def test_fetch_csv_returns_none_on_404():
    """A 404 from AFDB indicates no AlphaMissense data for the accession (e.g. non-human protein)."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    session.get.return_value = response
    assert _fetch_csv("https://example.com/missing.csv", session) is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_alphamissense_fetch_full_p04637_predictions():
    """TP53 (P04637) returns the full saturation grid: 393 residues x 19 alts = 7467 predictions."""
    output = run_alphamissense_fetch(AlphaMissenseFetchInput(uniprot_id="P04637"), AlphaMissenseFetchConfig())
    assert output.success
    assert output.tool_id == "alphamissense-fetch"
    assert output.uniprot_accession == "P04637"
    assert output.num_predictions == 393 * 19
    assert output.mean_pathogenicity is not None
    assert 0.0 <= output.mean_pathogenicity <= 1.0
    valid_classes = {"likely_benign", "ambiguous", "likely_pathogenic"}
    for prediction in output.predictions:
        assert 0.0 <= prediction.pathogenicity_score <= 1.0
        assert prediction.classification in valid_classes
    assert output.source_url.endswith("AF-P04637-F1-aa-substitutions.csv")


@pytest.mark.integration
def test_alphamissense_fetch_known_pathogenic_tp53_r175h():
    """R175H is the most common cancer-driver TP53 mutation; AlphaMissense must call it pathogenic.

    Filtering happens client-side now; the wrapper returns the full grid and
    the caller picks the position/alt of interest.
    """
    output = run_alphamissense_fetch(
        AlphaMissenseFetchInput(uniprot_id="P04637"),
        AlphaMissenseFetchConfig(),
    )
    assert output.success
    r175h = next(p for p in output.predictions if p.position == 175 and p.alt_aa == "H")
    assert r175h.wild_type_aa == "R"
    assert r175h.classification == "likely_pathogenic"
    assert r175h.pathogenicity_score > 0.7  # well above the 0.564 likely-pathogenic threshold


@pytest.mark.integration
def test_alphamissense_fetch_client_side_filter_at_hotspots():
    """Realistic constraint loop: caller post-filters to confidently pathogenic at TP53 hotspots 175/248/273."""
    output = run_alphamissense_fetch(
        AlphaMissenseFetchInput(uniprot_id="P04637"),
        AlphaMissenseFetchConfig(),
    )
    assert output.success
    assert output.num_predictions == 393 * 19

    hotspot_positions = {175, 248, 273}
    high_confidence_pathogenic = [
        p for p in output.predictions if p.position in hotspot_positions and p.pathogenicity_score >= 0.8
    ]
    assert 0 < len(high_confidence_pathogenic) <= 3 * 19
    for prediction in high_confidence_pathogenic:
        assert prediction.classification == "likely_pathogenic"


@pytest.mark.integration
@pytest.mark.parametrize(
    "uniprot_id, reason",
    [
        ("Q0Q0Q0", "fake accession"),
        ("P0A6F5", "non-human (E. coli GroEL)"),
    ],
    ids=["fake-accession", "non-human"],
)
def test_alphamissense_fetch_returns_failure_when_no_data(uniprot_id, reason):
    """Both bogus accessions and non-human accessions surface as a failure with a clear error."""
    output = run_alphamissense_fetch(AlphaMissenseFetchInput(uniprot_id=uniprot_id), AlphaMissenseFetchConfig())
    assert output.success is False, f"unexpected success for {reason}: {uniprot_id}"
    assert output.tool_id == "alphamissense-fetch"
    assert any("AlphaMissense has no predictions" in err for err in output.errors)


@pytest.mark.integration
def test_workflow_uniprot_then_alphamissense_grid_consistency():
    """End-to-end workflow: gene symbol -> UniProt accession -> AlphaMissense saturation grid.

    Steps:
      1. UniProt: resolve KRAS in human (`prefer_pdb_crossref` to bias the ranker
         toward the Swiss-Prot reviewed canonical entry P01116).
      2. AlphaMissense: pull all predictions for that accession.
      3. Verify the saturation grid is complete (length x 19 alts) AND that the
         per-position wild-type letters in the AlphaMissense grid agree with the
         UniProt canonical sequence -- a silent disagreement would mean the
         constraint is scoring the wrong residues.

    This catches the cross-tool failure mode the previous tests don't: AlphaMissense
    indexes by canonical isoform too, but if the ranker change ever surfaced a
    different isoform from UniProt, the wild-type letters would no longer line up.
    """
    # 1. UniProt: KRAS canonical
    uniprot = run_uniprot_fetch(
        UniProtFetchInput(target_name="KRAS", organism="Homo sapiens", prefer_pdb_crossref=True),
        UniProtFetchConfig(),
    )
    assert uniprot.success
    assert uniprot.accession == "P01116"
    assert uniprot.length == 189

    # 2. AlphaMissense: full saturation grid
    am = run_alphamissense_fetch(
        AlphaMissenseFetchInput(uniprot_id=uniprot.accession),
        AlphaMissenseFetchConfig(),
    )
    assert am.success
    assert am.uniprot_accession == "P01116"
    assert am.num_predictions == uniprot.length * 19  # 189 * 19 = 3591

    # 3. Cross-tool sequence consistency: every WT letter in AlphaMissense's grid
    #    must match the UniProt canonical sequence at the same 1-indexed position.
    wt_by_position: dict[int, str] = {}
    for prediction in am.predictions:
        existing = wt_by_position.setdefault(prediction.position, prediction.wild_type_aa)
        assert existing == prediction.wild_type_aa, (
            f"AlphaMissense reports inconsistent WT letters at position {prediction.position}: "
            f"{existing!r} vs {prediction.wild_type_aa!r}"
        )
    for position, wt in wt_by_position.items():
        assert wt == uniprot.sequence[position - 1], (
            f"WT mismatch at position {position}: AlphaMissense={wt!r}, UniProt={uniprot.sequence[position - 1]!r}"
        )


@pytest.mark.integration
def test_alphamissense_fetch_hg38_returns_genomic_coords():
    """`coordinate_system='hg38'` returns the genomic-coord CSV with chrom/pos/ref/alt populated."""
    output = run_alphamissense_fetch(
        AlphaMissenseFetchInput(uniprot_id="P04637"),
        AlphaMissenseFetchConfig(coordinate_system="hg38"),
    )
    assert output.success
    # Genomic CSV is a subset (SNV-accessible only) — fewer rows than the full grid.
    assert 0 < output.num_predictions < 393 * 19
    assert output.source_url.endswith("AF-P04637-F1-hg38.csv")

    # Every prediction has the genomic-coordinate fields populated.
    for prediction in output.predictions:
        assert prediction.chrom and prediction.chrom.startswith("chr")
        assert prediction.pos is not None and prediction.pos > 0
        assert prediction.ref in {"A", "C", "G", "T"}
        assert prediction.alt in {"A", "C", "G", "T"}
        assert prediction.transcript_id and prediction.transcript_id.startswith("ENST")


@pytest.mark.integration
def test_alphamissense_fetch_hg19_vs_hg38_have_same_classifications():
    """A given protein variant gets the same pathogenicity score in hg19 vs hg38 mode."""
    hg19 = run_alphamissense_fetch(
        AlphaMissenseFetchInput(uniprot_id="P04637"),
        AlphaMissenseFetchConfig(coordinate_system="hg19"),
    )
    hg38 = run_alphamissense_fetch(
        AlphaMissenseFetchInput(uniprot_id="P04637"),
        AlphaMissenseFetchConfig(coordinate_system="hg38"),
    )
    assert hg19.success and hg38.success
    # Same protein variants are reachable by SNV in both genome builds, so the
    # row counts and per-variant scores must match.
    assert hg19.num_predictions == hg38.num_predictions
    hg19_by_variant = {(p.position, p.alt_aa): p.pathogenicity_score for p in hg19.predictions}
    for p38 in hg38.predictions:
        score19 = hg19_by_variant.get((p38.position, p38.alt_aa))
        if score19 is not None:
            assert score19 == pytest.approx(p38.pathogenicity_score, abs=1e-6)
