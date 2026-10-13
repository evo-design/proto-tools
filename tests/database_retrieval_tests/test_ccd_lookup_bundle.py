"""Tests for ccd-lookup against a small CCD bundle with entries RDKit cannot sanitize.

Kept apart from ``test_ccd_lookup.py`` because that module's persistent worker is bound to the
real bundle; these tests point ``PROTO_CCD_LOOKUP_WEIGHTS_DIR`` at a fixture before each one-shot call.
"""

import shutil
from pathlib import Path

import pytest

from proto_tools.tools.database_retrieval import CcdLookupConfig, CcdLookupInput, run_ccd_lookup
from tests.tool_infra_tests.test_export_functionality import validate_output

# ATP, SEP, MSE, plus two entries whose sanitization raises outside pdbeccdutils' error handling:
# CZL (Fe8S9 cluster) overflows RDKit's CIP labeler digraph, HB1 (Ru complex) fails to kekulize.
_FIXTURE_BUNDLE = Path(__file__).parents[1] / "dummy_data" / "ccd_lookup_test_examples" / "components.cif"


@pytest.fixture
def fixture_bundle(tmp_path, monkeypatch):
    """Point the ccd_lookup weights dir at a copy of the fixture bundle."""
    shutil.copy(_FIXTURE_BUNDLE, tmp_path / "components.cif")
    monkeypatch.setenv("PROTO_CCD_LOOKUP_WEIGHTS_DIR", str(tmp_path))


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_ignores_unsanitizable_entries_elsewhere_in_the_bundle(fixture_bundle):
    """Entries RDKit cannot sanitize do not break lookups of other codes in the same bundle."""
    result = run_ccd_lookup(CcdLookupInput(identifiers=["ATP", "SEP", "MSE"]), CcdLookupConfig())
    validate_output(result)

    assert [e.fragment.ccd_code for e in result.results] == ["ATP", "SEP", "MSE"]
    assert all(not e.errors for e in result.results)
    assert result.results[0].formula == "C10 H16 N5 O13 P3"
    assert result.results[1].parent_ccd_code == "SER"


@pytest.mark.skip_ci
@pytest.mark.integration
def test_ccd_lookup_falls_back_to_unsanitized_parse_for_unsanitizable_entries(fixture_bundle):
    """A requested entry that fails sanitization still returns its CCD metadata, with a warning."""
    result = run_ccd_lookup(CcdLookupInput(identifiers=["CZL", "HB1", "ATP"]), CcdLookupConfig())
    validate_output(result)
    czl, hb1, atp = result.results

    assert czl.formula == "Fe8 H6 S9"
    assert hb1.formula == "C23 H13 N3 O3 Ru"
    for entry in (czl, hb1):
        assert not entry.errors
        assert any("sanitiz" in w.lower() for w in entry.warnings)
    assert atp.fragment.ccd_code == "ATP" and not atp.warnings
