"""Tests for the IPSAE interface quality scoring tool."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures.structure import BFactorType, Structure
from proto_tools.tools import IPSAEScoringConfig, IPSAEScoringInput, run_ipsae_scoring
from proto_tools.tools.structure_scoring.ipsae.ipsae_scoring import example_input
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

FIXTURE_DIR = Path(__file__).parent.parent.parent / "proto_tools" / "tools" / "structure_scoring" / "ipsae"


def _bundled_structure(*, with_pae: bool = True) -> Structure:
    """Load the bundled IPSAE fixture (4INS insulin A+B) with optional PAE matrix."""
    if with_pae:
        return example_input().structure
    return Structure.from_file(FIXTURE_DIR / "example_input_fixture.pdb", b_factor_type=BFactorType.PLDDT, metrics={})


# ── Input validation ──────────────────────────────────────────────────────────


def test_binder_in_target_chains_rejected():
    """binder_chain appearing in target_chains must raise."""
    with pytest.raises(ValidationError, match="must not appear in target_chains"):
        IPSAEScoringInput(structure=_bundled_structure(), binder_chain="A", target_chains=["A", "B"])


def test_missing_pae_matrix_rejected():
    """Structure without a PAE matrix must raise."""
    with pytest.raises(ValidationError, match="pae_matrix"):
        IPSAEScoringInput(structure=_bundled_structure(with_pae=False), binder_chain="A", target_chains=["B"])


def test_unknown_chain_rejected():
    """Chain ID not present in the structure must raise."""
    with pytest.raises(ValidationError, match="not found in structure"):
        IPSAEScoringInput(structure=_bundled_structure(), binder_chain="Z", target_chains=["B"])


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ipsae_result():
    """Run IPSAE once and share across integration tests."""
    return run_ipsae_scoring(example_input(), IPSAEScoringConfig())


@pytest.mark.integration
def test_run_ipsae_scoring_success(ipsae_result):
    """End-to-end run returns success with populated metrics."""
    assert ipsae_result.success
    assert ipsae_result.tool_id == "ipsae-scoring"
    m = ipsae_result.metrics
    assert m.ipsae >= 0
    assert m.pdockq2 >= 0
    assert m.lis >= 0
    assert m.pdockq >= 0
    assert m.iptm_d0chn >= 0
    assert len(m.chain_pair_results) > 0
    for pair in m.chain_pair_results:
        assert pair.pair_type in ("asym", "max")


@pytest.mark.integration
def test_metrics_satisfy_spec(ipsae_result):
    """All emitted metrics must be within declared metric_spec bounds."""
    assert_metrics_in_spec(ipsae_result)


@pytest.mark.integration
def test_export_json_round_trip(ipsae_result, tmp_path):
    """Exported JSON should round-trip all metric values."""
    ipsae_result.export(name="ipsae_result", export_path=str(tmp_path), file_format="json")
    payload = json.loads((tmp_path / "ipsae_result.json").read_text())
    assert payload["ipsae"] == pytest.approx(ipsae_result.metrics.ipsae)
    assert payload["pdockq2"] == pytest.approx(ipsae_result.metrics.pdockq2)
    assert payload["lis"] == pytest.approx(ipsae_result.metrics.lis)
