"""tests/structure_prediction_tests/test_esmfold2.py.

Tests for ESMFold2 all-atom complex structure prediction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from proto_tools.entities.complex import Chain, Complex
from proto_tools.entities.ligands import Fragment
from proto_tools.tools.structure_prediction.esmfold2 import (
    ESMFold2Config,
    ESMFold2Input,
    ESMFold2Metrics,
    ESMFold2Output,
    run_esmfold2,
)
from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import _chain_to_payload
from tests.conftest import benchmark_twice, random_protein_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

# Two short helix peptides forming a heterodimer.
_HELIX_A = "EELLKKLEELLKKLEELLKK"
_HELIX_B = "KKLLEEKKLLEEKKLLEEKK"


# ── Config validator (no GPU) ───────────────────────────────────────────────


def test_esmfold2_config_forbids_msa_with_fast_variant():
    """ESMFold2-Fast is single-sequence; pairing it with use_msa=True must raise."""
    with pytest.raises(ValidationError, match="does not support MSA conditioning"):
        ESMFold2Config(model_checkpoint="esmfold2-fast", use_msa=True)


# ── Serialization tests (no GPU) ─────────────────────────────────────────────


def test_esmfold2_chain_to_payload_stamps_resolved_id():
    """The worker payload carries the resolved chain ID plus the chain's content."""
    protein = _chain_to_payload(Chain(sequence=_HELIX_A, entity_type="protein"), "A")
    assert protein == {"id": "A", "entity_type": "protein", "sequence": _HELIX_A}

    ligand = _chain_to_payload(Fragment(ccd_code="ATP"), "L")
    assert ligand == {"id": "L", "entity_type": "ligand", "ccd_code": "ATP"}


def test_esmfold2_payload_carries_deep_unpaired_msas():
    """The worker payload carries the deep per-chain unpaired MSAs alongside the paired rows.

    The standalone reads the paired rows (``key=`` headers) for cross-chain pairing and
    appends the unpaired rows block-diagonally for per-chain depth.
    """
    from unittest.mock import patch

    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import _run_esmfold2_on_complex
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    class _Stop(Exception):
        pass

    cx = Complex(
        chains=[Chain(sequence=_HELIX_A, entity_type="protein"), Chain(sequence=_HELIX_B, entity_type="protein")]
    )
    paired = {i: MSA(aligned_sequences=[s, s]) for i, s in enumerate([_HELIX_A, _HELIX_B])}  # 2 rows
    unpaired = {i: MSA(aligned_sequences=[s, s, s, s, s]) for i, s in enumerate([_HELIX_A, _HELIX_B])}  # 5 rows
    complex_msas = ComplexMSAs(per_chain=paired, paired=True, unpaired_per_chain=unpaired)

    captured: dict = {}

    def fake_dispatch(_name, input_data, **_kwargs):
        captured["input"] = input_data
        raise _Stop

    with (
        patch(
            "proto_tools.tools.structure_prediction.esmfold2.esmfold2.ToolInstance.dispatch",
            side_effect=fake_dispatch,
        ),
        pytest.raises(_Stop),
    ):
        _run_esmfold2_on_complex(
            ESMFold2Config(model_checkpoint="esmfold2", use_msa=True), cx, complex_msas=complex_msas
        )

    payload = captured["input"]
    assert payload["msas_paired"] is True
    assert payload["msas"] == {"0": [_HELIX_A, _HELIX_A], "1": [_HELIX_B, _HELIX_B]}  # paired rows
    assert payload["unpaired_msas"] == {"0": [_HELIX_A] * 5, "1": [_HELIX_B] * 5}  # deep unpaired depth


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("esmfold2-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold2_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark esmfold2-prediction: 8 random 150-residue proteins on the deployed esmfold2-fast checkpoint, MSA-free (cold + warm)."""
    sequences = random_protein_sequences(n=8, length=150, seed=0)
    inputs = ESMFold2Input(complexes=sequences)
    config = ESMFold2Config(model_checkpoint="esmfold2-fast", use_msa=False, seed=0)

    result = benchmark_twice(request, "esmfold2", lambda: run_esmfold2(inputs, config))

    assert result.success, "ESMFold2 benchmark run failed"
    assert result.tool_id == "esmfold2-prediction"
    assert isinstance(result, ESMFold2Output)
    assert len(result.structures) == 8
    for structure in result.structures:
        assert structure.structure_cif and len(structure.structure_cif) > 0
        assert isinstance(structure.metrics, ESMFold2Metrics)
    assert_metrics_in_spec(result)
