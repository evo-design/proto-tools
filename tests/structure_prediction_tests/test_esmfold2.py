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
from tests._structure_fixtures import synthetic_cif
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


def test_esmfold2_full_checkpoint_reports_random_1024_row_draw(monkeypatch):
    """The MSA checkpoint consumes primary rows under the pinned 1024-row msa_max_depth draw."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import _run_esmfold2_on_complex
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.esmfold2.esmfold2.ToolInstance.dispatch",
        lambda *args, **kwargs: {
            "structure_cif_output": synthetic_cif(["A", "B"]),
            "metrics": {"plddt": 0.8, "ptm": 0.7, "avg_pae": 1.0},
        },
    )
    complex_ = Complex(
        chains=[Chain(sequence=_HELIX_A, entity_type="protein"), Chain(sequence=_HELIX_B, entity_type="protein")]
    )
    structure = _run_esmfold2_on_complex(
        ESMFold2Config(model_checkpoint="esmfold2"),
        complex_,
        complex_msas=ComplexMSAs(
            per_chain={0: MSA(aligned_sequences=[_HELIX_A, _HELIX_A])},
            unpaired_per_chain={1: MSA(aligned_sequences=[_HELIX_B, _HELIX_B])},
        ),
    )

    report = structure.msa_conditioning
    assert [chain.status for chain in report.chains] == ["consumed", "ignored"]
    assert (report.subsampling.mode, report.subsampling.max_rows) == ("stochastic", 1024)


def test_esmfold2_fast_reports_supplied_msa_as_ignored(monkeypatch):
    """The single-sequence checkpoint must not report supplied rows as consumed."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.esmfold2.esmfold2 import _run_esmfold2_on_complex
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    def fake_dispatch(*args, **kwargs):
        return {
            "structure_cif_output": synthetic_cif(["A"]),
            "metrics": {"plddt": 0.8, "ptm": 0.7, "avg_pae": 1.0},
        }

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.esmfold2.esmfold2.ToolInstance.dispatch",
        fake_dispatch,
    )
    msa = MSA(aligned_sequences=[_HELIX_A, _HELIX_A])
    structure = _run_esmfold2_on_complex(
        ESMFold2Config(model_checkpoint="esmfold2-fast", use_msa=False),
        Complex(chains=[Chain(sequence=_HELIX_A, entity_type="protein")]),
        complex_msas=ComplexMSAs(per_chain={0: msa}),
    )

    chain = structure.msa_conditioning.chains[0]
    assert chain.status == "ignored"
    assert chain.ignored_reason == "proto-tools sends no MSA to esmfold2-fast, a single-sequence checkpoint"
    assert structure.msa_conditioning.subsampling.mode == "none"


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


@pytest.mark.integration
def test_esmfold2_fold_accepts_pinned_msa_max_depth():
    """fold() takes the msa_max_depth the wrapper pins and reports."""
    from proto_tools.utils import run_in_env

    code = (
        "import inspect\n"
        "from esm.models.esmfold2 import ESMFold2InputBuilder\n"
        "print('msa_max_depth' in inspect.signature(ESMFold2InputBuilder.fold).parameters)\n"
    )
    assert run_in_env("esmfold2", code=code, timeout=1800).strip().splitlines()[-1] == "True"
