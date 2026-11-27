"""Tests for AlphaFold2 prediction and gradient tools."""

import math
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import MSASubsampling, is_valid_structure
from proto_tools.tools.structure_prediction import (
    AlphaFold2Config,
    AlphaFold2Input,
    Complex,
    run_alphafold2,
)
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from proto_tools.utils.tool_instance import ToolInstance
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex

# Mock standalone_helpers so we can import standalone inference utilities without
# the full ColabDesign/JAX environment that only exists inside the tool venv.
_sh = sys.modules.setdefault("standalone_helpers", types.SimpleNamespace())
for _attr, _val in [
    ("AMINO_ACIDS_LIST", list(PROTEIN_AMINO_ACIDS)),
    ("get_jax_memory_stats", lambda **_kw: {}),
    ("resolve_jax_device", lambda d: d),
    ("serialize_output", lambda v: v),
]:
    if not hasattr(_sh, _attr):
        setattr(_sh, _attr, _val)

_CANONICAL_VOCAB = list(PROTEIN_AMINO_ACIDS)
_HOMOOLIGOMER_SEQ = "MARFLGLYTWHK"
_EXAMPLE_PDB_PATH = (
    Path(__file__).resolve().parents[2]
    / "proto_tools/tools/structure_prediction/alphafold2/examples/example_output/alphafold2_structure/structure_0.pdb"
)
_EXAMPLE_PDB = _EXAMPLE_PDB_PATH.read_text()
# PD-L1 target + nanobody binder complex (from germinal/pdbs/pdl1.pdb).
_GRADIENT_EXAMPLE_PDB_PATH = Path(__file__).resolve().parents[1] / "dummy_data/pdl1.pdb"


@pytest.fixture(scope="module", autouse=True)
def _persistent_worker(request):
    if request.config.getoption("--cpu-only"):
        yield
        return
    with ToolInstance.scope():
        yield


# -- Input/config validation ---------------------------------------------------


def test_input_rejects_non_protein():
    with pytest.raises(ValidationError, match="unsupported entity types"):
        AlphaFold2Input(complexes=["MKTL123"])


def test_input_accepts_x_residue():
    assert AlphaFold2Input(complexes=["MKTLX"]).complexes[0].chain_sequences[0] == "MKTLX"


def test_config_rejects_model_num_and_ensemble_together():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        AlphaFold2Config(model_num=2, num_ensemble_models=3)


def _softmax_rows(logits):
    """Softmax each row of a logits matrix."""
    rows = []
    for row in logits:
        peak = max(row)
        exponentiated = [math.exp(value - peak) for value in row]
        total = sum(exponentiated)
        rows.append([value / total for value in exponentiated])
    return rows


def _load_af2_standalone():
    """Load the AF2 standalone inference module with standalone_helpers mocked."""
    import importlib.util
    import logging as _logging

    if not hasattr(_sh, "get_logger"):
        _sh.get_logger = lambda name=None: _logging.getLogger(name)
    path = (
        Path(__file__).resolve().parents[2]
        / "proto_tools/tools/structure_prediction/alphafold2/standalone/inference.py"
    )
    spec = importlib.util.spec_from_file_location("_af2_si_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- Dispatch contracts --------------------------------------------------------


def test_prediction_dispatch_contract(monkeypatch):
    captured: list[dict] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append(payload)
        return {"pdb": _EXAMPLE_PDB, "avg_plddt": 0.85, "ptm": 0.72, "iptm": None, "avg_pae": 1.5, "pae": None}

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_alphafold2(AlphaFold2Input(complexes=["MKTL"]), AlphaFold2Config(use_msa=False, device="cpu"))

    assert captured[0]["operation"] == "predict"
    assert result.success
    structure = result.structures[0]
    assert is_valid_structure(structure.structure_cif)
    assert structure.b_factor_type.value == "pLDDT"
    # Structure.per_residue_plddt must normalize 0-100 B-factors to [0, 1].
    plddt = structure.per_residue_plddt
    assert plddt is not None and all(0.0 <= v <= 1.0 for v in plddt)
    assert structure.msa_conditioning.chains[0].status == "not_requested"
    assert structure.msa_conditioning.subsampling.mode == "none"


def test_heteromultimer_msa_dispatch_payload(monkeypatch):
    """Heteromultimer with caller-supplied paired MSAs flows per-chain A3Ms (not just chain 0)."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    captured: list[dict] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append(payload)
        return {"pdb": _EXAMPLE_PDB, "avg_plddt": 0.85, "ptm": 0.72, "iptm": 0.6, "avg_pae": 1.5, "pae": None}

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2.ToolInstance.dispatch",
        fake_dispatch,
    )

    # Two distinct chains (heteromultimer), each with a paired (row-aligned) MSA of depth 2.
    msa_a = MSA.from_fasta_string(">qA\nMKTLAV\n>hA1\nMKTRAV\n")
    msa_b = MSA.from_fasta_string(">qB\nGARYTW\n>hB1\nGARYTM\n")
    complex_msas = ComplexMSAs(per_chain={0: msa_a, 1: msa_b}, paired=True)

    result = run_alphafold2(
        AlphaFold2Input(complexes=[["MKTLAV", "GARYTW"]], msas=[complex_msas]),
        AlphaFold2Config(use_msa=False, device="cpu"),
    )

    payload = captured[0]
    assert payload["msa_a3m_content"] is None, (
        "Heteromultimer path must NOT pack a single combined A3M — per_chain_msas_a3m carries the work"
    )
    assert payload["is_paired"] is True
    per_chain_a3m = payload["per_chain_msas_a3m"]
    assert per_chain_a3m is not None and len(per_chain_a3m) == 2, "two-chain complex → two A3Ms"
    assert per_chain_a3m[0] is not None and "MKTLAV" in per_chain_a3m[0]
    assert per_chain_a3m[1] is not None and "GARYTW" in per_chain_a3m[1]
    report = result.structures[0].msa_conditioning
    assert [chain.status for chain in report.chains] == ["consumed", "consumed"]
    assert report.pairing_mode == "paired"
    assert report.subsampling == MSASubsampling(mode="stochastic", max_rows=512, max_extra_rows=1024)


def test_heteromultimer_paired_dispatch_carries_deep_unpaired_a3m(monkeypatch):
    """Paired heteromultimer flows the deep per-chain unpaired MSAs alongside the paired rows."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    captured: list[dict] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append(payload)
        return {"pdb": _EXAMPLE_PDB, "avg_plddt": 0.85, "ptm": 0.72, "iptm": 0.6, "avg_pae": 1.5, "pae": None}

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2.ToolInstance.dispatch",
        fake_dispatch,
    )

    # Shallow paired (depth 2) plus a deeper per-chain unpaired MSA (depth 4) per chain.
    paired_a = MSA.from_fasta_string(">qA\nMKTLAV\n>hA1\nMKTRAV\n")
    paired_b = MSA.from_fasta_string(">qB\nGARYTW\n>hB1\nGARYTM\n")
    unpaired_a = MSA.from_fasta_string(">qA\nMKTLAV\n>uA1\nMKTRAV\n>uA2\nMKSLAV\n>uA3\nMKTLAW\n")
    unpaired_b = MSA.from_fasta_string(">qB\nGARYTW\n>uB1\nGARYTM\n>uB2\nGARYSW\n>uB3\nGTRYTW\n")
    complex_msas = ComplexMSAs(
        per_chain={0: paired_a, 1: paired_b},
        paired=True,
        unpaired_per_chain={0: unpaired_a, 1: unpaired_b},
    )

    run_alphafold2(
        AlphaFold2Input(complexes=[["MKTLAV", "GARYTW"]], msas=[complex_msas]),
        AlphaFold2Config(use_msa=False, device="cpu"),
    )

    payload = captured[0]
    assert payload["is_paired"] is True
    unpaired_a3m = payload["unpaired_per_chain_msas_a3m"]
    assert unpaired_a3m is not None and len(unpaired_a3m) == 2, "two-chain complex → two deep unpaired A3Ms"
    assert unpaired_a3m[0].count(">") == 4 and unpaired_a3m[1].count(">") == 4  # deep unpaired depth carried


def test_homooligomer_uses_single_a3m_dispatch_path(monkeypatch):
    """Homo-oligomer takes the single-A3M path (ColabDesign expands via copies=N)."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    captured: list[dict] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append(payload)
        return {"pdb": _EXAMPLE_PDB, "avg_plddt": 0.85, "ptm": 0.72, "iptm": 0.6, "avg_pae": 1.5, "pae": None}

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2.ToolInstance.dispatch",
        fake_dispatch,
    )

    seq = "MKTLAV"
    msa = MSA.from_fasta_string(f">q\n{seq}\n>h\nMKTRAV\n")
    complex_msas = ComplexMSAs(per_chain={0: msa, 1: msa}, paired=False)

    run_alphafold2(
        AlphaFold2Input(complexes=[[seq, seq]], msas=[complex_msas]),
        AlphaFold2Config(use_msa=False, device="cpu"),
    )

    payload = captured[0]
    assert payload["per_chain_msas_a3m"] is None, "homo-oligomer must NOT use per-chain path"
    assert payload["msa_a3m_content"] is not None and seq in payload["msa_a3m_content"], (
        "homo-oligomer should pass chain 0's MSA as a single A3M for ColabDesign to expand"
    )


# -- GPU integration tests -----------------------------------------------------


@pytest.mark.uses_gpu
def test_homooligomer():
    output = run_alphafold2(
        AlphaFold2Input(complexes=[Complex(chains=[_HOMOOLIGOMER_SEQ, _HOMOOLIGOMER_SEQ])]),
        AlphaFold2Config(use_msa=False),
    )
    assert output.success
    assert is_valid_structure(output.structures[0].structure_cif)
    assert 0 <= output.structures[0].metrics["avg_plddt"] <= 1.0
    assert output.structures[0].metrics["iptm"] is not None


# -- Benchmark ----------------------------------------------------------------


@pytest.mark.benchmark("alphafold2-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphafold2_benchmark(request):
    """Benchmark alphafold2-prediction on the trp heterodimer with MSA (cold + warm).

    Two-chain protein complex (~500 residues total). The MSA search is included because it is
    usually the dominant cost of a real fold, and because nothing else exercises that path.
    Cold pass measures weight load + first inference; warm pass measures inference only.
    """
    complex_ = load_benchmark_complex("trp_heterodimer")
    inputs = AlphaFold2Input(complexes=[complex_])
    config = AlphaFold2Config(use_msa=True, verbose=True)

    result = benchmark_twice(request, "alphafold2", lambda: run_alphafold2(inputs=inputs, config=config))

    assert result.success, "AlphaFold2 benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
