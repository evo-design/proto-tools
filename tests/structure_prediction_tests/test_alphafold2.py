"""Tests for AlphaFold2 prediction and gradient tools."""

import json
import math
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import BFactorType, Structure, is_valid_structure
from proto_tools.tools.structure_prediction import (
    AlphaFold2BinderConfig,
    AlphaFold2BinderInput,
    AlphaFold2BinderOutput,
    AlphaFold2Config,
    AlphaFold2Input,
    StructurePredictionComplex,
    run_alphafold2,
    run_alphafold2_binder,
)
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from proto_tools.utils.tool_instance import ToolInstance
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests.test_export_functionality import validate_output

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


def test_gradient_input_requires_20_columns():
    AlphaFold2BinderInput(logits=[[0.0] * 20, [1.0] * 20], target_pdb=_GRADIENT_EXAMPLE_PDB_PATH, temperature=0.75)
    with pytest.raises(ValidationError, match="20 columns"):
        AlphaFold2BinderInput(logits=[[0.0] * 19], target_pdb=_GRADIENT_EXAMPLE_PDB_PATH, temperature=1.0)


def test_gradient_config_rejects_bad_loss_weights():
    AlphaFold2BinderConfig(loss_weights={"helix": -0.3})
    with pytest.raises(ValidationError, match="Unknown loss_weights"):
        AlphaFold2BinderConfig(loss_weights={"pldd": 1.0})


def test_gradient_input_requires_target_pdb():
    with pytest.raises(ValidationError, match="target_pdb"):
        AlphaFold2BinderInput(logits=[[0.0] * 20], temperature=1.0)


# -- Dispatch contracts --------------------------------------------------------


def test_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(toolkit=toolkit, payload=payload)
        # target_pdb is a tempfile that the wrapper cleans up after dispatch returns;
        # snapshot its content here while it still exists.
        captured["target_pdb_content"] = Path(payload["target_pdb"]).read_text()
        return {
            "gradient": [[0.1] * 20] * 2,
            "loss": 1.25,
            "metrics": {"avg_plddt": 0.8, "ptm": 0.7, "iptm": 0.5, "avg_pae": 1.5, "pae": None},
            "vocab": _CANONICAL_VOCAB,
            "pdb": _EXAMPLE_PDB,
        }

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2_binder.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = AlphaFold2BinderInput(
        logits=[[0.0] * 20, [1.0] * 20],
        temperature=0.8,
        target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
        target_chain="A",
        binder_chain="B",
        design_positions=[0, 1],
    )
    config = AlphaFold2BinderConfig(
        loss_weights={"plddt": 1.0},
        device="cpu",
    )
    result = run_alphafold2_binder(inputs=inputs, config=config)

    validate_output(result)
    assert captured["toolkit"] == "alphafold2"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert payload["temperature"] == 0.8
    # target_pdb is materialized to a tempfile (path is local-to-runtime, not the
    # caller's input path). The dispatch sees a real PDB-shaped file at that path.
    assert payload["target_pdb"].endswith(".pdb")
    assert captured["target_pdb_content"].strip().startswith(("ATOM", "HEADER"))
    assert payload["backend"] == "base"
    assert payload["soft"] == 1.0
    assert payload["recycle_mode"] == "last"
    assert payload["starting_binder_seq"] is None
    assert result.gradient == [[0.1] * 20] * 2
    assert result.loss == 1.25
    assert result.structure.source == "alphafold2-binder"
    assert result.structure.b_factor_type.value == "pLDDT"
    assert is_valid_structure(result.structure.structure)
    # per_residue_plddt must be normalized to [0, 1] for downstream consumers
    # (proto-language's pLDDT-weighted semigreedy uses `1 - plddt` for sampling).
    plddt = result.structure.per_residue_plddt
    assert plddt is not None
    assert all(0.0 <= v <= 1.0 for v in plddt)


@pytest.mark.parametrize("name", ["step", "step_v1.5"])
def test_gradient_export_writes_pdb_sidecar(tmp_path, name):
    AlphaFold2BinderOutput(
        gradient=[[0.1] * 20] * 2,
        loss=1.25,
        metrics={"avg_plddt": 0.8},
        vocab=_CANONICAL_VOCAB,
        structure=Structure(structure=_EXAMPLE_PDB, b_factor_type=BFactorType.PLDDT),
    ).export(name, export_path=tmp_path)

    payload = json.loads((tmp_path / f"{name}.json").read_text())
    assert payload["structure_pdb"] == f"{name}.pdb"
    assert is_valid_structure((tmp_path / f"{name}.pdb").read_text())


def test_gradient_dispatch_forwards_recycle_mode_and_starting_seq(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(payload=payload)
        return {
            "gradient": [[0.0] * 20] * 3,
            "loss": 0.0,
            "metrics": {"avg_plddt": 0.5, "ptm": 0.5, "avg_pae": 1.0, "pae": None},
            "vocab": _CANONICAL_VOCAB,
            "pdb": _EXAMPLE_PDB,
        }

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2_binder.ToolInstance.dispatch",
        fake_dispatch,
    )

    run_alphafold2_binder(
        AlphaFold2BinderInput(
            logits=[[0.0] * 20] * 3,
            temperature=1.0,
            target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
            binder_chain="B",
        ),
        AlphaFold2BinderConfig(
            recycle_mode="average",
            starting_binder_seq="EVQ",
            backend="germinal",
            device="cpu",
        ),
    )
    assert captured["payload"]["recycle_mode"] == "average"
    assert captured["payload"]["starting_binder_seq"] == "EVQ"
    assert captured["payload"]["backend"] == "germinal"
    assert captured["payload"]["compute_gradient"] is True


def test_gradient_dispatch_omits_embedded_ablm_metadata(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(payload=payload)
        return {
            "gradient": [[0.0] * 20] * 3,
            "loss": 0.0,
            "metrics": {"avg_plddt": 0.5, "ptm": 0.5, "avg_pae": 1.0, "pae": None},
            "vocab": _CANONICAL_VOCAB,
            "pdb": _EXAMPLE_PDB,
        }

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2_binder.ToolInstance.dispatch",
        fake_dispatch,
    )

    run_alphafold2_binder(
        AlphaFold2BinderInput(
            logits=[[0.0] * 20] * 3,
            target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
            binder_chain="B",
        ),
        AlphaFold2BinderConfig(backend="germinal", device="cpu"),
    )

    assert "ablm_model" not in captured["payload"]
    assert "cdr_lengths" not in captured["payload"]
    assert "framework_lengths" not in captured["payload"]
    assert "ablm_temp" not in captured["payload"]
    assert "iglm_species" not in captured["payload"]


def test_forward_mode_dispatch_contract(monkeypatch):
    """compute_gradient=False forwards the flag and returns gradient=None."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update(payload=payload)
        return {
            "gradient": None,
            "loss": 0.75,
            "metrics": {"avg_plddt": 0.82, "ptm": 0.65, "iptm": 0.55, "avg_pae": 2.1, "pae": None},
            "vocab": _CANONICAL_VOCAB,
            "pdb": _EXAMPLE_PDB,
        }

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2_binder.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_alphafold2_binder(
        AlphaFold2BinderInput(
            logits=[[0.0] * 20] * 2,
            target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
            binder_chain="B",
        ),
        AlphaFold2BinderConfig(compute_gradient=False, hard=1.0, device="cpu"),
    )

    assert captured["payload"]["compute_gradient"] is False
    assert captured["payload"]["hard"] == 1.0
    assert result.gradient is None
    assert result.loss == 0.75
    assert result.structure.source == "alphafold2-binder"
    assert result.structure.per_residue_plddt is not None


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


# -- GPU integration tests -----------------------------------------------------


@pytest.mark.uses_gpu
def test_gradient_end_to_end():
    result = run_alphafold2_binder(
        AlphaFold2BinderInput(
            logits=[[0.0] * 20] * 5,
            temperature=1.0,
            target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
            target_chain="A",
            binder_chain="B",
        ),
        AlphaFold2BinderConfig(
            num_recycles=1,
            loss_weights={"plddt": 1.0},
        ),
    )
    validate_output(result)
    assert result.tool_id == "alphafold2-binder"
    assert len(result.gradient) == 5
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert math.isfinite(result.loss) and result.loss != 0.0
    assert result.vocab == _CANONICAL_VOCAB
    assert 0 <= result.metrics["avg_plddt"] <= 1.0
    # Structure must be populated with per-residue pLDDT normalized to [0, 1]
    # for proto-language's pLDDT-weighted semigreedy (1 - plddt sampling).
    assert result.structure.source == "alphafold2-binder"
    assert is_valid_structure(result.structure.structure)
    plddt = result.structure.per_residue_plddt
    assert plddt is not None and all(0.0 <= v <= 1.0 for v in plddt)


@pytest.mark.uses_gpu
def test_forward_vs_gradient_consistency():
    """compute_gradient=False should produce loss/metrics close to gradient mode.

    XLA compiles ``fn`` and ``value_and_grad(fn)`` separately; FP reductions
    across dropout-enabled Evoformer layers drift a few percent even with
    the same seed, so the tolerance is generous rather than bit-exact.
    """
    target_kwargs = {
        "target_pdb": str(_GRADIENT_EXAMPLE_PDB_PATH),
        "target_chain": "A",
        "binder_chain": "B",
    }
    inputs = AlphaFold2BinderInput(logits=[[0.0] * 20] * 5, temperature=1.0, **target_kwargs)
    grad = run_alphafold2_binder(inputs, AlphaFold2BinderConfig(num_recycles=1, loss_weights={"plddt": 1.0}, seed=42))
    fwd = run_alphafold2_binder(
        inputs,
        AlphaFold2BinderConfig(num_recycles=1, loss_weights={"plddt": 1.0}, seed=42, compute_gradient=False),
    )
    assert grad.gradient is not None and fwd.gradient is None
    assert math.isclose(grad.loss, fwd.loss, rel_tol=0.1)
    assert math.isclose(grad.metrics["avg_plddt"], fwd.metrics["avg_plddt"], rel_tol=0.1)
    assert math.isclose(grad.metrics["ptm"], fwd.metrics["ptm"], rel_tol=0.1)


@pytest.mark.uses_gpu
def test_gradient_recycle_mode_threads_through():
    """Non-default recycle_mode must reach ``_get_model`` without tripping the cache."""
    target_kwargs = {
        "target_pdb": str(_GRADIENT_EXAMPLE_PDB_PATH),
        "target_chain": "A",
        "binder_chain": "B",
    }
    result = run_alphafold2_binder(
        AlphaFold2BinderInput(logits=[[0.0] * 20] * 5, temperature=1.0, **target_kwargs),
        AlphaFold2BinderConfig(num_recycles=2, recycle_mode="first", loss_weights={"plddt": 1.0}),
    )
    validate_output(result)
    assert all(math.isfinite(v) for row in result.gradient for v in row)


@pytest.mark.uses_gpu
def test_gradient_sensitivity():
    target_kwargs = {
        "target_pdb": str(_GRADIENT_EXAMPLE_PDB_PATH),
        "target_chain": "A",
        "binder_chain": "B",
    }
    config = AlphaFold2BinderConfig(num_recycles=1, loss_weights={"plddt": 1.0})
    r1 = run_alphafold2_binder(AlphaFold2BinderInput(logits=[[0.0] * 20] * 5, temperature=1.0, **target_kwargs), config)
    r2 = run_alphafold2_binder(
        AlphaFold2BinderInput(logits=[[5.0] + [0.0] * 19] * 5, temperature=1.0, **target_kwargs), config
    )
    assert r1.loss != r2.loss
    assert r1.gradient != r2.gradient


@pytest.mark.uses_gpu
def test_gradient_loss_weights_matter():
    logits = [[0.1 * i + 0.05 * j for j in range(20)] for i in range(5)]
    target_kwargs = {
        "target_pdb": str(_GRADIENT_EXAMPLE_PDB_PATH),
        "target_chain": "A",
        "binder_chain": "B",
    }
    r_plddt = run_alphafold2_binder(
        AlphaFold2BinderInput(logits=logits, temperature=1.0, **target_kwargs),
        AlphaFold2BinderConfig(num_recycles=1, loss_weights={"plddt": 1.0}),
    )
    r_con = run_alphafold2_binder(
        AlphaFold2BinderInput(logits=logits, temperature=1.0, **target_kwargs),
        AlphaFold2BinderConfig(num_recycles=1, loss_weights={"con": 1.0}),
    )
    assert r_plddt.gradient != r_con.gradient
    assert all(math.isfinite(v) for row in r_plddt.gradient for v in row)
    assert all(math.isfinite(v) for row in r_con.gradient for v in row)


@pytest.mark.uses_gpu
def test_homooligomer():
    output = run_alphafold2(
        AlphaFold2Input(complexes=[StructurePredictionComplex(chains=[_HOMOOLIGOMER_SEQ, _HOMOOLIGOMER_SEQ])]),
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
    """Benchmark alphafold2-prediction on the trp heterodimer (cold + warm).

    Two-chain protein complex (~500 residues total) folded without MSA. Cold
    pass measures weight load + first inference; warm pass measures inference
    only.
    """
    complex_ = load_benchmark_complex("trp_heterodimer")
    inputs = AlphaFold2Input(complexes=[complex_])
    config = AlphaFold2Config(use_msa=False, verbose=True)

    result = benchmark_twice(request, "alphafold2", lambda: run_alphafold2(inputs=inputs, config=config))

    assert result.success, "AlphaFold2 benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
