"""Tests for AlphaFold2 prediction and gradient tools."""

import math
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    AlphaFold2Config,
    AlphaFold2GradientConfig,
    AlphaFold2GradientInput,
    AlphaFold2Input,
    StructurePredictionComplex,
    run_alphafold2,
    run_alphafold2_gradient,
)
from proto_tools.utils.sequence import PROTEIN_AMINO_ACIDS
from proto_tools.utils.tool_instance import ToolInstance
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
    if request.config.getoption("--cpu"):
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
    AlphaFold2GradientInput(logits=[[0.0] * 20, [1.0] * 20], target_pdb="/t.pdb", temperature=0.75)
    with pytest.raises(ValidationError, match="20 columns"):
        AlphaFold2GradientInput(logits=[[0.0] * 19], target_pdb="/t.pdb", temperature=1.0)


def test_gradient_config_rejects_bad_loss_weights():
    with pytest.raises(ValidationError, match="non-negative"):
        AlphaFold2GradientConfig(loss_weights={"plddt": -0.1})
    with pytest.raises(ValidationError, match="Unknown loss_weights"):
        AlphaFold2GradientConfig(loss_weights={"pldd": 1.0})


def test_gradient_input_requires_target_pdb():
    with pytest.raises(ValidationError, match="target_pdb"):
        AlphaFold2GradientInput(logits=[[0.0] * 20], temperature=1.0)


# -- Dispatch contracts --------------------------------------------------------


def test_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(tool_name, payload, *, instance=None, config=None):
        captured.update(tool_name=tool_name, payload=payload)
        return {"gradient": [[0.1] * 20] * 2, "loss": 1.25, "metrics": {"avg_plddt": 0.8}, "vocab": _CANONICAL_VOCAB}

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    inputs = AlphaFold2GradientInput(
        logits=[[0.0] * 20, [1.0] * 20],
        temperature=0.8,
        target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
        target_chain="A",
        binder_chain="B",
        design_positions=[0, 1],
    )
    config = AlphaFold2GradientConfig(
        loss_weights={"plddt": 1.0},
        device="cpu",
    )
    result = run_alphafold2_gradient(inputs=inputs, config=config)

    validate_output(result)
    assert captured["tool_name"] == "alphafold2"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert payload["temperature"] == 0.8
    assert payload["target_pdb"] == str(_GRADIENT_EXAMPLE_PDB_PATH)
    assert payload["backend"] == "base"
    assert payload["soft"] == 1.0
    assert result.gradient == [[0.1] * 20] * 2
    assert result.loss == 1.25


def test_prediction_dispatch_contract(monkeypatch):
    captured: list[dict] = []

    def fake_dispatch(tool_name, payload, *, instance=None, config=None):
        captured.append(payload)
        return {"pdb": _EXAMPLE_PDB, "avg_plddt": 0.85, "ptm": 0.72, "iptm": None, "avg_pae": 1.5}

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.alphafold2.alphafold2.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_alphafold2(AlphaFold2Input(complexes=["MKTL"]), AlphaFold2Config(use_msa=False, device="cpu"))

    assert captured[0]["operation"] == "predict"
    assert result.success
    assert is_valid_structure(result.structures[0].structure_cif)


# -- GPU integration tests -----------------------------------------------------


@pytest.mark.uses_gpu
def test_gradient_end_to_end():
    result = run_alphafold2_gradient(
        AlphaFold2GradientInput(
            logits=[[0.0] * 20] * 5,
            temperature=1.0,
            target_pdb=str(_GRADIENT_EXAMPLE_PDB_PATH),
            target_chain="A",
            binder_chain="B",
        ),
        AlphaFold2GradientConfig(
            num_recycles=1,
            loss_weights={"plddt": 1.0},
        ),
    )
    validate_output(result)
    assert result.tool_id == "alphafold2-gradient"
    assert len(result.gradient) == 5
    assert all(len(row) == 20 for row in result.gradient)
    assert all(math.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert math.isfinite(result.loss) and result.loss != 0.0
    assert result.vocab == _CANONICAL_VOCAB
    assert 0 <= result.metrics["avg_plddt"] <= 1.0


@pytest.mark.uses_gpu
def test_gradient_sensitivity():
    target_kwargs = {
        "target_pdb": str(_GRADIENT_EXAMPLE_PDB_PATH),
        "target_chain": "A",
        "binder_chain": "B",
    }
    config = AlphaFold2GradientConfig(num_recycles=1, loss_weights={"plddt": 1.0})
    r1 = run_alphafold2_gradient(
        AlphaFold2GradientInput(logits=[[0.0] * 20] * 5, temperature=1.0, **target_kwargs), config
    )
    r2 = run_alphafold2_gradient(
        AlphaFold2GradientInput(logits=[[5.0] + [0.0] * 19] * 5, temperature=1.0, **target_kwargs), config
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
    r_plddt = run_alphafold2_gradient(
        AlphaFold2GradientInput(logits=logits, temperature=1.0, **target_kwargs),
        AlphaFold2GradientConfig(num_recycles=1, loss_weights={"plddt": 1.0}),
    )
    r_con = run_alphafold2_gradient(
        AlphaFold2GradientInput(logits=logits, temperature=1.0, **target_kwargs),
        AlphaFold2GradientConfig(num_recycles=1, loss_weights={"con": 1.0}),
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
