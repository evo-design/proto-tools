"""tests/structure_prediction_tests/test_esmfold.py.

Benchmark tests for ESMFold structure prediction.

Cross-tool integration coverage lives in ``test_structure_prediction.py``;
this file holds the cold/warm benchmark and any ESMFold-specific tests.
"""

from unittest.mock import patch

import pytest

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Complex,
    ESMFoldConfig,
    ESMFoldGradientConfig,
    ESMFoldGradientInput,
    ESMFoldInput,
    run_esmfold,
    run_esmfold_gradient,
)
from proto_tools.utils import one_hot_protein_logits
from tests.conftest import benchmark_twice, random_protein_sequences


def _minimal_pdb(num_residues: int) -> str:
    """Build a simple single-chain PDB with ``num_residues`` glycine residues."""
    lines = []
    atom_id = 1
    for residue_id in range(1, num_residues + 1):
        base = float(residue_id * 3)
        for atom_name, dx, dy in [("N", 0.0, 0.0), ("CA", 1.4, 0.0), ("C", 2.1, 1.2), ("O", 1.7, 2.3)]:
            lines.append(
                f"ATOM  {atom_id:5d} {atom_name:<4} GLY A{residue_id:4d}    "
                f"{base + dx:8.3f}{dy:8.3f}{0.0:8.3f}{1.0:6.2f}{0.8:6.2f}           {atom_name[0]:>2}"
            )
            atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def test_esmfold_predict_retries_on_cuda_oom():
    """On CUDA OOM the wrapper halves max_batch_residues and re-splits the sub-batch."""
    seq_a, seq_b = "A" * 100, "C" * 100  # each 100 residues; default split would put both in one batch
    inputs = ESMFoldInput(
        complexes=[
            Complex(chains=[{"sequence": seq_a, "entity_type": "protein"}]),
            Complex(chains=[{"sequence": seq_b, "entity_type": "protein"}]),
        ]
    )
    config = ESMFoldConfig(max_batch_residues=200, num_recycles=1)
    one_result = {"pdb": _minimal_pdb(4), "avg_plddt": 0.8, "ptm": 0.5, "avg_pae": 6.0}

    calls: list[int] = []

    def fake_dispatch(_name, payload, **_kwargs):
        n = len(payload["batch_data"])
        calls.append(n)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return {"results": [one_result] * n}

    with patch(
        "proto_tools.tools.structure_prediction.esmfold.esmfold.ToolInstance.dispatch", side_effect=fake_dispatch
    ):
        output = run_esmfold(inputs=inputs, config=config)

    # First call: original batch of 2 OOMs. Cap halves 200→100, splitter yields two batches of 1.
    assert calls == [2, 1, 1]
    assert len(output.structures) == 2


def test_esmfold_predict_does_not_duplicate_results_on_mid_attempt_oom():
    """Mid-iteration OOM discards earlier chunks' results so retry doesn't double-count."""
    inputs = ESMFoldInput(
        complexes=[Complex(chains=[{"sequence": ch * 50, "entity_type": "protein"}]) for ch in ("A", "C", "D", "E")]
    )
    config = ESMFoldConfig(max_batch_residues=200, num_recycles=1)
    one_result = {"pdb": _minimal_pdb(4), "avg_plddt": 0.8, "ptm": 0.5, "avg_pae": 6.0}

    call_idx = 0

    def fake_dispatch(_name, payload, **_kwargs):
        nonlocal call_idx
        call_idx += 1
        # OOM full batch (call 1), then chunk 2 of the halved attempt (call 3); cap=50 succeeds.
        if call_idx in (1, 3):
            raise RuntimeError("CUDA out of memory")
        return {"results": [one_result] * len(payload["batch_data"])}

    with patch(
        "proto_tools.tools.structure_prediction.esmfold.esmfold.ToolInstance.dispatch", side_effect=fake_dispatch
    ):
        output = run_esmfold(inputs=inputs, config=config)

    assert len(output.structures) == 4


def test_esmfold_predict_propagates_non_oom_errors():
    """Non-OOM RuntimeError surfaces immediately without halving."""
    inputs = ESMFoldInput(complexes=[Complex(chains=[{"sequence": "A" * 50, "entity_type": "protein"}])])
    with patch(
        "proto_tools.tools.structure_prediction.esmfold.esmfold.ToolInstance.dispatch",
        side_effect=RuntimeError("unrelated worker failure"),
    ):
        with pytest.raises(RuntimeError, match=r"unrelated worker failure"):
            run_esmfold(inputs=inputs, config=ESMFoldConfig(num_recycles=1))


def test_esmfold_gradient_input_validation():
    """Gradient input validates target-chain/logit shape before dispatch."""
    logits = one_hot_protein_logits("MKTL", sharpness=2.0)
    ok = ESMFoldGradientInput(logits=logits, chains=["AAAA", "MKTL"], target_chain_indices=[1])
    assert ok.target_chain_indices == [1]

    with pytest.raises(ValueError, match="must have length"):
        ESMFoldGradientInput(logits=logits, chains=["MKT"], target_chain_indices=[0])


def test_esmfold_gradient_dispatch_builds_output():
    """Wrapper sends the gradient payload to the standalone worker and wraps the structure."""
    dispatch_result = {
        "gradient": [[0.1] * 20 for _ in range(4)],
        "loss": 1.5,
        "metrics": {
            "avg_plddt": 0.8,
            "ptm": 0.4,
            "avg_pae": 6.0,
            "loss_plddt": 0.2,
            "loss_ptm": 0.6,
        },
        "vocab": list("ACDEFGHIKLMNPQRSTVWY"),
        "pdb": _minimal_pdb(4),
    }

    with patch("proto_tools.tools.structure_prediction.esmfold.esmfold.ToolInstance.dispatch") as mock_dispatch:
        mock_dispatch.return_value = dispatch_result
        output = run_esmfold_gradient(
            ESMFoldGradientInput(logits=one_hot_protein_logits("MKTL", sharpness=2.0), chains=["MKTL"]),
            ESMFoldGradientConfig(num_recycles=1, loss_weights={"plddt": 2.0, "ptm": 0.5}),
        )

    payload = mock_dispatch.call_args.args[1]
    assert payload["operation"] == "compute_gradient"
    assert payload["loss_weights"] == {"plddt": 2.0, "ptm": 0.5}
    assert payload["target_chain_indices"] == [0]
    assert output.gradient is not None and len(output.gradient) == 4
    assert output.loss == pytest.approx(1.5)
    assert output.structure.metrics["avg_plddt"] == pytest.approx(0.8)


@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold_gradient_smoke():
    """Real ESMFold gradient returns finite logits-gradient on a tiny peptide."""
    output = run_esmfold_gradient(
        ESMFoldGradientInput(logits=one_hot_protein_logits("MKTL", sharpness=2.0), chains=["MKTL"]),
        ESMFoldGradientConfig(num_recycles=1, loss_weights={"plddt": 1.0}),
    )

    assert output.gradient is not None
    assert len(output.gradient) == 4
    assert all(len(row) == 20 for row in output.gradient)
    assert output.loss == pytest.approx(output.metrics["loss_plddt"])


@pytest.mark.benchmark("esmfold-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmfold_benchmark(request):
    """Benchmark esmfold-prediction on 10 random 300-residue proteins (cold + warm).

    Mirrors a realistic batched ESMFold screen of ~10 designs at typical
    single-domain length. ``max_batch_residues=4096`` lets the entire 3000-residue
    workload pack into a single GPU forward pass on a modern accelerator —
    exercising the high-throughput batched path rather than the default ``1200``
    which would split into three sub-batches.
    """
    sequences = random_protein_sequences(n=10, length=300, seed=0)
    complexes = [Complex(chains=[seq]) for seq in sequences]
    inputs = ESMFoldInput(complexes=complexes)
    config = ESMFoldConfig(max_batch_residues=4096)

    result = benchmark_twice(request, "esmfold", lambda: run_esmfold(inputs=inputs, config=config))

    assert result.success, "ESMFold benchmark run failed"
    assert len(result.structures) == 10
    for structure in result.structures:
        assert is_valid_structure(structure.structure_cif)
