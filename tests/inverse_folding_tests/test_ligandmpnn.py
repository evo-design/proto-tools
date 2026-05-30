"""tests/inverse_folding_tests/test_ligandmpnn.py.

Tests for LigandMPNN sampling and scoring.
"""

import json
import math
from pathlib import Path

import pytest

from proto_tools.entities.complex import Chain
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingScoringMetrics,
    InverseFoldingStructureInput,
    LigandMPNNSampleConfig,
    LigandMPNNScoringConfig,
    LigandMPNNScoringInput,
    SequenceStructurePair,
    run_ligandmpnn_sample,
    run_ligandmpnn_score,
)
from proto_tools.tools.inverse_folding.ligandmpnn.ligandmpnn_sample import (
    LigandMPNNDesign,
    LigandMPNNDesignMetrics,
)
from proto_tools.tools.inverse_folding.ligandmpnn.ligandmpnn_sample import (
    example_input as ligandmpnn_example_input,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_CIF_FILE = Path(__file__).parent.parent / "dummy_data" / "renin.cif"
DEFAULT_CHECKPOINT = Path.home() / ".foundry" / "checkpoints" / "ligandmpnn_v_32_010_25.pt"


_persistent_tool = make_persistent_fixture("ligandmpnn")


@pytest.fixture(scope="module")
def cif_structure():
    return Structure.from_file(TEST_CIF_FILE)


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_simple(cif_structure: Structure):
    """Basic LigandMPNN sampling with a single structure."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"])])
    config = LigandMPNNSampleConfig(num_sequences_per_structure=2, temperature=0.1, seed=42)

    output = run_ligandmpnn_sample(inp, config)

    validate_output(output)

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 2
    for design in design_set.complexes:
        # LigandMPNN returns every chain (designed + fixed) per design.
        assert len(design.chains) > 0
        assert all(isinstance(chain.sequence, str) for chain in design.chains)
        assert all(len(chain.sequence) > 0 for chain in design.chains)
        recovery = design.metrics["sequence_recovery"]
        assert 0.0 <= recovery <= 1.0
        interface_recovery = design.metrics["ligand_interface_sequence_recovery"]
        assert 0.0 <= interface_recovery <= 1.0
    assert_metrics_in_spec(output)


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_no_ligand(cif_structure: Structure):
    """Mixed batch maps no-ligand to None per-instance and round-trips strict JSON."""
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"]),
            ligandmpnn_example_input().inputs[0],  # example_input_fixture.pdb has zero HETATM
        ]
    )
    config = LigandMPNNSampleConfig(num_sequences_per_structure=2, seed=0)
    output = run_ligandmpnn_sample(inp, config)
    assert output.success

    with_ligand, without_ligand = output.design_sets
    for design in with_ligand.complexes:
        interface_recovery = design.metrics["ligand_interface_sequence_recovery"]
        assert math.isfinite(interface_recovery)
        assert 0.0 <= interface_recovery <= 1.0
    for design in without_ligand.complexes:
        # No ligand interface: metric is NaN or absent.
        assert "ligand_interface_sequence_recovery" not in design.metrics or math.isnan(
            design.metrics["ligand_interface_sequence_recovery"]
        )
    assert all(
        math.isfinite(design.metrics["sequence_recovery"])
        for design_set in (with_ligand, without_ligand)
        for design in design_set.complexes
    )
    json.dumps(output.model_dump(mode="json"), allow_nan=False)


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_chunked_batching(cif_structure: Structure):
    """Chunked batching produces the correct number of sequences."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"])])
    config = LigandMPNNSampleConfig(
        num_sequences_per_structure=6,
        batch_size=2,
        temperature=0.1,
        seed=42,
    )
    output = run_ligandmpnn_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 6
    for design in design_set.complexes:
        assert all(isinstance(chain.sequence, str) for chain in design.chains)
        assert all(len(chain.sequence) > 0 for chain in design.chains)
        assert math.isfinite(design.metrics["sequence_recovery"])
        assert "ligand_interface_sequence_recovery" in design.metrics


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_multiple_structures(cif_structure: Structure):
    """Sampling with multiple input structures."""
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"]),
            InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"]),
        ]
    )
    config = LigandMPNNSampleConfig(num_sequences_per_structure=3, temperature=0.1, seed=42)

    output = run_ligandmpnn_sample(inp, config)

    validate_output(output)

    assert len(output.design_sets) == 2
    for design_set in output.design_sets:
        assert len(design_set.complexes) == 3
        for design in design_set.complexes:
            assert all(isinstance(chain.sequence, str) for chain in design.chains)


def test_ligandmpnn_score_dispatch_contract(monkeypatch):
    captured = {}
    structure = ligandmpnn_example_input().inputs[0].structure
    sequence = structure.get_chain_sequence("A")
    vocab = list("ACDEFGHIKLMNPQRSTVWYX")

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {
            "metrics": {"log_likelihood": -10.0, "avg_log_likelihood": -1.0, "perplexity": math.e},
            "logits": [[0.0] * len(vocab)] * len(sequence),
            "vocab": vocab,
        }

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.ligandmpnn.ligandmpnn_score.ToolInstance.dispatch",
        fake_dispatch,
    )

    output = run_ligandmpnn_score(
        LigandMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=sequence,
                    structure=structure,
                    fixed_positions={"A": [1]},
                )
            ]
        ),
        LigandMPNNScoringConfig(
            seed=42,
            device="cpu",
            return_logits=True,
            scoring_mode="autoregressive",
        ),
    )

    assert output.tool_id == "ligandmpnn-score"
    assert isinstance(output.scores[0], InverseFoldingScoringMetrics)
    assert output.scores[0].logits is not None
    assert captured["toolkit"] == "ligandmpnn"
    payload = captured["payload"]
    assert payload["operation"] == "score"
    assert payload["sequence"] == sequence
    assert payload["fixed_positions"] == {"A": [1]}
    assert payload["return_logits"] is True
    assert payload["model_type"] == "ligand_mpnn"
    assert payload["scoring_mode"] == "autoregressive"


@pytest.mark.uses_gpu
def test_ligandmpnn_score():
    structure = ligandmpnn_example_input().inputs[0].structure
    sequence = structure.get_chain_sequence("A")
    inputs = LigandMPNNScoringInput(
        sequence_structure_pairs=[SequenceStructurePair(sequence=sequence, structure=structure)]
    )
    output = run_ligandmpnn_score(inputs, LigandMPNNScoringConfig(seed=42, return_logits=True))

    validate_output(output)
    assert output.tool_id == "ligandmpnn-score"
    assert len(output.scores) == 1
    score = output.scores[0]
    assert score.logits is not None
    assert len(score.logits) == len(sequence)
    assert len(score.logits[0]) == len(score.vocab)
    assert math.isclose(score.log_likelihood, score.avg_log_likelihood * len(sequence), rel_tol=1e-5)
    assert math.isclose(score.perplexity, math.exp(-score.avg_log_likelihood), rel_tol=1e-5)


def test_ligandmpnn_multichain_design_structure():
    """A LigandMPNNDesign preserves chain id, order, redesigned flag, and metrics."""
    design = LigandMPNNDesign(
        chains=[
            Chain(id="A", sequence="MKTL"),
            Chain(id="B", sequence="GGSG"),
        ],
        designed=[True, False],
        metrics=LigandMPNNDesignMetrics(
            sequence_recovery=0.5,
            ligand_interface_sequence_recovery=0.3,
        ),
    )

    assert [chain.id for chain in design.chains] == ["A", "B"]
    assert design.designed == [True, False]
    assert design.chain_sequences == ["MKTL", "GGSG"]
    assert design.chain_sequence_map() == {"A": "MKTL", "B": "GGSG"}
    assert [chain.id for chain in design.designed_chains] == ["A"]
    assert design.metrics["sequence_recovery"] == 0.5
    assert design.metrics["ligand_interface_sequence_recovery"] == 0.3
    assert design.design_metrics()["sequence_recovery"] == 0.5


def test_ligandmpnn_design_feeds_structure_predictor():
    """A LigandMPNNDesign feeds an SP tool input directly via LSP (subclass-as-Complex)."""
    from proto_tools.tools.structure_prediction import ESMFoldInput

    design = LigandMPNNDesign(
        chains=[
            Chain(id="A", sequence="MKTLVLSP"),
            Chain(id="B", sequence="GGSGGSGG"),
        ],
        designed=[True, False],
        metrics=LigandMPNNDesignMetrics(sequence_recovery=0.7),
    )

    inp = ESMFoldInput(complexes=[design])
    assert inp.complexes[0] is design
    assert [c.sequence for c in inp.complexes[0].chains] == design.chain_sequences


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("ligandmpnn-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_ligandmpnn_sample_benchmark(request: pytest.FixtureRequest, cif_structure: Structure) -> None:
    """Benchmark ligandmpnn-sample: 50 complexes of renin chain A (~217 aa) at batch_size=16 (cold + warm)."""
    inputs = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"])]
    )
    config = LigandMPNNSampleConfig(
        num_sequences_per_structure=50,
        batch_size=16,
        temperature=0.1,
        seed=0,
    )

    result = benchmark_twice(request, "ligandmpnn", lambda: run_ligandmpnn_sample(inputs, config))

    assert result.tool_id == "ligandmpnn-sample"
    assert len(result.design_sets) == 1, "Should have one design set per input structure"
    design_set = result.design_sets[0]
    assert len(design_set.complexes) == 50, "Should have 50 complexes"
    lengths = {sum(len(s) for s in design.chain_sequences) for design in design_set.complexes}
    assert len(lengths) == 1, f"All complexes should have the same length, got {lengths}"
    assert next(iter(lengths)) > 0, "Designs should be non-empty"
