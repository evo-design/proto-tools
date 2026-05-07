"""tests/inverse_folding_tests/test_ligandmpnn.py.

Tests for LigandMPNN sampling and scoring.
"""

import json
import math
from pathlib import Path

import pytest

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
    example_input as ligandmpnn_example_input,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
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

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 2
    assert all(isinstance(sequence, str) for sequence in designs.sequences)
    assert all(len(seq) > 0 for seq in designs.sequences)
    assert len(designs.sequence_recovery) == 2
    assert designs.ligand_interface_sequence_recovery is not None
    assert len(designs.ligand_interface_sequence_recovery) == 2
    assert all(0.0 <= r <= 1.0 for r in designs.sequence_recovery)
    assert all(0.0 <= r <= 1.0 for r in designs.ligand_interface_sequence_recovery)


@pytest.mark.uses_gpu
def test_ligandmpnn_sample_no_ligand(cif_structure: Structure):
    """#704: mixed batch maps no-ligand to None per-instance and round-trips strict JSON."""
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=cif_structure, chains_to_redesign=["A"]),
            ligandmpnn_example_input().inputs[0],  # example_input_fixture.pdb has zero HETATM
        ]
    )
    config = LigandMPNNSampleConfig(num_sequences_per_structure=2, seed=0)
    output = run_ligandmpnn_sample(inp, config)
    assert output.success

    with_ligand, without_ligand = output.designed_sequences
    assert with_ligand.ligand_interface_sequence_recovery is not None
    assert all(0.0 <= r <= 1.0 for r in with_ligand.ligand_interface_sequence_recovery)
    assert without_ligand.ligand_interface_sequence_recovery is None
    assert all(math.isfinite(r) for d in (with_ligand, without_ligand) for r in d.sequence_recovery)
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

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 6
    assert all(isinstance(seq, str) for seq in designs.sequences)
    assert all(len(seq) > 0 for seq in designs.sequences)
    assert len(designs.sequence_recovery) == 6
    assert len(designs.ligand_interface_sequence_recovery) == 6


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

    assert len(output.designed_sequences) == 2
    for designs in output.designed_sequences:
        assert len(designs.sequences) == 3
        assert all(isinstance(sequence, str) for sequence in designs.sequences)


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
            sequence_structure_pairs=[SequenceStructurePair(sequence=sequence, structure=structure)]
        ),
        LigandMPNNScoringConfig(
            fixed_positions={"A": [1]},
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


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("ligandmpnn-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_ligandmpnn_sample_benchmark(request: pytest.FixtureRequest, cif_structure: Structure) -> None:
    """Benchmark ligandmpnn-sample: 50 designs of renin chain A (~217 aa) at batch_size=16 (cold + warm)."""
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
    assert len(result.designed_sequences) == 1, "Should have one DesignedSequences per input structure"
    designs = result.designed_sequences[0]
    assert len(designs.sequences) == 50, "Should have 50 designed sequences"
    lengths = {len(seq) for seq in designs.sequences}
    assert len(lengths) == 1, f"All designed sequences should have the same length, got {lengths}"
    assert next(iter(lengths)) > 0, "Designed sequences should be non-empty"
