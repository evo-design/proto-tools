"""tests/inverse_folding_tests/test_esm_if1.py.

Tests for ESM-IF1/ProteinDPO sampling and scoring.
"""

from pathlib import Path

import numpy as np
import pytest

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.esm_if1 import (
    ESMIF1SampleConfig,
    ESMIF1ScoringConfig,
    ESMIF1ScoringInput,
    run_esm_if1_sample,
    run_esm_if1_score,
)
from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingInput,
    InverseFoldingScoringMetrics,
    InverseFoldingStructureInput,
    SequenceStructurePair,
)
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"

_persistent_tool = make_persistent_fixture("esm_if1")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure.from_file(TEST_PDB_FILE)


# ============================================================================
# Sampling Tests
# ============================================================================
@pytest.mark.uses_gpu
def test_esm_if1_sample_simple(pdb_structure: Structure):
    """Basic sampling with default ProteinDPO config."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=1,
        temperature=0.1,
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    validate_output(output)
    assert output.tool_id == "esm-if1-sample"

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 1
    assert isinstance(designs.sequences[0], str)
    assert len(designs.sequences[0]) > 0
    assert len(designs.log_likelihoods) == 1
    assert isinstance(designs.log_likelihoods[0], float)
    assert np.isfinite(designs.log_likelihoods[0])


@pytest.mark.uses_gpu
def test_esm_if1_sample_multiple(pdb_structure: Structure):
    """Sampling multiple sequences per structure."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=3,
        temperature=0.1,
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 3
    assert all(isinstance(seq, str) for seq in designs.sequences)
    assert all(len(seq) > 0 for seq in designs.sequences)
    assert len(designs.log_likelihoods) == 3
    assert all(isinstance(ll, float) for ll in designs.log_likelihoods)
    assert all(np.isfinite(ll) for ll in designs.log_likelihoods)


@pytest.mark.uses_gpu
def test_esm_if1_sample_chunked_batching(pdb_structure: Structure):
    """Chunked batching produces the correct number of sequences."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=4,
        batch_size=2,
        temperature=0.1,
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 4
    assert all(isinstance(seq, str) for seq in designs.sequences)
    assert all(len(seq) > 0 for seq in designs.sequences)
    assert len(designs.log_likelihoods) == 4


@pytest.mark.uses_gpu
def test_esm_if1_sample_fixed_positions(pdb_structure: Structure):
    """Fixed positions in sampled sequences match the native residues."""
    native_seq = pdb_structure.get_chain_sequence("A")
    # Fix positions 1, 5, 10 (1-indexed) to native residues
    fixed_pos = [1, 5, 10]

    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=pdb_structure,
                chains_to_redesign=["A"],
                fixed_positions={"A": fixed_pos},
            )
        ]
    )
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=2,
        temperature=0.5,  # higher temp to ensure non-fixed positions vary
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Fixed positions sampling failed: {output}"

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 2
    for seq in designs.sequences:
        for pos in fixed_pos:
            assert seq[pos - 1] == native_seq[pos - 1], (
                f"Position {pos}: expected '{native_seq[pos - 1]}', got '{seq[pos - 1]}'"
            )


@pytest.mark.uses_gpu
def test_esm_if1_sample_dpo_weights(pdb_structure: Structure):
    """Sampling with explicit ProteinDPO weights."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=1,
        temperature=0.1,
        weights_variant="protein_dpo",
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Failed to sample with DPO weights: {output}"

    designs = output.designed_sequences[0]
    assert len(designs.sequences) == 1
    assert isinstance(designs.sequences[0], str)
    assert len(designs.sequences[0]) > 0


# ============================================================================
# Scoring Tests
# ============================================================================
@pytest.mark.uses_gpu
def test_esm_if1_score(pdb_structure: Structure):
    """Score a sequence against a structure."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ESMIF1ScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
        ]
    )
    config = ESMIF1ScoringConfig()
    output = run_esm_if1_score(inp, config)
    assert output.success, f"Failed to score: {output}"

    validate_output(output)
    assert output.tool_id == "esm-if1-score"
    assert len(output.scores) == 1
    assert isinstance(output.scores[0], InverseFoldingScoringMetrics)


@pytest.mark.uses_gpu
def test_esm_if1_score_fields(pdb_structure: Structure):
    """Scoring fields and mathematical relationships are correct."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ESMIF1ScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
        ]
    )
    config = ESMIF1ScoringConfig()
    output = run_esm_if1_score(inp, config)
    assert output.success
    assert_metrics_in_spec(output)

    score = output.scores[0]

    # Value range checks
    assert score.avg_log_likelihood <= 0
    assert score.perplexity >= 1.0

    # Mathematical relationship: perplexity = exp(-avg_log_likelihood)
    assert np.isclose(
        score.perplexity,
        np.exp(-score.avg_log_likelihood),
        rtol=1e-5,
    )


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("esm-if1-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm_if1_sample_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark esm-if1-sample: 50 ProteinDPO designs of renin (~340 aa) at batch_size=16 (cold + warm)."""
    inputs = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=50,
        batch_size=16,
        temperature=0.1,
        seed=0,
    )

    result = benchmark_twice(request, "esm_if1", lambda: run_esm_if1_sample(inputs, config))

    assert result.tool_id == "esm-if1-sample"
    assert len(result.designed_sequences) == 1
    designs = result.designed_sequences[0]
    assert len(designs.sequences) == 50
    target_len = len(pdb_structure.get_chain_sequence("A"))
    for seq in designs.sequences:
        assert len(seq) == target_len
    assert len(designs.log_likelihoods) == 50
    assert all(np.isfinite(ll) for ll in designs.log_likelihoods)


@pytest.mark.benchmark("esm-if1-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm_if1_score_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark esm-if1-score on 50 sequence-structure pairs against renin (cold + warm)."""
    target_len = len(pdb_structure.get_chain_sequence("A"))
    sequences = random_protein_sequences(n=50, length=target_len, seed=1)
    pairs = [SequenceStructurePair(sequence=s, structure=pdb_structure) for s in sequences]

    inputs = ESMIF1ScoringInput(sequence_structure_pairs=pairs)
    config = ESMIF1ScoringConfig()

    result = benchmark_twice(request, "esm_if1", lambda: run_esm_if1_score(inputs, config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "esm-if1-score"
    assert len(result.scores) == 50
    for score in result.scores:
        assert score["perplexity"] >= 1.0
