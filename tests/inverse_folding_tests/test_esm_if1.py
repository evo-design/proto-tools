"""tests/inverse_folding_tests/test_esm_if1.py

Tests for ESM-IF1/ProteinDPO sampling and scoring."""

from pathlib import Path

import numpy as np
import pytest

from bio_programming_tools.entities.structures.structure import Structure
from bio_programming_tools.tools.inverse_folding.esm_if1 import (
    ESMIF1SampleConfig,
    ESMIF1ScoringConfig,
    ESMIF1ScoringInput,
    run_esm_if1_sample,
    run_esm_if1_score,
)
from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
    SequenceScores,
    SequenceStructurePair,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"

_persistent_tool = make_persistent_fixture("esm_if1")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure(structure_filepath_or_content=TEST_PDB_FILE)


# ============================================================================
# Sampling Tests
# ============================================================================
@pytest.mark.include_in_env_report(category="inverse_folding")
@pytest.mark.uses_gpu
def test_esm_if1_sample_simple(pdb_structure: Structure):
    """Basic sampling with default ProteinDPO config."""
    inp = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=pdb_structure)]
    )
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=1, temperature=0.1, seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    validate_output(output)
    assert output.tool_id == "esm-if1-sample"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 1
    assert isinstance(designed.sequences[0], str)
    assert len(designed.sequences[0]) > 0
    assert len(designed.log_likelihoods) == 1
    assert isinstance(designed.log_likelihoods[0], float)
    assert np.isfinite(designed.log_likelihoods[0])


@pytest.mark.uses_gpu
def test_esm_if1_sample_multiple(pdb_structure: Structure):
    """Sampling multiple sequences per structure."""
    inp = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=pdb_structure)]
    )
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=3, temperature=0.1, seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 3
    assert all(isinstance(seq, str) for seq in designed.sequences)
    assert all(len(seq) > 0 for seq in designed.sequences)
    assert len(designed.log_likelihoods) == 3
    assert all(isinstance(ll, float) for ll in designed.log_likelihoods)
    assert all(np.isfinite(ll) for ll in designed.log_likelihoods)


@pytest.mark.uses_gpu
def test_esm_if1_sample_chunked_batching(pdb_structure: Structure):
    """Chunked batching produces the correct number of sequences."""
    inp = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=pdb_structure)]
    )
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=4,
        batch_size=2,
        temperature=0.1,
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 4
    assert all(isinstance(seq, str) for seq in designed.sequences)
    assert all(len(seq) > 0 for seq in designed.sequences)
    assert len(designed.log_likelihoods) == 4


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
                chain_ids=["A"],
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

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 2
    for seq in designed.sequences:
        for pos in fixed_pos:
            assert seq[pos - 1] == native_seq[pos - 1], (
                f"Position {pos}: expected '{native_seq[pos - 1]}', "
                f"got '{seq[pos - 1]}'"
            )


@pytest.mark.uses_gpu
def test_esm_if1_sample_dpo_weights(pdb_structure: Structure):
    """Sampling with explicit ProteinDPO weights."""
    inp = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=pdb_structure)]
    )
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=1,
        temperature=0.1,
        weights_variant="protein_dpo",
        seed=42,
    )
    output = run_esm_if1_sample(inp, config)
    assert output.success, f"Failed to sample with DPO weights: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 1
    assert isinstance(designed.sequences[0], str)
    assert len(designed.sequences[0]) > 0


# ============================================================================
# Scoring Tests
# ============================================================================
@pytest.mark.uses_gpu
def test_esm_if1_score(pdb_structure: Structure):
    """Score a sequence against a structure."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ESMIF1ScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ESMIF1ScoringConfig()
    output = run_esm_if1_score(inp, config)
    assert output.success, f"Failed to score: {output}"

    validate_output(output)
    assert output.tool_id == "esm-if1-score"
    assert len(output.scores) == 1
    assert isinstance(output.scores[0], SequenceScores)


@pytest.mark.uses_gpu
def test_esm_if1_score_fields(pdb_structure: Structure):
    """Scoring fields and mathematical relationships are correct."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ESMIF1ScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ESMIF1ScoringConfig()
    output = run_esm_if1_score(inp, config)
    assert output.success

    score = output.scores[0]

    # Type checks
    assert isinstance(score.avg_log_likelihood, float)
    assert isinstance(score.perplexity, float)

    # Value range checks
    assert score.avg_log_likelihood <= 0
    assert score.perplexity >= 1.0

    # Mathematical relationship: perplexity = exp(-avg_log_likelihood)
    assert np.isclose(
        score.perplexity,
        np.exp(-score.avg_log_likelihood),
        rtol=1e-5,
    )
