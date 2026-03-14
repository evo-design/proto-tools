"""Tests for ProteinMPNN sampling and scoring."""

import random
from pathlib import Path

import numpy as np
import pytest

from bio_programming_tools.entities.structures.structure import Structure
from bio_programming_tools.tools.inverse_folding.proteinmpnn import (
    ProteinMPNNSampleConfig,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    run_proteinmpnn_sample,
    run_proteinmpnn_score,
)
from bio_programming_tools.tools.inverse_folding.proteinmpnn.standalone.inference import (
    ALPHAFOLD_VOCAB,
)
from bio_programming_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingStructureInput,
    SequenceScores,
    SequenceStructurePair,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


_persistent_tool = make_persistent_fixture("proteinmpnn")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure(structure_filepath_or_content=TEST_PDB_FILE)


@pytest.mark.include_in_env_report(category="inverse_folding")
@pytest.mark.uses_gpu
def test_proteinmpnn_sample_simple(pdb_structure: Structure):
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=pdb_structure),
            InverseFoldingStructureInput(structure=pdb_structure),
            InverseFoldingStructureInput(structure=pdb_structure),
        ]
    )
    config = InverseFoldingConfig(
        num_sequences_per_structure=10, temperature=1.0, seed=42,
    )
    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    validate_output(output)
    assert output.tool_id == "proteinmpnn-sample"

    for designed_sequences in output.designed_sequences:
        assert len(designed_sequences) == 10
        assert all(
            isinstance(sequence, str) for sequence in designed_sequences.sequences
        )
        assert all(
            isinstance(perplexity, float)
            for perplexity in designed_sequences.perplexity
        )
        assert all(
            isinstance(identity, float)
            for identity in designed_sequences.sequence_identity
        )


@pytest.mark.uses_gpu
def test_proteinmpnn_sample_chunked_batching(pdb_structure: Structure):
    """Chunked batching produces the correct number of sequences."""
    inp = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=pdb_structure)]
    )
    config = InverseFoldingConfig(
        num_sequences_per_structure=6,
        batch_size=2,
        temperature=0.1,
        seed=42,
    )
    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 6
    assert all(isinstance(seq, str) for seq in designed.sequences)
    assert all(len(seq) > 0 for seq in designed.sequences)
    assert len(designed.perplexity) == 6
    assert all(isinstance(p, float) for p in designed.perplexity)
    assert len(designed.sequence_identity) == 6
    assert all(isinstance(s, float) for s in designed.sequence_identity)


@pytest.mark.uses_gpu
def test_proteinmpnn_sample_advanced_args(pdb_structure: Structure):
    """Fixed positions and excluded amino acids are respected."""
    chain_A = pdb_structure.get_chain_sequence("A")

    # Find all indices of the amino acid "C" in chain A
    c_positions = [i + 1 for i, aa in enumerate(chain_A) if aa == "C"]

    # Make a list of fixed indices that do not contain the "C" positions
    fixed_positions = random.sample(
        list(set(np.arange(len(chain_A)) + 1) - set(c_positions)), 200
    )

    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=pdb_structure, fixed_positions={"A": fixed_positions}
            ),
            InverseFoldingStructureInput(
                structure=pdb_structure, fixed_positions={"A": fixed_positions}
            ),
        ]
    )
    config = InverseFoldingConfig(
        num_sequences_per_structure=10,
        temperature=1.0,
        seed=42,
        excluded_amino_acids=["C"],
    )

    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    validate_output(output)

    for designed_sequences in output.designed_sequences:
        for sequence in designed_sequences.sequences:
            assert "C" not in sequence, f"Sequence contains excluded 'C': {sequence}"

            for position in fixed_positions:
                assert (
                    sequence[position - 1] == chain_A[position - 1]
                ), f"Position {position}: {sequence[position-1]} != {chain_A[position-1]}"


@pytest.mark.uses_gpu
def test_proteinmpnn_score(pdb_structure: Structure):
    original_sequence = pdb_structure.get_chain_sequence("A")

    modified_sequence = list(original_sequence)
    for index in random.sample(range(len(modified_sequence)), 100):
        modified_sequence[index] = "C"
    modified_sequence = "".join(modified_sequence)

    fixed_positions = random.sample(list(range(len(original_sequence))), 100)
    fixed_positions = {
        "A": [position + 1 for position in fixed_positions],
    }

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
            SequenceStructurePair(
                sequence=modified_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(
        fixed_positions=fixed_positions, seed=42, return_logits=True
    )
    output = run_proteinmpnn_score(inp, config)
    assert output.success, f"Failed to score: {output}"

    validate_output(output)
    assert output.tool_id == "proteinmpnn-score"
    assert output.vocab == ALPHAFOLD_VOCAB

    assert len(output.scores) == 2
    assert all(isinstance(score, SequenceScores) for score in output.scores)

    # Original sequence should have lower perplexity than the modified one
    assert output.scores[0].perplexity < output.scores[1].perplexity


@pytest.mark.uses_gpu
def test_proteinmpnn_score_fields(pdb_structure: Structure):
    """All scoring fields and their mathematical relationships are correct."""
    original_sequence = pdb_structure.get_chain_sequence("A")
    seq_len = len(original_sequence)

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)
    assert output.success

    validate_output(output)

    score = output.scores[0]

    # Validate metrics via attribute access
    assert isinstance(score.log_likelihood, float)
    assert isinstance(score.avg_log_likelihood, float)
    assert isinstance(score.perplexity, float)

    # Validate metrics via dict access
    assert isinstance(score.metrics["log_likelihood"], float)
    assert isinstance(score.metrics["avg_log_likelihood"], float)
    assert isinstance(score.metrics["perplexity"], float)

    # Validate logits shape
    logits_arr = np.array(score.logits)
    assert logits_arr.ndim == 2
    assert logits_arr.shape == (seq_len, len(ALPHAFOLD_VOCAB))

    # log_likelihood = avg_log_likelihood * seq_len
    assert np.isclose(
        score.log_likelihood, score.avg_log_likelihood * seq_len, rtol=1e-5
    )

    # perplexity = exp(-avg_log_likelihood)
    assert np.isclose(
        score.perplexity, np.exp(-score.avg_log_likelihood), rtol=1e-5
    )

    assert score.avg_log_likelihood <= 0
    assert score.perplexity >= 1.0


@pytest.mark.uses_gpu
def test_proteinmpnn_score_vocab(pdb_structure: Structure):
    """Vocab property on output matches ALPHAFOLD_VOCAB."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42)
    output = run_proteinmpnn_score(inp, config)
    assert output.success

    validate_output(output)

    assert output.vocab == ALPHAFOLD_VOCAB
    assert output.scores[0].vocab == ALPHAFOLD_VOCAB


@pytest.mark.uses_gpu
def test_proteinmpnn_score_single_pair(pdb_structure: Structure):
    """Scoring with a single sequence-structure pair succeeds."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)

    assert output.success
    validate_output(output)
    assert len(output.scores) == 1
    assert output.scores[0].perplexity >= 1.0
    assert output.scores[0].logits is not None


@pytest.mark.uses_gpu
def test_proteinmpnn_score_batched(pdb_structure: Structure):
    """Batched scoring with multiple sequence-structure pairs."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    modified_1 = list(original_sequence)
    for i in random.sample(range(len(modified_1)), 30):
        modified_1[i] = "A"
    modified_1 = "".join(modified_1)
    modified_2 = list(original_sequence)
    for i in random.sample(range(len(modified_2)), 30):
        modified_2[i] = "G"
    modified_2 = "".join(modified_2)

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
            SequenceStructurePair(
                sequence=modified_1, structure=pdb_structure
            ),
            SequenceStructurePair(
                sequence=modified_2, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)

    assert output.success
    validate_output(output)
    assert len(output.scores) == 3
    for score in output.scores:
        assert score.perplexity >= 1.0
        assert score.logits is not None
        assert "log_likelihood" in score.metrics
        assert "avg_log_likelihood" in score.metrics
        assert "perplexity" in score.metrics


@pytest.mark.uses_gpu
def test_proteinmpnn_score_cache(pdb_structure: Structure):
    """Caching returns consistent scores across passes."""
    from bio_programming_tools.utils.tool_cache import (
        ToolCache,
        _program_tool_cache,
        get_cache_info,
    )

    original_sequence = pdb_structure.get_chain_sequence("A")

    modified_sequence_1 = list(original_sequence)
    for index in random.sample(range(len(modified_sequence_1)), 50):
        modified_sequence_1[index] = "A"
    modified_sequence_1 = "".join(modified_sequence_1)

    modified_sequence_2 = list(original_sequence)
    for index in random.sample(range(len(modified_sequence_2)), 50):
        modified_sequence_2[index] = "G"
    modified_sequence_2 = "".join(modified_sequence_2)

    modified_sequence_3 = list(original_sequence)
    for index in random.sample(range(len(modified_sequence_3)), 50):
        modified_sequence_3[index] = "V"
    modified_sequence_3 = "".join(modified_sequence_3)

    cache = ToolCache()
    _program_tool_cache.set(cache)

    try:
        # First pass: score original and first two modified sequences
        input_first_pass = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=modified_sequence_1, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=modified_sequence_2, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
        output_first_pass = run_proteinmpnn_score(input_first_pass, config)

        assert output_first_pass.success
        assert len(output_first_pass.scores) == 3
        validate_output(output_first_pass)

        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 3

        # Second pass: overlapping sequences plus one new
        input_second_pass = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=modified_sequence_1, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=modified_sequence_2, structure=pdb_structure
                ),
                SequenceStructurePair(
                    sequence=modified_sequence_3, structure=pdb_structure
                ),
            ]
        )
        output_second_pass = run_proteinmpnn_score(input_second_pass, config)

        assert output_second_pass.success
        assert len(output_second_pass.scores) == 4
        validate_output(output_second_pass)

        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 4

        # First three scores should match exactly
        assert (
            output_second_pass.scores[0].perplexity
            == output_first_pass.scores[0].perplexity
        )
        assert (
            output_second_pass.scores[1].perplexity
            == output_first_pass.scores[1].perplexity
        )
        assert (
            output_second_pass.scores[2].perplexity
            == output_first_pass.scores[2].perplexity
        )

        # Logits arrays should be identical for cached results
        assert np.allclose(
            output_second_pass.scores[0].logits, output_first_pass.scores[0].logits
        )
        assert np.allclose(
            output_second_pass.scores[1].logits, output_first_pass.scores[1].logits
        )
        assert np.allclose(
            output_second_pass.scores[2].logits, output_first_pass.scores[2].logits
        )

    finally:
        _program_tool_cache.set(None)


@pytest.mark.uses_gpu
def test_proteinmpnn_score_logits_disabled_by_default(pdb_structure: Structure):
    """Logits are None when return_logits=False (default)."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42)
    output = run_proteinmpnn_score(inp, config)

    assert output.success
    validate_output(output)

    for score in output.scores:
        assert score.logits is None


@pytest.mark.uses_gpu
def test_proteinmpnn_score_logits_serialization(pdb_structure: Structure):
    """Logits are properly serialized as nested lists."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(
                sequence=original_sequence, structure=pdb_structure
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)

    assert output.success
    validate_output(output)

    score = output.scores[0]

    assert isinstance(score.logits, (list, np.ndarray))

    if isinstance(score.logits, list):
        assert len(score.logits) > 0
        assert isinstance(score.logits[0], list)
        assert len(score.logits[0]) == len(ALPHAFOLD_VOCAB)

        for position_logits in score.logits:
            for logit_value in position_logits:
                assert isinstance(logit_value, (int, float))
    else:
        assert score.logits.ndim == 2
        assert score.logits.shape[1] == len(ALPHAFOLD_VOCAB)


# ============================================================================
# AbMPNN (antibody-optimized weights) tests
# ============================================================================
@pytest.mark.uses_gpu
def test_abmpnn_sample(pdb_structure: Structure):
    """AbMPNN weights load and produce valid samples."""
    inp = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=pdb_structure)]
    )
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=2,
        temperature=0.1,
        seed=42,
        model_choice="abmpnn",
    )
    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"AbMPNN sampling failed: {output}"
    assert len(output.designed_sequences[0].sequences) == 2
    assert all(isinstance(s, str) for s in output.designed_sequences[0].sequences)


@pytest.mark.uses_gpu
def test_abmpnn_score(pdb_structure: Structure):
    """AbMPNN weights load and produce valid scores."""
    sequence = pdb_structure.get_chain_sequence("A")
    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=sequence, structure=pdb_structure),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, model_choice="abmpnn")
    output = run_proteinmpnn_score(inp, config)
    assert output.success, f"AbMPNN scoring failed: {output}"
    assert output.scores[0].perplexity >= 1.0
    assert "avg_log_likelihood" in output.scores[0].metrics
