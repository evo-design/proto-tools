"""
test_proteinmpnn.py

Tests for ProteinMPNN inverse folding tools.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from bio_programming.bio_tools.tools.inverse_folding.proteinmpnn import (
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    run_proteinmpnn_sample,
    run_proteinmpnn_score,
)
from bio_programming.bio_tools.tools.inverse_folding.proteinmpnn.standalone.inference import (
    ALPHAFOLD_VOCAB,
)
from bio_programming.bio_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingConfig,
    InverseFoldingInput,
    InverseFoldingStructureInput,
    SequenceStructurePair,
)
from bio_programming.bio_tools.tools.inverse_folding.shared_data_models import SequenceScores
from bio_programming.bio_tools.entities.structures.structure import Structure
from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure(structure_filepath_or_content=TEST_PDB_FILE)


class TestProteinMPNNSample:

    @pytest.mark.uses_gpu
    def test_proteinmpnn_sample_simple(self, pdb_structure: Structure):
        input = InverseFoldingInput(
            inputs=[
                InverseFoldingStructureInput(structure=pdb_structure),
                InverseFoldingStructureInput(structure=pdb_structure),
                InverseFoldingStructureInput(structure=pdb_structure),
            ]
        )
        config = InverseFoldingConfig(
            batch_size=10, temperature=1.0, seed=42,
        )
        output = run_proteinmpnn_sample(input, config)
        assert (
            output.success
        ), f"Failed to sample protein sequences using ProteinMPNN: {output}"

        validate_output(output)
        assert output.tool_id == "proteinmpnn-sample"

        # Validate outputs
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
    def test_proteinmpnn_sample_advanced_args(self, pdb_structure: Structure):
        """
        A more complex tests that ensures that advanced config settings work as expected.
        Tests the following advanced config settings:
        - fixed_positions
        - excluded_amino_acids
        """
        chain_A = pdb_structure.get_chain_sequence("A")

        # Find all indicies of the amino acid "C" in chain A
        c_positions = [i + 1 for i, aa in enumerate(chain_A) if aa == "C"]

        # Make a list of fixed indices that do not contain the "C" positions
        fixed_positions = random.sample(
            list(set(np.arange(len(chain_A)) + 1) - set(c_positions)), 200
        )

        input = InverseFoldingInput(
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
            batch_size=10,
            temperature=1.0,
            seed=42,
            excluded_amino_acids=["C"],
        )

        output = run_proteinmpnn_sample(input, config)
        assert (
            output.success
        ), f"Failed to sample protein sequences using ProteinMPNN: {output}"

        validate_output(output)

        # Validate outputs followed the advanced config settings
        for designed_sequences in output.designed_sequences:
            for sequence in designed_sequences.sequences:
                # Ensure that the sequence does not contain the "C" amino acid
                # - We hardcoded that all C positions are designable above
                # - We do not allow C amino acids in the sequence
                assert (
                    "C" not in sequence
                ), f"Sequence contains 'C' amino acid: {sequence}"

                # Ensure that the sequence does not contain the fixed indices
                for position in fixed_positions:
                    assert (
                        sequence[position - 1] == chain_A[position - 1]
                    ), f"Designed sequence is not the same as the PDB prompt sequence at fixed position {position} (index {position-1}): {sequence[position-1]} != {chain_A[position-1]}"


class TestProteinMPNNScore:

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score(self, pdb_structure: Structure):
        original_sequence = pdb_structure.get_chain_sequence("A")

        # Randomly change a few positions to 'C'
        modified_sequence = list(original_sequence)
        for index in random.sample(range(len(modified_sequence)), 100):
            modified_sequence[index] = "C"
        modified_sequence = "".join(modified_sequence)

        # Randomly select 100 positions to fix
        fixed_positions = random.sample(list(range(len(original_sequence))), 100)
        fixed_positions = {
            "A": [position + 1 for position in fixed_positions],
        }

        input = ProteinMPNNScoringInput(
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
        output = run_proteinmpnn_score(input, config)
        assert (
            output.success
        ), f"Failed to score protein sequence using ProteinMPNN: {output}"

        validate_output(output)
        assert output.tool_id == "proteinmpnn-score"
        assert output.vocab == ALPHAFOLD_VOCAB

        # Validate outputs
        assert len(output.scores) == 2
        assert all(isinstance(score, SequenceScores) for score in output.scores)

        # Ensure the perplexity of the original sequence is lower than the modified sequence
        assert output.scores[0].perplexity < output.scores[1].perplexity

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_fields(self, pdb_structure: Structure):
        """Test all scoring fields and their mathematical relationships."""
        original_sequence = pdb_structure.get_chain_sequence("A")
        seq_len = len(original_sequence)

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
        output = run_proteinmpnn_score(input, config)
        assert output.success

        validate_output(output)

        score = output.scores[0]

        # Validate metrics via attribute access (via __getattr__)
        assert isinstance(score.log_likelihood, float)
        assert isinstance(score.avg_log_likelihood, float)
        assert isinstance(score.perplexity, float)

        # Validate metrics via dict access
        assert isinstance(score.metrics["log_likelihood"], float)
        assert isinstance(score.metrics["avg_log_likelihood"], float)
        assert isinstance(score.metrics["perplexity"], float)

        # Validate logits (convert from nested list to ndarray for numeric checks)
        logits_arr = np.array(score.logits)
        assert logits_arr.ndim == 2

        # Validate mathematical relationships
        # log_likelihood = avg_log_likelihood * seq_len
        assert np.isclose(
            score.log_likelihood, score.avg_log_likelihood * seq_len, rtol=1e-5
        )

        # perplexity = exp(-avg_log_likelihood)
        assert np.isclose(
            score.perplexity, np.exp(-score.avg_log_likelihood), rtol=1e-5
        ), "Perplexity should equal exp(-avg_log_likelihood)"

        # avg_log_likelihood should be negative (log probs are <= 0)
        assert score.avg_log_likelihood <= 0

        # perplexity should be >= 1 (since exp(-x) >= 1 when x <= 0)
        assert score.perplexity >= 1.0

        # Validate logits shape: (seq_len, vocab_size)
        assert logits_arr.shape == (seq_len, len(ALPHAFOLD_VOCAB))

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_vocab(self, pdb_structure: Structure):
        """Test the vocab property on output (from each SequenceScores)."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42)
        output = run_proteinmpnn_score(input, config)
        assert output.success

        validate_output(output)

        # Test vocab on output (convenience from first score)
        assert output.vocab == ALPHAFOLD_VOCAB
        # Test vocab on each score
        assert output.scores[0].vocab == ALPHAFOLD_VOCAB

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_single_pair(self, pdb_structure: Structure):
        """Test ProteinMPNN scoring with a single sequence-structure pair."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
        output = run_proteinmpnn_score(input, config)

        assert output.success
        validate_output(output)
        assert len(output.scores) == 1
        assert output.scores[0].perplexity >= 1.0
        assert output.scores[0].logits is not None

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_batched(self, pdb_structure: Structure):
        """Test batched scoring with multiple sequence-structure pairs."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        # Create a few modified sequences
        modified_1 = list(original_sequence)
        for i in random.sample(range(len(modified_1)), 30):
            modified_1[i] = "A"
        modified_1 = "".join(modified_1)
        modified_2 = list(original_sequence)
        for i in random.sample(range(len(modified_2)), 30):
            modified_2[i] = "G"
        modified_2 = "".join(modified_2)

        input = ProteinMPNNScoringInput(
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
        output = run_proteinmpnn_score(input, config)

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
    def test_proteinmpnn_score_cache(self, pdb_structure: Structure):
        """
        Tests the caching functionality of ProteinMPNN scoring tool
        """
        from bio_programming.bio_tools.tools.infra.tool_cache import (
            ToolCache,
            _program_tool_cache,
            get_cache_info,
        )

        original_sequence = pdb_structure.get_chain_sequence("A")

        # Create three different modified sequences
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

        # Set up the cache
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

            # Verify first pass succeeded
            assert (
                output_first_pass.success
            ), f"Failed to score protein sequence using ProteinMPNN: {output_first_pass}"
            assert len(output_first_pass.scores) == 3
            validate_output(output_first_pass)

            # Cache should have three entries
            cache_info = get_cache_info()
            assert cache_info["total_entries"] == 3

            # Second pass: score with overlapping sequences plus one new sequence
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

            # Verify second pass succeeded
            assert (
                output_second_pass.success
            ), f"Failed to score protein sequence using ProteinMPNN: {output_second_pass}"
            assert len(output_second_pass.scores) == 4
            validate_output(output_second_pass)

            # Cache should have four entries (first three were cached, one new)
            cache_info = get_cache_info()
            assert cache_info["total_entries"] == 4

            # Verify scores are consistent - first three should match exactly
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

            # Verify logits arrays are identical for cached results
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
            # Clean up cache
            _program_tool_cache.set(None)


# ============================================================================
# Logits-Specific Tests
# ============================================================================

class TestProteinMPNNLogits:

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_logits_disabled_by_default(self, pdb_structure: Structure):
        """Test that logits are None when return_logits=False (default)."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42)
        output = run_proteinmpnn_score(input, config)

        assert output.success
        validate_output(output)

        # Logits should be None when return_logits=False
        for score in output.scores:
            assert score.logits is None, "Logits should be None when return_logits=False"

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_logits_enabled(self, pdb_structure: Structure):
        """Test that logits are correctly returned when return_logits=True."""
        original_sequence = pdb_structure.get_chain_sequence("A")
        seq_len = len(original_sequence)

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
        output = run_proteinmpnn_score(input, config)

        assert output.success
        validate_output(output)

        score = output.scores[0]

        # Logits should be present with correct shape
        assert score.logits is not None, "Logits should not be None when return_logits=True"
        assert isinstance(score.logits, (list, np.ndarray)), f"Logits should be list or ndarray, got {type(score.logits)}"
        
        # Convert to ndarray for shape validation if it's a list
        logits_arr = np.array(score.logits)
        assert logits_arr.shape[0] == seq_len, f"Logits length should be {seq_len}, got {logits_arr.shape[0]}"
        assert logits_arr.shape[1] == len(ALPHAFOLD_VOCAB), f"ProteinMPNN vocab size should be {len(ALPHAFOLD_VOCAB)}, got {logits_arr.shape[1]}"

    @pytest.mark.uses_gpu
    def test_proteinmpnn_score_logits_serialization(self, pdb_structure: Structure):
        """Test that logits are properly serialized as nested lists."""
        original_sequence = pdb_structure.get_chain_sequence("A")

        input = ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(
                    sequence=original_sequence, structure=pdb_structure
                ),
            ]
        )
        config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
        output = run_proteinmpnn_score(input, config)

        assert output.success
        validate_output(output)

        score = output.scores[0]
        
        # Logits should be serialized as nested lists (not tensors)
        assert isinstance(score.logits, (list, np.ndarray)), "Logits should be list or ndarray"
        
        if isinstance(score.logits, list):
            # Verify nested list structure
            assert len(score.logits) > 0, "Logits list should not be empty"
            assert isinstance(score.logits[0], list), "Logits should be a list of lists"
            assert len(score.logits[0]) == len(ALPHAFOLD_VOCAB), f"Inner logits list should have {len(ALPHAFOLD_VOCAB)} elements (vocab size)"
            
            # Verify all values are numeric
            for position_logits in score.logits:
                for logit_value in position_logits:
                    assert isinstance(logit_value, (int, float)), f"Logit value should be numeric, got {type(logit_value)}"
        else:
            # If ndarray, verify shape
            assert score.logits.ndim == 2, "Logits should be 2D array"
            assert score.logits.shape[1] == len(ALPHAFOLD_VOCAB), f"ProteinMPNN vocab size should be {len(ALPHAFOLD_VOCAB)}"
