"""tests/inverse_folding_tests/test_fampnn.py

Tests for FAMPNN sampling, packing, and scoring."""

from pathlib import Path

import pytest

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.fampnn import (
    FAMPNNPackConfig,
    FAMPNNPackInput,
    FAMPNNSampleConfig,
    FAMPNNSampleInput,
    FAMPNNScoreAllMutationsConfig,
    FAMPNNScoreAllMutationsInput,
    FAMPNNScoreConfig,
    FAMPNNScoreInput,
    FAMPNNStructureInput,
    MutationInput,
    run_fampnn_pack,
    run_fampnn_sample,
    run_fampnn_score,
    run_fampnn_score_all_mutations,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"


_persistent_tool = make_persistent_fixture("fampnn")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure(structure_filepath_or_content=TEST_PDB_FILE)


# ============================================================================
# Unit tests (no GPU required, schema and data model validation)
# ============================================================================
def test_fampnn_sample_input_schema():
    """FAMPNNSampleInput accepts a structure path and validates."""
    inp = FAMPNNSampleInput(
        inputs=[FAMPNNStructureInput(structure=str(TEST_PDB_FILE))]
    )
    assert len(inp.inputs) == 1
    assert inp.inputs[0].structure is not None


def test_fampnn_sample_config_defaults():
    """FAMPNNSampleConfig has correct defaults."""
    config = FAMPNNSampleConfig()
    assert config.model_variant == "0.3"
    assert config.num_steps == 100
    assert config.temperature == 0.1
    assert config.seq_only is False
    assert config.repack_last is True
    assert config.psce_threshold == 0.3
    assert config.scn_diffusion_steps == 50
    assert config.scn_step_scale == 1.5
    assert config.num_sequences_per_structure == 1


def test_fampnn_pack_config_defaults():
    """FAMPNNPackConfig has correct defaults."""
    config = FAMPNNPackConfig()
    assert config.model_variant == "0.0"
    assert config.num_samples_per_structure == 1
    assert config.scn_diffusion_steps == 50


def test_fampnn_score_config_defaults():
    """FAMPNNScoreConfig has correct defaults."""
    config = FAMPNNScoreConfig()
    assert config.model_variant == "0.3_cath"
    assert config.batch_size == 16
    assert config.seq_only is False


def test_fampnn_score_all_mutations_config_defaults():
    """FAMPNNScoreAllMutationsConfig has correct defaults."""
    config = FAMPNNScoreAllMutationsConfig()
    assert config.model_variant == "0.3_cath"
    assert config.batch_size == 16


def test_fampnn_structure_input_with_sidechain_positions(pdb_structure):
    """FAMPNNStructureInput accepts fixed_sidechain_positions."""
    chain_ids = pdb_structure.get_chain_ids()
    first_chain = chain_ids[0]
    positions = pdb_structure.get_chain_positions(first_chain)[:5]

    inp = FAMPNNStructureInput(
        structure=pdb_structure,
        chain_ids=[first_chain],
        fixed_positions={first_chain: positions},
        fixed_sidechain_positions={first_chain: positions},
    )
    assert inp.fixed_sidechain_positions == {first_chain: positions}
    assert inp.fixed_positions == {first_chain: positions}


def test_fampnn_pack_input_schema(pdb_structure):
    """FAMPNNPackInput accepts a structure."""
    inp = FAMPNNPackInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    assert len(inp.inputs) == 1


def test_fampnn_score_input_schema(pdb_structure):
    """FAMPNNScoreInput accepts a structure and mutations."""
    inp = FAMPNNScoreInput(
        inputs=[MutationInput(
            structure=pdb_structure,
            mutations=["A1V", "G6L"],
        )]
    )
    assert len(inp.inputs) == 1
    assert inp.inputs[0].mutations == ["A1V", "G6L"]


def test_fampnn_score_all_mutations_input_schema(pdb_structure):
    """FAMPNNScoreAllMutationsInput accepts a structure."""
    inp = FAMPNNScoreAllMutationsInput(inputs=[pdb_structure])
    assert len(inp.inputs) == 1


def test_fampnn_sample_config_batch_size_defaults():
    """batch_size defaults to num_sequences_per_structure."""
    config = FAMPNNSampleConfig(num_sequences_per_structure=5)
    assert config.batch_size == 5


def test_fampnn_sample_config_custom_batch_size():
    """Custom batch_size is respected."""
    config = FAMPNNSampleConfig(num_sequences_per_structure=10, batch_size=3)
    assert config.batch_size == 3


def test_fampnn_sample_config_rejects_excluded_amino_acids():
    """excluded_amino_acids is not supported and should raise."""
    with pytest.raises(Exception, match="excluded_amino_acids.*not supported"):
        FAMPNNSampleConfig(excluded_amino_acids=["C"])


# ============================================================================
# Integration tests (GPU required, run with --integration or --all)
# ============================================================================
@pytest.mark.uses_gpu
def test_fampnn_sample_simple(pdb_structure):
    """Basic FAMPNN sequence sampling with a single structure."""
    inp = FAMPNNSampleInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=2,
        temperature=0.1,
        num_steps=10,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"
    validate_output(output)
    assert output.tool_id == "fampnn-sample"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 2
    assert all(isinstance(seq, str) for seq in designed.sequences)
    assert all(len(seq) > 0 for seq in designed.sequences)
    # FAMPNN-specific: PDB strings and pSCE
    assert len(designed.output_pdb_strings) == 2
    assert all(isinstance(pdb, str) for pdb in designed.output_pdb_strings)
    assert all("ATOM" in pdb for pdb in designed.output_pdb_strings)
    assert len(designed.psce) == 2
    assert all(isinstance(psce, list) for psce in designed.psce)
    assert all(isinstance(v, float) for psce in designed.psce for v in psce)


@pytest.mark.uses_gpu
def test_fampnn_sample_chunked_batching(pdb_structure):
    """Chunked batching produces the correct number of sequences."""
    inp = FAMPNNSampleInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=4,
        batch_size=2,
        temperature=0.1,
        num_steps=10,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 4
    assert len(designed.output_pdb_strings) == 4
    assert len(designed.psce) == 4


@pytest.mark.uses_gpu
def test_fampnn_sample_seq_only(pdb_structure):
    """seq_only mode produces sequences without sidechains."""
    inp = FAMPNNSampleInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=1,
        num_steps=10,
        seq_only=True,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"seq_only sampling failed: {output}"

    designed = output.designed_sequences[0]
    assert len(designed.sequences) == 1
    assert isinstance(designed.sequences[0], str)
    assert len(designed.sequences[0]) > 0


@pytest.mark.uses_gpu
def test_fampnn_sample_multiple_structures(pdb_structure):
    """Sampling with multiple input structures."""
    inp = FAMPNNSampleInput(
        inputs=[
            FAMPNNStructureInput(structure=pdb_structure),
            FAMPNNStructureInput(structure=pdb_structure),
        ]
    )
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=2,
        num_steps=10,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success

    assert len(output.designed_sequences) == 2
    for designed in output.designed_sequences:
        assert len(designed.sequences) == 2


@pytest.mark.uses_gpu
def test_fampnn_sample_with_fixed_positions(pdb_structure):
    """Fixed positions are accepted by the sampling tool."""
    chain_ids = pdb_structure.get_chain_ids()
    first_chain = chain_ids[0]
    all_positions = pdb_structure.get_chain_positions(first_chain)
    fixed_pos = all_positions[:10]

    inp = FAMPNNSampleInput(
        inputs=[FAMPNNStructureInput(
            structure=pdb_structure,
            chain_ids=[first_chain],
            fixed_positions={first_chain: fixed_pos},
        )]
    )
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=1,
        num_steps=10,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"Fixed positions sampling failed: {output}"
    assert len(output.designed_sequences[0].sequences) == 1


@pytest.mark.uses_gpu
def test_fampnn_pack_simple(pdb_structure):
    """Basic sidechain packing with a single structure."""
    inp = FAMPNNPackInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    config = FAMPNNPackConfig(
        num_samples_per_structure=1,
        seed=42,
    )
    output = run_fampnn_pack(inp, config)
    assert output.success, f"Failed to pack: {output}"
    validate_output(output)
    assert output.tool_id == "fampnn-pack"

    assert len(output.packed_structures) == 1
    assert len(output.packed_structures[0]) == 1
    pdb_str = output.packed_structures[0][0]
    assert isinstance(pdb_str, str)
    assert "ATOM" in pdb_str

    # pSCE should be present
    assert len(output.psce) == 1
    assert len(output.psce[0]) == 1
    assert all(isinstance(v, float) for v in output.psce[0][0])


@pytest.mark.uses_gpu
def test_fampnn_pack_multiple_samples(pdb_structure):
    """Packing produces the correct number of samples."""
    inp = FAMPNNPackInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    config = FAMPNNPackConfig(
        num_samples_per_structure=3,
        batch_size=2,
        seed=42,
    )
    output = run_fampnn_pack(inp, config)
    assert output.success

    assert len(output.packed_structures[0]) == 3
    assert len(output.psce[0]) == 3


@pytest.mark.uses_gpu
def test_fampnn_pack_psce_values(pdb_structure):
    """pSCE values are non-negative and in a reasonable range."""
    inp = FAMPNNPackInput(
        inputs=[FAMPNNStructureInput(structure=pdb_structure)]
    )
    config = FAMPNNPackConfig(seed=42)
    output = run_fampnn_pack(inp, config)
    assert output.success

    psce_values = output.psce[0][0]
    assert all(v >= 0 for v in psce_values)
    # Per-residue pSCE should generally be < 4 Angstroms for well-packed structures
    assert all(v < 10.0 for v in psce_values)


@pytest.mark.uses_gpu
def test_fampnn_score_mutations(pdb_structure):
    """Score specific mutations against a structure."""
    # Get the sequence to construct valid mutations
    chain_ids = pdb_structure.get_chain_ids()
    first_chain = chain_ids[0]
    seq = pdb_structure.get_chain_sequence(first_chain)

    # Create valid mutations using 1-indexed positions
    wt_res_1 = seq[0]
    mut_res_1 = "A" if wt_res_1 != "A" else "G"
    wt_res_6 = seq[5]
    mut_res_6 = "V" if wt_res_6 != "V" else "L"

    mutations = [
        f"{wt_res_1}1{mut_res_1}",
        f"{wt_res_6}6{mut_res_6}",
        "wt",
    ]

    inp = FAMPNNScoreInput(
        inputs=[MutationInput(
            structure=pdb_structure,
            mutations=mutations,
        )]
    )
    config = FAMPNNScoreConfig(seed=42)
    output = run_fampnn_score(inp, config)
    assert output.success, f"Failed to score: {output}"
    validate_output(output)
    assert output.tool_id == "fampnn-score"

    result = output.results[0]
    assert len(result.mutations) == 3
    assert len(result.scores) == 3
    assert all(isinstance(s, float) for s in result.scores)
    # Wild-type score should be 0
    assert result.scores[2] == 0.0


@pytest.mark.uses_gpu
def test_fampnn_score_mutations_seq_only(pdb_structure):
    """Score mutations in seq_only mode (no sidechain context)."""
    chain_ids = pdb_structure.get_chain_ids()
    first_chain = chain_ids[0]
    seq = pdb_structure.get_chain_sequence(first_chain)

    wt_res = seq[0]
    mut_res = "A" if wt_res != "A" else "G"

    inp = FAMPNNScoreInput(
        inputs=[MutationInput(
            structure=pdb_structure,
            mutations=[f"{wt_res}1{mut_res}"],
        )]
    )
    config = FAMPNNScoreConfig(seq_only=True, seed=42)
    output = run_fampnn_score(inp, config)
    assert output.success

    assert len(output.results[0].scores) == 1
    assert isinstance(output.results[0].scores[0], float)


@pytest.mark.uses_gpu
def test_fampnn_score_all_mutations(pdb_structure):
    """Score all possible single mutations at every position."""
    inp = FAMPNNScoreAllMutationsInput(inputs=[pdb_structure])
    config = FAMPNNScoreAllMutationsConfig(seed=42)
    output = run_fampnn_score_all_mutations(inp, config)
    assert output.success, f"Failed to score all mutations: {output}"
    validate_output(output)
    assert output.tool_id == "fampnn-score-all-mutations"

    result = output.results[0]
    assert isinstance(result.scores, dict)
    assert len(result.scores) > 0

    # Each position should have scores for all 20 amino acids
    first_key = next(iter(result.scores))
    assert isinstance(result.scores[first_key], dict)
    assert len(result.scores[first_key]) == 20

    # Wild-type mutation should have score ~0
    for pos_label, scores in result.scores.items():
        wt_residue = pos_label[-1]  # last char is the wild-type residue
        if wt_residue in scores:
            assert abs(scores[wt_residue]) < 1e-5, (
                f"Wild-type score at {pos_label} should be ~0, got {scores[wt_residue]}"
            )


@pytest.mark.uses_gpu
def test_fampnn_score_all_mutations_scores_range(pdb_structure):
    """All-mutations scores are finite and in a reasonable range."""
    inp = FAMPNNScoreAllMutationsInput(inputs=[pdb_structure])
    config = FAMPNNScoreAllMutationsConfig(seed=42)
    output = run_fampnn_score_all_mutations(inp, config)
    assert output.success

    for pos_label, scores in output.results[0].scores.items():
        for residue, score in scores.items():
            assert isinstance(score, float)
            assert not (score != score), f"NaN score at {pos_label}->{residue}"
            # Log-likelihood ratios should be bounded
            assert abs(score) < 50, f"Extreme score at {pos_label}->{residue}: {score}"
