"""tests/inverse_folding_tests/test_fampnn.py.

Tests for FAMPNN sampling, packing, and scoring.
"""

from pathlib import Path

import pytest

from proto_tools.entities.complex import Chain
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
from proto_tools.tools.inverse_folding.fampnn.fampnn_sample import (
    FAMPNNDesign,
    FAMPNNDesignMetrics,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
BENCHMARK_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"

# Minimal 3-residue single-chain backbone PDB used to build a Structure without disk or GPU.
_SMALL_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.326   1.562   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.941   2.877   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.449   2.831   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.074   1.772   0.000  1.00  0.00           O
ATOM      9  N   SER A   3       6.032   4.027   0.000  1.00  0.00           N
ATOM     10  CA  SER A   3       7.476   4.180   0.000  1.00  0.00           C
ATOM     11  C   SER A   3       8.064   5.572   0.000  1.00  0.00           C
ATOM     12  O   SER A   3       7.337   6.562   0.000  1.00  0.00           O
TER
END
"""


_persistent_tool = make_persistent_fixture("fampnn")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure.from_file(TEST_PDB_FILE)


@pytest.fixture(scope="module")
def benchmark_pdb_structure():
    return Structure.from_file(BENCHMARK_PDB_FILE)


# ============================================================================
# Unit tests (no GPU required, schema and data model validation)
# ============================================================================
def test_fampnn_sample_input_schema():
    """FAMPNNSampleInput accepts a structure path and validates."""
    inp = FAMPNNSampleInput(inputs=[FAMPNNStructureInput(structure=str(TEST_PDB_FILE))])
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
    """FAMPNNStructureInput accepts a fixed_sidechain_positions selection."""
    chain_ids = pdb_structure.get_chain_ids()
    first_chain = chain_ids[0]
    positions = pdb_structure.get_chain_positions(first_chain)[:5]

    inp = FAMPNNStructureInput(
        structure=pdb_structure,
        chains_to_redesign=[first_chain],
        fixed_positions={first_chain: positions},
        fixed_sidechain_positions={first_chain: positions},
    )
    assert inp.fixed_sidechain_positions is not None
    assert inp.fixed_sidechain_positions.chains == {first_chain: positions}
    assert inp.fixed_positions is not None
    assert inp.fixed_positions.chains == {first_chain: positions}


def test_fampnn_pack_input_schema(pdb_structure):
    """FAMPNNPackInput accepts a structure."""
    inp = FAMPNNPackInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
    assert len(inp.inputs) == 1


def test_fampnn_score_input_schema(pdb_structure):
    """FAMPNNScoreInput accepts a structure and mutations."""
    inp = FAMPNNScoreInput(
        inputs=[
            MutationInput(
                structure=pdb_structure,
                mutations=["A1V", "G6L"],
            )
        ]
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
    """FAMPNN does not declare excluded_amino_acids; extra='forbid' rejects it at construction."""
    with pytest.raises(Exception, match=r"[Ee]xtra"):
        FAMPNNSampleConfig(excluded_amino_acids=["C"])


def test_fampnn_design_structure_and_metrics():
    """FAMPNNDesign holds all chains, a packed structure, and full-atom pSCE metrics."""
    structure = Structure(structure=_SMALL_PDB, structure_format="pdb", source="test")
    design = FAMPNNDesign(
        chains=[
            Chain(id="A", sequence="AGS"),
            Chain(id="B", sequence="MKT"),
        ],
        structure=structure,
        designed=[True, False],
        metrics=FAMPNNDesignMetrics(avg_psce=0.25, psce=[0.2, 0.3]),
    )

    assert len(design.chains) == 2
    assert design.chains[0].id == "A"
    assert design.designed[0] is True
    assert design.chains[1].id == "B"
    assert design.designed[1] is False
    assert design.designed_chains == [design.chains[0]]
    assert design.chain_sequences == ["AGS", "MKT"]
    assert design.chain_sequence_map() == {"A": "AGS", "B": "MKT"}

    assert design.structure is not None
    assert design.structure.structure_format == "pdb"
    assert "ATOM" in design.structure.structure

    assert design.metrics["avg_psce"] == 0.25
    assert design.metrics.psce == [0.2, 0.3]
    assert design.design_metrics()["avg_psce"] == 0.25


def test_fampnn_design_feeds_structure_predictor():
    """A FAMPNNDesign feeds an SP tool input directly via LSP (subclass-as-Complex)."""
    from proto_tools.tools.structure_prediction import ESMFoldInput

    structure = Structure(structure=_SMALL_PDB, structure_format="pdb", source="test")
    design = FAMPNNDesign(
        chains=[
            Chain(id="A", sequence="AGS"),
            Chain(id="B", sequence="MKT"),
        ],
        structure=structure,
        designed=[True, False],
        metrics=FAMPNNDesignMetrics(avg_psce=0.1, psce=[0.1, 0.1]),
    )

    inp = ESMFoldInput(complexes=[design])
    assert inp.complexes[0] is design
    assert inp.complexes[0].chain_sequences == ["AGS", "MKT"]


# ============================================================================
# Integration tests (GPU required, run with --integration or --all)
# ============================================================================
@pytest.mark.uses_gpu
def test_fampnn_sample_simple(pdb_structure):
    """Basic FAMPNN sequence sampling with a single structure."""
    inp = FAMPNNSampleInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
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

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 2
    assert all(isinstance(d.chains[0].sequence, str) for d in design_set.complexes)
    assert all(len(d.chains[0].sequence) > 0 for d in design_set.complexes)
    # FAMPNN-specific: per-design packed structure and pSCE metrics
    assert all(d.structure.structure_format == "pdb" for d in design_set.complexes)
    assert all("ATOM" in d.structure.structure for d in design_set.complexes)
    for d in design_set.complexes:
        assert isinstance(d.metrics["avg_psce"], float)
        assert isinstance(d.metrics.psce, list)
        assert all(isinstance(v, float) for v in d.metrics.psce)
    assert_metrics_in_spec(output)


@pytest.mark.uses_gpu
def test_fampnn_sample_chunked_batching(pdb_structure):
    """Chunked batching produces the correct number of sequences."""
    inp = FAMPNNSampleInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=4,
        batch_size=2,
        temperature=0.1,
        num_steps=10,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 4
    assert all(d.structure.structure_format == "pdb" for d in design_set.complexes)
    assert all(isinstance(d.metrics.psce, list) for d in design_set.complexes)


@pytest.mark.uses_gpu
def test_fampnn_sample_seq_only(pdb_structure):
    """seq_only mode produces sequences without sidechains."""
    inp = FAMPNNSampleInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=1,
        num_steps=10,
        seq_only=True,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"seq_only sampling failed: {output}"

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 1
    seq = design_set.complexes[0].chains[0].sequence
    assert isinstance(seq, str)
    assert len(seq) > 0


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

    assert len(output.design_sets) == 2
    for design_set in output.design_sets:
        assert len(design_set.complexes) == 2


@pytest.mark.uses_gpu
def test_fampnn_sample_with_fixed_positions(pdb_structure):
    """Fixed positions are accepted by the sampling tool."""
    chain_ids = pdb_structure.get_chain_ids()
    first_chain = chain_ids[0]
    all_positions = pdb_structure.get_chain_positions(first_chain)
    fixed_pos = all_positions[:10]

    inp = FAMPNNSampleInput(
        inputs=[
            FAMPNNStructureInput(
                structure=pdb_structure,
                chains_to_redesign=[first_chain],
                fixed_positions={first_chain: fixed_pos},
            )
        ]
    )
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=1,
        num_steps=10,
        seed=42,
    )
    output = run_fampnn_sample(inp, config)
    assert output.success, f"Fixed positions sampling failed: {output}"
    assert len(output.design_sets[0].complexes) == 1


@pytest.mark.uses_gpu
def test_fampnn_pack_simple(pdb_structure):
    """Basic sidechain packing with a single structure."""
    inp = FAMPNNPackInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
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
    struct = output.packed_structures[0][0]
    assert struct.structure_format == "pdb"
    assert "ATOM" in struct.structure

    # pSCE should be present
    assert len(output.psce) == 1
    assert len(output.psce[0]) == 1
    assert all(isinstance(v, float) for v in output.psce[0][0])


@pytest.mark.uses_gpu
def test_fampnn_pack_multiple_samples(pdb_structure):
    """Packing produces the correct number of samples."""
    inp = FAMPNNPackInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
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
    """PSCE values are non-negative and in a reasonable range."""
    inp = FAMPNNPackInput(inputs=[FAMPNNStructureInput(structure=pdb_structure)])
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
        inputs=[
            MutationInput(
                structure=pdb_structure,
                mutations=mutations,
            )
        ]
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
        inputs=[
            MutationInput(
                structure=pdb_structure,
                mutations=[f"{wt_res}1{mut_res}"],
            )
        ]
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
            assert score == score, f"NaN score at {pos_label}->{residue}"
            # Log-likelihood ratios should be bounded
            assert abs(score) < 50, f"Extreme score at {pos_label}->{residue}: {score}"


# ============================================================================
# Benchmarks
# ============================================================================
def _random_mutations(sequence: str, n: int, seed: int) -> list[str]:
    """Generate ``n`` deterministic random single-point mutations on ``sequence``."""
    import random as _random

    rng = _random.Random(seed)
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    mutations: list[str] = []
    for _ in range(n):
        pos = rng.randint(1, len(sequence))
        wt = sequence[pos - 1]
        mut = rng.choice([aa for aa in aa_alphabet if aa != wt])
        mutations.append(f"{wt}{pos}{mut}")
    return mutations


@pytest.mark.benchmark("fampnn-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_fampnn_sample_benchmark(request: pytest.FixtureRequest, benchmark_pdb_structure: Structure) -> None:
    """Benchmark fampnn-sample: 20 full-atom complexes of renin (~340 aa) at batch_size=8 (cold + warm).

    Uses default ``num_steps=100`` and ``model_variant="0.3"`` so each design exercises
    the full iterative-MLM + sidechain-diffusion path. FAMPNN sample is much heavier
    per sequence than ProteinMPNN, so the workload is sized smaller than proteinmpnn-sample.
    """
    inputs = FAMPNNSampleInput(inputs=[FAMPNNStructureInput(structure=benchmark_pdb_structure)])
    config = FAMPNNSampleConfig(
        num_sequences_per_structure=20,
        batch_size=8,
        temperature=0.1,
        seed=0,
    )

    result = benchmark_twice(request, "fampnn", lambda: run_fampnn_sample(inputs, config))

    assert result.tool_id == "fampnn-sample"
    assert len(result.design_sets) == 1
    design_set = result.design_sets[0]
    assert len(design_set.complexes) == 20
    target_len = len(benchmark_pdb_structure.get_chain_sequence("A"))
    for d in design_set.complexes:
        assert len(d.chains[0].sequence) == target_len
    assert all("ATOM" in d.structure.structure for d in design_set.complexes)


@pytest.mark.benchmark("fampnn-pack")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_fampnn_pack_benchmark(request: pytest.FixtureRequest, benchmark_pdb_structure: Structure) -> None:
    """Benchmark fampnn-pack: 20 sidechain repacks of renin (~340 aa) at batch_size=8 (cold + warm)."""
    inputs = FAMPNNPackInput(inputs=[FAMPNNStructureInput(structure=benchmark_pdb_structure)])
    config = FAMPNNPackConfig(
        num_samples_per_structure=20,
        batch_size=8,
        seed=0,
    )

    result = benchmark_twice(request, "fampnn", lambda: run_fampnn_pack(inputs, config))

    assert result.tool_id == "fampnn-pack"
    assert len(result.packed_structures) == 1
    assert len(result.packed_structures[0]) == 20
    for struct in result.packed_structures[0]:
        assert "ATOM" in struct.structure
    assert len(result.psce[0]) == 20


@pytest.mark.benchmark("fampnn-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_fampnn_score_benchmark(request: pytest.FixtureRequest, benchmark_pdb_structure: Structure) -> None:
    """Benchmark fampnn-score on 200 random single-point mutations of renin at batch_size=16 (cold + warm).

    fampnn-score runs the full denoiser + sidechain diffusion (50 steps) per
    batch, so peak GPU memory scales sharply with batch_size for large
    structures. batch_size=16 (the implementation default) fits a 340-aa
    structure on an 80 GiB GPU; 32 OOMs.
    """
    seq = benchmark_pdb_structure.get_chain_sequence("A")
    mutations = _random_mutations(seq, n=200, seed=0)

    inputs = FAMPNNScoreInput(
        inputs=[
            MutationInput(structure=benchmark_pdb_structure, mutations=mutations),
        ]
    )
    config = FAMPNNScoreConfig(batch_size=16, seed=0)

    result = benchmark_twice(request, "fampnn", lambda: run_fampnn_score(inputs, config))

    assert result.tool_id == "fampnn-score"
    assert len(result.results) == 1
    assert len(result.results[0].mutations) == 200
    assert len(result.results[0].scores) == 200
    assert all(isinstance(s, float) for s in result.results[0].scores)


@pytest.mark.benchmark("fampnn-score-all-mutations")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_fampnn_score_all_mutations_benchmark(
    request: pytest.FixtureRequest, benchmark_pdb_structure: Structure
) -> None:
    """Benchmark fampnn-score-all-mutations on renin (~340 positions x 20 AAs) at batch_size=32 (cold + warm)."""
    inputs = FAMPNNScoreAllMutationsInput(inputs=[benchmark_pdb_structure])
    config = FAMPNNScoreAllMutationsConfig(batch_size=32)

    result = benchmark_twice(request, "fampnn", lambda: run_fampnn_score_all_mutations(inputs, config))

    assert result.tool_id == "fampnn-score-all-mutations"
    assert len(result.results) == 1
    scores = result.results[0].scores
    target_len = len(benchmark_pdb_structure.get_chain_sequence("A"))
    assert len(scores) == target_len
    first_pos = next(iter(scores))
    assert len(scores[first_pos]) == 20
