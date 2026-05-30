"""tests/inverse_folding_tests/test_esm_if1.py.

Tests for ESM-IF1/ProteinDPO sampling and scoring.
"""

from pathlib import Path

import numpy as np
import pytest

from proto_tools.entities.complex import Chain
from proto_tools.entities.structures import SingleChainSelection
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.esm_if1 import (
    ESMIF1SampleConfig,
    ESMIF1ScoringConfig,
    ESMIF1ScoringInput,
    ESMIF1ScoringPair,
    run_esm_if1_sample,
    run_esm_if1_score,
)
from proto_tools.tools.inverse_folding.esm_if1.esm_if1_sample import (
    ESMIF1Design,
    ESMIF1DesignMetrics,
)
from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingInput,
    InverseFoldingScoringMetrics,
    InverseFoldingStructureInput,
)
from tests.conftest import benchmark_twice, make_persistent_fixture, random_protein_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
MULTICHAIN_PDB_FILE = (
    Path(__file__).parents[2] / "proto_tools" / "tools" / "structure_scoring" / "ipsae" / "example_input_fixture.pdb"
)

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

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 1
    design = design_set.complexes[0]
    seq = design.designed_chains[0].sequence
    assert isinstance(seq, str)
    assert len(seq) > 0
    ll = design.metrics["log_likelihood"]
    assert isinstance(ll, float)
    assert np.isfinite(ll)


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

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 3
    seqs = [d.designed_chains[0].sequence for d in design_set.complexes]
    assert all(isinstance(seq, str) for seq in seqs)
    assert all(len(seq) > 0 for seq in seqs)
    lls = [d.metrics["log_likelihood"] for d in design_set.complexes]
    assert all(isinstance(ll, float) for ll in lls)
    assert all(np.isfinite(ll) for ll in lls)


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

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 4
    seqs = [d.designed_chains[0].sequence for d in design_set.complexes]
    assert all(isinstance(seq, str) for seq in seqs)
    assert all(len(seq) > 0 for seq in seqs)
    assert all(np.isfinite(d.metrics["log_likelihood"]) for d in design_set.complexes)


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

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 2
    for design in design_set.complexes:
        seq = design.designed_chains[0].sequence
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

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 1
    seq = design_set.complexes[0].designed_chains[0].sequence
    assert isinstance(seq, str)
    assert len(seq) > 0


# ============================================================================
# Design structure tests (no dispatch)
# ============================================================================
def test_esm_if1_design_single_target_chain_structure():
    """An ESMIF1Design holds one redesigned target chain plus fixed context chains."""
    # Chain B is the redesigned target; A and C are carried-over context chains.
    chains = [
        Chain(id="A", sequence="MKTAYIAK"),
        Chain(id="B", sequence="GGGSSSGG"),
        Chain(id="C", sequence="LLLVVVLL"),
    ]
    design = ESMIF1Design(
        chains=chains,
        designed=[False, True, False],
        metrics=ESMIF1DesignMetrics(log_likelihood=-9.6, avg_log_likelihood=-1.2, perplexity=3.32),
    )

    designed = design.designed_chains
    assert len(designed) == 1
    assert designed[0].id == "B"
    assert designed[0].sequence == "GGGSSSGG"

    # Chain ids and order are preserved from the input structure.
    assert [c.id for c in design.chains] == ["A", "B", "C"]
    assert design.chain_sequences == ["MKTAYIAK", "GGGSSSGG", "LLLVVVLL"]
    assert design.chain_sequence_map() == {"A": "MKTAYIAK", "B": "GGGSSSGG", "C": "LLLVVVLL"}

    # Metrics expose the standard triple; perplexity is the primary value.
    assert design.metrics["log_likelihood"] == -9.6
    assert design.metrics["avg_log_likelihood"] == -1.2
    assert design.metrics.primary_value == 3.32
    assert design.design_metrics()["perplexity"] == 3.32


def test_esm_if1_design_feeds_structure_predictor():
    """An ESMIF1Design feeds an SP tool input directly via LSP (subclass-as-Complex)."""
    from proto_tools.tools.structure_prediction import ESMFoldInput

    design = ESMIF1Design(
        chains=[
            Chain(id="A", sequence="MKTAYIAK"),
            Chain(id="B", sequence="GGGSSSGG"),
        ],
        designed=[False, True],
        metrics=ESMIF1DesignMetrics(log_likelihood=-5.6, avg_log_likelihood=-0.7, perplexity=2.01),
    )

    inp = ESMFoldInput(complexes=[design])
    assert inp.complexes[0] is design
    assert inp.complexes[0].chain_sequences == ["MKTAYIAK", "GGGSSSGG"]


# ============================================================================
# Scoring Tests
# ============================================================================
@pytest.mark.uses_gpu
def test_esm_if1_score(pdb_structure: Structure):
    """Score a sequence against a structure."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ESMIF1ScoringInput(
        sequence_structure_pairs=[
            ESMIF1ScoringPair(sequence=original_sequence, structure=pdb_structure),
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
            ESMIF1ScoringPair(sequence=original_sequence, structure=pdb_structure),
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


# ============================================================================
# Multi-chain scoring tests
# ============================================================================
@pytest.fixture(scope="module")
def multichain_structure():
    """2-chain complex fixture (chain A: 21 aa, chain B: 30 aa)."""
    return Structure.from_file(MULTICHAIN_PDB_FILE)


@pytest.mark.uses_gpu
def test_esm_if1_score_multichain(multichain_structure: Structure):
    """Score each chain of a 2-chain complex separately within the complex context."""
    chain_ids = multichain_structure.get_chain_ids()
    assert chain_ids == ["A", "B"], f"Fixture changed shape: chains={chain_ids}"

    pairs = [
        ESMIF1ScoringPair(
            sequence=multichain_structure.get_chain_sequence(chain),
            structure=multichain_structure,
            target_chain=chain,
        )
        for chain in chain_ids
    ]
    output = run_esm_if1_score(ESMIF1ScoringInput(sequence_structure_pairs=pairs), ESMIF1ScoringConfig())
    assert output.success, f"Failed to score multi-chain complex: {output}"
    assert len(output.scores) == 2
    for score in output.scores:
        assert isinstance(score, InverseFoldingScoringMetrics)
        assert score.avg_log_likelihood <= 0
        assert score.perplexity >= 1.0
        assert np.isfinite(score.avg_log_likelihood)


@pytest.mark.uses_gpu
def test_esm_if1_score_multichain_context_matters(multichain_structure: Structure):
    """The complex context conditions the score: chain A scored in A+B differs from chain A scored alone.

    Bug-coverage test: with the pre-fix wrapper, the "in complex" call
    crashed with a shape mismatch on multi-chain inputs;
    if both calls returned the same number, the wrapper would be ignoring
    the other chains' coordinates instead of conditioning on them.
    """
    target_chain = "A"
    target_seq = multichain_structure.get_chain_sequence(target_chain)
    chain_a_only = multichain_structure.select_chain(target_chain)

    inp = ESMIF1ScoringInput(
        sequence_structure_pairs=[
            ESMIF1ScoringPair(sequence=target_seq, structure=multichain_structure, target_chain=target_chain),
            ESMIF1ScoringPair(sequence=target_seq, structure=chain_a_only, target_chain=target_chain),
        ]
    )
    output = run_esm_if1_score(inp, ESMIF1ScoringConfig())
    assert output.success, f"Failed to score: {output}"

    ll_in_complex = output.scores[0].avg_log_likelihood
    ll_alone = output.scores[1].avg_log_likelihood
    assert np.isfinite(ll_in_complex) and np.isfinite(ll_alone)
    assert not np.isclose(ll_in_complex, ll_alone), (
        f"Multi-chain context appears ignored: chain A scored in A+B ({ll_in_complex}) "
        f"matches chain A scored alone ({ll_alone})."
    )


def test_esm_if1_score_multichain_requires_target_chain(multichain_structure: Structure):
    """Multi-chain structure without target_chain raises a clear ValueError at pair construction."""
    chain_b_seq = multichain_structure.get_chain_sequence("B")
    with pytest.raises(ValueError, match=r"target_chain.*required for multi-chain"):
        ESMIF1ScoringPair(sequence=chain_b_seq, structure=multichain_structure)


def test_esm_if1_score_length_validation(multichain_structure: Structure):
    """Sequence length must match the target chain; mismatch raises at pair construction."""
    chain_a_seq = multichain_structure.get_chain_sequence("A")  # length 21
    with pytest.raises(ValueError, match="does not match target chain"):
        ESMIF1ScoringPair(sequence=chain_a_seq, structure=multichain_structure, target_chain="B")


def test_esm_if1_score_mse_treated_as_methionine():
    """MSE (selenomethionine) counts as M in the chain length, matching the standalone."""
    pdb = "\n".join(
        [
            "ATOM      1  N   MET A   1      -8.901   4.127  -0.555  1.00 30.00           N",
            "ATOM      2  CA  MET A   1      -8.608   3.135  -1.618  1.00 30.00           C",
            "ATOM      3  C   MET A   1      -7.117   2.964  -1.897  1.00 30.00           C",
            "ATOM      4  O   MET A   1      -6.634   1.849  -1.972  1.00 30.00           O",
            "HETATM    5  N   MSE A   2      -6.379   4.073  -2.041  1.00 30.00           N",
            "HETATM    6  CA  MSE A   2      -4.923   4.075  -2.305  1.00 30.00           C",
            "HETATM    7  C   MSE A   2      -4.633   3.541  -3.713  1.00 30.00           C",
            "HETATM    8  O   MSE A   2      -5.531   3.124  -4.444  1.00 30.00           O",
            "ATOM      9  N   GLY A   3      -3.348   3.560  -4.085  1.00 30.00           N",
            "ATOM     10  CA  GLY A   3      -2.913   3.062  -5.394  1.00 30.00           C",
            "ATOM     11  C   GLY A   3      -3.424   3.948  -6.524  1.00 30.00           C",
            "ATOM     12  O   GLY A   3      -3.793   5.105  -6.317  1.00 30.00           O",
            "END",
            "",
        ]
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb)
        path = f.name
    structure = Structure.from_file(path)
    # Length-3 sequence (matching the model-visible count) must validate.
    pair = ESMIF1ScoringPair(sequence="MMG", structure=structure, target_chain="A")
    assert pair.sequence == "MMG"
    # Length-2 (treating MSE as dropped) must NOT validate — MSE is model-visible.
    with pytest.raises(ValueError, match="does not match target chain"):
        ESMIF1ScoringPair(sequence="MG", structure=structure, target_chain="A")


def test_esm_if1_score_ligand_dropped_from_chain_length():
    """A HETATM ligand residue (e.g. HEM) the model can't decode is dropped from the count."""
    pdb = "\n".join(
        [
            "ATOM      1  N   MET A   1      -8.901   4.127  -0.555  1.00 30.00           N",
            "ATOM      2  CA  MET A   1      -8.608   3.135  -1.618  1.00 30.00           C",
            "ATOM      3  C   MET A   1      -7.117   2.964  -1.897  1.00 30.00           C",
            "ATOM      4  O   MET A   1      -6.634   1.849  -1.972  1.00 30.00           O",
            "ATOM      5  N   ALA A   2      -6.379   4.073  -2.041  1.00 30.00           N",
            "ATOM      6  CA  ALA A   2      -4.923   4.075  -2.305  1.00 30.00           C",
            "ATOM      7  C   ALA A   2      -4.633   3.541  -3.713  1.00 30.00           C",
            "ATOM      8  O   ALA A   2      -5.531   3.124  -4.444  1.00 30.00           O",
            "HETATM    9  N   HEM A   3       7.000   5.000   5.000  1.00 30.00           N",
            "HETATM   10  CA  HEM A   3       8.000   5.000   5.000  1.00 30.00           C",
            "HETATM   11  C   HEM A   3       9.000   5.000   5.000  1.00 30.00           C",
            "HETATM   12  O   HEM A   3      10.000   5.000   5.000  1.00 30.00           O",
            "END",
            "",
        ]
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb)
        path = f.name
    structure = Structure.from_file(path)
    pair = ESMIF1ScoringPair(sequence="MA", structure=structure, target_chain="A")
    assert pair.sequence == "MA"


def test_esm_if1_score_unknown_target_chain(multichain_structure: Structure):
    """Naming a chain that isn't in the structure raises a clear error at pair construction."""
    with pytest.raises(ValueError, match="not in the structure's chains"):
        ESMIF1ScoringPair(sequence="AAA", structure=multichain_structure, target_chain="Z")


def test_esm_if1_score_target_chain_is_single_chain_selection(multichain_structure: Structure):
    """target_chain is a SingleChainSelection.

    A bare string coerces, an explicit object is equivalent, and a list is a
    usage error pointing at ChainSelection.
    """
    seq = multichain_structure.get_chain_sequence("A")
    coerced = ESMIF1ScoringPair(sequence=seq, structure=multichain_structure, target_chain="A")
    assert isinstance(coerced.target_chain, SingleChainSelection)
    assert coerced.target_chain.chain == "A"
    explicit = ESMIF1ScoringPair(
        sequence=seq, structure=multichain_structure, target_chain=SingleChainSelection(chain="A")
    )
    assert explicit.target_chain == coerced.target_chain
    with pytest.raises(ValueError, match="single chain"):
        ESMIF1ScoringPair(sequence=seq, structure=multichain_structure, target_chain=["A", "B"])


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("esm-if1-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm_if1_sample_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark esm-if1-sample: 50 ProteinDPO complexes of renin (~340 aa) at batch_size=16 (cold + warm)."""
    inputs = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ESMIF1SampleConfig(
        num_sequences_per_structure=50,
        batch_size=16,
        temperature=0.1,
        seed=0,
    )

    result = benchmark_twice(request, "esm_if1", lambda: run_esm_if1_sample(inputs, config))

    assert result.tool_id == "esm-if1-sample"
    assert len(result.design_sets) == 1
    design_set = result.design_sets[0]
    assert len(design_set.complexes) == 50
    target_len = len(pdb_structure.get_chain_sequence("A"))
    for design in design_set.complexes:
        assert len(design.designed_chains[0].sequence) == target_len
    assert all(np.isfinite(d.metrics["log_likelihood"]) for d in design_set.complexes)


@pytest.mark.benchmark("esm-if1-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esm_if1_score_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark esm-if1-score on 50 sequence-structure pairs against renin (cold + warm)."""
    target_len = len(pdb_structure.get_chain_sequence("A"))
    sequences = random_protein_sequences(n=50, length=target_len, seed=1)
    pairs = [ESMIF1ScoringPair(sequence=s, structure=pdb_structure) for s in sequences]

    inputs = ESMIF1ScoringInput(sequence_structure_pairs=pairs)
    config = ESMIF1ScoringConfig()

    result = benchmark_twice(request, "esm_if1", lambda: run_esm_if1_score(inputs, config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "esm-if1-score"
    assert len(result.scores) == 50
    for score in result.scores:
        assert score["perplexity"] >= 1.0
