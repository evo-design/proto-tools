"""tests/inverse_folding_tests/test_proteinmpnn.py.

Tests for ProteinMPNN sampling and scoring.
"""

import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from proto_tools.entities.complex import Chain
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.proteinmpnn import (
    ProteinMPNNGradientConfig,
    ProteinMPNNGradientInput,
    ProteinMPNNSampleConfig,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    run_proteinmpnn_gradient,
    run_proteinmpnn_sample,
    run_proteinmpnn_score,
)
from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample import (
    ProteinMPNNDesign,
    ProteinMPNNDesignMetrics,
)
from proto_tools.tools.inverse_folding.proteinmpnn.standalone.inference import (
    ALPHAFOLD_VOCAB,
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
_CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
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
ATOM     13  N   VAL A   4       9.377   5.660   0.000  1.00  0.00           N
ATOM     14  CA  VAL A   4      10.044   6.955   0.000  1.00  0.00           C
ATOM     15  C   VAL A   4      11.548   6.820   0.000  1.00  0.00           C
ATOM     16  O   VAL A   4      12.101   5.720   0.000  1.00  0.00           O
ATOM     17  N   LEU A   5      12.207   7.978   0.000  1.00  0.00           N
ATOM     18  CA  LEU A   5      13.655   8.068   0.000  1.00  0.00           C
ATOM     19  C   LEU A   5      14.195   9.485   0.000  1.00  0.00           C
ATOM     20  O   LEU A   5      13.424  10.440   0.000  1.00  0.00           O
END
"""


_persistent_tool = make_persistent_fixture("proteinmpnn")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure.from_file(TEST_PDB_FILE)


def _small_structure() -> Structure:
    return Structure(structure=_SMALL_PDB)


def test_proteinmpnn_sample_backbone_noise_uses_model_constructor(monkeypatch, tmp_path):
    """ColabDesign's mk_mpnn_model has no set_opt; backbone_noise is a constructor arg."""
    from proto_tools.tools.inverse_folding.proteinmpnn.standalone import inference as proteinmpnn_inference

    calls: list[float] = []

    class FakeMpnnModel:
        def __init__(self, *, model_name, weights, backbone_noise=0.0):
            calls.append(backbone_noise)
            self._model = SimpleNamespace(params={"p": np.array([1.0])})

        def prep_inputs(self, *args, **kwargs):
            return None

        def sample_parallel(self, *args, **kwargs):
            return {
                "seq": np.array(["AGSVL"]),
                "score": np.array([0.0]),
                "seqid": np.array([1.0]),
                "logits": np.zeros((1, 5, len(ALPHAFOLD_VOCAB))),
            }

    fake_colabdesign = ModuleType("colabdesign")
    fake_mpnn = ModuleType("colabdesign.mpnn")
    fake_mpnn.mk_mpnn_model = FakeMpnnModel
    fake_colabdesign.mpnn = fake_mpnn
    monkeypatch.setitem(sys.modules, "colabdesign", fake_colabdesign)
    monkeypatch.setitem(sys.modules, "colabdesign.mpnn", fake_mpnn)

    standalone_helpers = sys.modules["standalone_helpers"]
    monkeypatch.setattr(standalone_helpers, "set_jax_seed", lambda seed: object())
    monkeypatch.setattr(
        proteinmpnn_inference.ProteinMPNNModel,
        "to_device",
        lambda self, device: setattr(self, "device", device),
    )
    monkeypatch.setattr(proteinmpnn_inference.ProteinMPNNModel, "unload", lambda self: None)

    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("END\n")
    model = proteinmpnn_inference.ProteinMPNNModel()

    noisy = model.sample(str(pdb_path), ["A"], batch_size=1, seed=1, device="cpu", backbone_noise=0.02)
    assert noisy["seq"].tolist() == ["AGSVL"]
    assert calls == [0.02]

    model.sample(str(pdb_path), ["A"], batch_size=1, seed=2, device="cpu", backbone_noise=0.0)
    assert calls == [0.02, 0.0]


@pytest.mark.uses_gpu
def test_proteinmpnn_sample_simple(pdb_structure: Structure):
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=pdb_structure),
            InverseFoldingStructureInput(structure=pdb_structure),
            InverseFoldingStructureInput(structure=pdb_structure),
        ]
    )
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=10,
        temperature=1.0,
        seed=42,
    )
    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    validate_output(output)
    assert output.tool_id == "proteinmpnn-sample"

    for design_set in output.design_sets:
        assert len(design_set.complexes) == 10
        for design in design_set.complexes:
            assert isinstance(design.chains[0].sequence, str)
            assert isinstance(design.metrics["perplexity"], float)
            assert isinstance(design.metrics["sequence_recovery"], float)


@pytest.mark.uses_gpu
def test_proteinmpnn_sample_chunked_batching(pdb_structure: Structure):
    """Chunked batching produces the correct number of sequences."""
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=6,
        batch_size=2,
        temperature=0.1,
        seed=42,
    )
    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"Chunked batching failed: {output}"

    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 6
    for design in design_set.complexes:
        seq = design.chains[0].sequence
        assert isinstance(seq, str)
        assert len(seq) > 0
        assert isinstance(design.metrics["perplexity"], float)
        assert isinstance(design.metrics["sequence_recovery"], float)


@pytest.mark.uses_gpu
def test_proteinmpnn_sample_advanced_args(pdb_structure: Structure):
    """Fixed positions and excluded amino acids are respected."""
    chain_A = pdb_structure.get_chain_sequence("A")

    # Find all indices of the amino acid "C" in chain A
    c_positions = [i + 1 for i, aa in enumerate(chain_A) if aa == "C"]

    # Make a list of fixed indices that do not contain the "C" positions
    fixed_positions = random.sample(list(set(np.arange(len(chain_A)) + 1) - set(c_positions)), 200)

    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=pdb_structure, fixed_positions={"A": fixed_positions}),
            InverseFoldingStructureInput(structure=pdb_structure, fixed_positions={"A": fixed_positions}),
        ]
    )
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=10,
        temperature=1.0,
        seed=42,
        excluded_amino_acids=["C"],
    )

    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"Failed to sample: {output}"

    validate_output(output)

    for design_set in output.design_sets:
        for design in design_set.complexes:
            sequence = design.chains[0].sequence
            assert "C" not in sequence, f"Sequence contains excluded 'C': {sequence}"

            for position in fixed_positions:
                assert sequence[position - 1] == chain_A[position - 1], (
                    f"Position {position}: {sequence[position - 1]} != {chain_A[position - 1]}"
                )


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
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
            SequenceStructurePair(sequence=modified_sequence, structure=pdb_structure),
        ]
    )
    config = ProteinMPNNScoringConfig(fixed_positions=fixed_positions, seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)
    assert output.success, f"Failed to score: {output}"

    validate_output(output)
    assert output.tool_id == "proteinmpnn-score"
    assert output.vocab == ALPHAFOLD_VOCAB

    assert len(output.scores) == 2
    assert all(isinstance(score, InverseFoldingScoringMetrics) for score in output.scores)
    scored_len = len(original_sequence) - len(fixed_positions["A"])
    assert all(
        np.isclose(score.log_likelihood, score.avg_log_likelihood * scored_len, rtol=1e-5) for score in output.scores
    )

    # Original sequence should have lower perplexity than the modified one
    assert output.scores[0].perplexity < output.scores[1].perplexity


@pytest.mark.uses_gpu
def test_proteinmpnn_score_fields(pdb_structure: Structure):
    """All scoring fields and their mathematical relationships are correct."""
    original_sequence = pdb_structure.get_chain_sequence("A")
    seq_len = len(original_sequence)

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)
    assert output.success

    validate_output(output)
    assert_metrics_in_spec(output)

    score = output.scores[0]

    # Validate logits shape
    logits_arr = np.array(score.logits)
    assert logits_arr.ndim == 2
    assert logits_arr.shape == (seq_len, len(ALPHAFOLD_VOCAB))

    # log_likelihood = avg_log_likelihood * seq_len
    assert np.isclose(score.log_likelihood, score.avg_log_likelihood * seq_len, rtol=1e-5)

    # perplexity = exp(-avg_log_likelihood)
    assert np.isclose(score.perplexity, np.exp(-score.avg_log_likelihood), rtol=1e-5)

    assert score.avg_log_likelihood <= 0
    assert score.perplexity >= 1.0


@pytest.mark.uses_gpu
def test_proteinmpnn_score_vocab(pdb_structure: Structure):
    """Vocab property on output matches ALPHAFOLD_VOCAB."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42)
    output = run_proteinmpnn_score(inp, config)
    assert output.success

    validate_output(output)

    assert output.vocab == ALPHAFOLD_VOCAB
    assert output.scores[0].vocab == ALPHAFOLD_VOCAB


def test_proteinmpnn_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        n = len(payload["logits"])
        return {
            "gradient": [[0.0] * 20] * n,
            "loss": 0.5,
            "metrics": {"avg_log_likelihood": -0.5, "perplexity": np.exp(0.5)},
            "vocab": list(_CANONICAL_AA),
        }

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_proteinmpnn_gradient(
        ProteinMPNNGradientInput(
            logits=[[0.0] * 20] * 5,
            structure=_small_structure(),
            chains_to_redesign=["A"],
            fixed_positions={"A": [1]},
        ),
        ProteinMPNNGradientConfig(use_ste=False, device="cpu"),
    )

    assert captured["toolkit"] == "proteinmpnn"
    assert captured["payload"]["operation"] == "compute_gradient"
    assert captured["payload"]["chain_ids"] == ["A"]
    assert captured["payload"]["fixed_positions"] == {"A": [1]}
    assert captured["payload"]["use_ste"] is False
    assert captured["payload"]["compute_gradient"] is True
    assert result.gradient == [[0.0] * 20] * 5


def test_proteinmpnn_sample_seeds_distinct_across_inputs(monkeypatch):
    """Per-input seed should differ — previously every input got base_seed + 0."""
    captured_seeds: list[int] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_seeds.append(payload["seed"])
        bs = payload["batch_size"]
        return {"seq": ["A" * 5] * bs, "score": [-1.0] * bs, "seqid": [1.0] * bs}

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample.ToolInstance.dispatch",
        fake_dispatch,
    )

    structure = _small_structure()
    run_proteinmpnn_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure) for _ in range(3)]),
        ProteinMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=42, device="cpu"),
    )

    assert len(captured_seeds) == 3
    assert len(set(captured_seeds)) == 3

    # Same base seed → same per-input seed stream (deterministic given config.seed).
    captured_seeds_again: list[int] = []

    def fake_dispatch_again(toolkit, payload, *, instance=None, config=None):
        captured_seeds_again.append(payload["seed"])
        bs = payload["batch_size"]
        return {"seq": ["A" * 5] * bs, "score": [-1.0] * bs, "seqid": [1.0] * bs}

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_sample.ToolInstance.dispatch",
        fake_dispatch_again,
    )
    run_proteinmpnn_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure) for _ in range(3)]),
        ProteinMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=42, device="cpu"),
    )
    assert captured_seeds == captured_seeds_again


def test_proteinmpnn_multichain_design_structure():
    """A multi-chain ProteinMPNNDesign exposes chain order, redesigned flags, and metrics."""
    design = ProteinMPNNDesign(
        chains=[
            Chain(id="A", sequence="MKTL"),
            Chain(id="B", sequence="GGSS"),
        ],
        designed=[True, False],
        metrics=ProteinMPNNDesignMetrics(perplexity=1.5, sequence_recovery=0.4),
    )

    # Chain ids preserve input order.
    assert [c.id for c in design.chains] == ["A", "B"]
    assert design.designed == [True, False]

    # Chain-level helpers.
    assert design.chain_sequences == ["MKTL", "GGSS"]
    assert design.as_chain_map() == {"A": "MKTL", "B": "GGSS"}
    assert [c.id for c in design.designed_chains] == ["A"]

    # Metric access via mapping, attribute, and primary value.
    assert design.metrics["perplexity"] == 1.5
    assert design.metrics.perplexity == 1.5
    assert design.metrics["sequence_recovery"] == 0.4
    assert design.metrics.primary_value == 1.5
    assert design.design_metrics() == {"perplexity": 1.5, "sequence_recovery": 0.4}


def test_proteinmpnn_design_feeds_structure_predictor():
    """A ProteinMPNNDesign feeds an SP tool input directly via LSP (subclass-as-Complex)."""
    from proto_tools.tools.structure_prediction import ESMFoldInput

    design = ProteinMPNNDesign(
        chains=[
            Chain(id="A", sequence="MKTL"),
            Chain(id="B", sequence="GGSS"),
        ],
        designed=[True, False],
        metrics=ProteinMPNNDesignMetrics(perplexity=1.5, sequence_recovery=0.4),
    )

    inp = ESMFoldInput(complexes=[design])
    assert inp.complexes[0] is design
    assert [c.sequence for c in inp.complexes[0].chains] == ["MKTL", "GGSS"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_proteinmpnn_gradient_end_to_end():
    """ProteinMPNN gradient returns finite non-zero gradients for a small backbone."""
    result = run_proteinmpnn_gradient(
        ProteinMPNNGradientInput(
            logits=[[0.0] * 20] * 5,
            structure=_small_structure(),
            chains_to_redesign=["A"],
            temperature=0.7,
        ),
        ProteinMPNNGradientConfig(seed=42),
    )

    validate_output(result)
    assert result.tool_id == "proteinmpnn-gradient"
    assert result.gradient is not None
    assert len(result.gradient) == 5
    assert all(len(row) == 20 for row in result.gradient)
    assert all(np.isfinite(v) for row in result.gradient for v in row)
    assert any(v != 0.0 for row in result.gradient for v in row)
    assert result.loss > 0.0
    assert result.metrics["avg_log_likelihood"] == pytest.approx(-result.loss, rel=1e-6)
    assert result.metrics["perplexity"] == pytest.approx(np.exp(result.loss), rel=1e-6)
    assert result.vocab == list(_CANONICAL_AA)


@pytest.mark.uses_gpu
def test_proteinmpnn_score_single_pair(pdb_structure: Structure):
    """Scoring with a single sequence-structure pair succeeds."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
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
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
            SequenceStructurePair(sequence=modified_1, structure=pdb_structure),
            SequenceStructurePair(sequence=modified_2, structure=pdb_structure),
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
        assert "log_likelihood" in score
        assert "avg_log_likelihood" in score
        assert "perplexity" in score


@pytest.mark.uses_gpu
def test_proteinmpnn_score_cache(pdb_structure: Structure):
    """Caching returns consistent scores across passes."""
    from proto_tools.utils.tool_cache import (
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
                SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
                SequenceStructurePair(sequence=modified_sequence_1, structure=pdb_structure),
                SequenceStructurePair(sequence=modified_sequence_2, structure=pdb_structure),
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
                SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
                SequenceStructurePair(sequence=modified_sequence_1, structure=pdb_structure),
                SequenceStructurePair(sequence=modified_sequence_2, structure=pdb_structure),
                SequenceStructurePair(sequence=modified_sequence_3, structure=pdb_structure),
            ]
        )
        output_second_pass = run_proteinmpnn_score(input_second_pass, config)

        assert output_second_pass.success
        assert len(output_second_pass.scores) == 4
        validate_output(output_second_pass)

        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 4

        # First three scores should match exactly
        assert output_second_pass.scores[0].perplexity == output_first_pass.scores[0].perplexity
        assert output_second_pass.scores[1].perplexity == output_first_pass.scores[1].perplexity
        assert output_second_pass.scores[2].perplexity == output_first_pass.scores[2].perplexity

        # Logits arrays should be identical for cached results
        assert np.allclose(output_second_pass.scores[0].logits, output_first_pass.scores[0].logits)
        assert np.allclose(output_second_pass.scores[1].logits, output_first_pass.scores[1].logits)
        assert np.allclose(output_second_pass.scores[2].logits, output_first_pass.scores[2].logits)

    finally:
        _program_tool_cache.set(None)


@pytest.mark.uses_gpu
def test_proteinmpnn_score_logits_disabled_by_default(pdb_structure: Structure):
    """Logits are None when return_logits=False (default)."""
    original_sequence = pdb_structure.get_chain_sequence("A")

    inp = ProteinMPNNScoringInput(
        sequence_structure_pairs=[
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
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
            SequenceStructurePair(sequence=original_sequence, structure=pdb_structure),
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
    inp = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=2,
        temperature=0.1,
        seed=42,
        model_choice="abmpnn",
    )
    output = run_proteinmpnn_sample(inp, config)
    assert output.success, f"AbMPNN sampling failed: {output}"
    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 2
    assert all(isinstance(d.chains[0].sequence, str) for d in design_set.complexes)


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
    assert "avg_log_likelihood" in output.scores[0]


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("proteinmpnn-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_proteinmpnn_sample_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark proteinmpnn-sample: 50 complexes of renin (~340 aa) at batch_size=16 (cold + warm)."""
    inputs = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=pdb_structure)])
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=50,
        batch_size=16,
        temperature=0.1,
        seed=0,
    )

    result = benchmark_twice(request, "proteinmpnn", lambda: run_proteinmpnn_sample(inputs, config))

    assert result.tool_id == "proteinmpnn-sample"
    assert len(result.design_sets) == 1, "Should have one design set per input structure"
    design_set = result.design_sets[0]
    assert len(design_set.complexes) == 50, "Should have 50 complexes"
    target_len = len(pdb_structure.get_chain_sequence("A"))
    for design in design_set.complexes:
        assert len(design.chains[0].sequence) == target_len, "Designed sequence should match structure length"


@pytest.mark.benchmark("proteinmpnn-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_proteinmpnn_score_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark proteinmpnn-score on 50 sequence-structure pairs against renin (cold + warm)."""
    target_len = len(pdb_structure.get_chain_sequence("A"))
    sequences = random_protein_sequences(n=50, length=target_len, seed=1)
    pairs = [SequenceStructurePair(sequence=s, structure=pdb_structure) for s in sequences]

    inputs = ProteinMPNNScoringInput(sequence_structure_pairs=pairs)
    config = ProteinMPNNScoringConfig(seed=42, return_logits=False)

    result = benchmark_twice(request, "proteinmpnn", lambda: run_proteinmpnn_score(inputs, config))
    assert_metrics_in_spec(result)

    assert result.tool_id == "proteinmpnn-score"
    assert len(result.scores) == 50
    for score in result.scores:
        assert score["perplexity"] >= 1.0
