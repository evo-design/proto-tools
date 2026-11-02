"""tests/inverse_folding_tests/test_proteinmpnn.py.

Tests for ProteinMPNN sampling and scoring.
"""

import logging
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import proto_tools.tools.inverse_folding.proteinmpnn
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.proteinmpnn import (
    ProteinMPNNGradientCandidate,
    ProteinMPNNGradientConfig,
    ProteinMPNNGradientInput,
    ProteinMPNNHomooligomerSampleConfig,
    ProteinMPNNHomooligomerSampleInput,
    ProteinMPNNHomooligomerStructureInput,
    ProteinMPNNSampleConfig,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    run_proteinmpnn_gradient,
    run_proteinmpnn_sample,
    run_proteinmpnn_sample_homooligomer,
    run_proteinmpnn_score,
)
from proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_gradient import (
    ProteinMPNNGradientOutput,
)
from proto_tools.tools.inverse_folding.proteinmpnn.standalone.inference import (
    PROTEINMPNN_AMINO_ACIDS_WITH_X,
)
from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingInput,
    InverseFoldingScoringMetrics,
    InverseFoldingStructureInput,
    SequenceStructurePair,
)
from proto_tools.utils import ToolInstance, one_hot_protein_logits
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


_RESIDUE_TEMPLATE = """ATOM      1  N   XXX A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  XXX A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   XXX A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   XXX A   1       1.246   2.390   0.000  1.00  0.00           O
"""
_PROTOMER = ["ALA", "GLY", "SER", "VAL", "LEU"]


def _oligomer_structure(chains: dict[str, list[str]]) -> Structure:
    """Build a multi-chain structure from explicit per-chain residue names.

    Args:
        chains (dict[str, list[str]]): Chain ID to its three-letter residue names.

    Returns:
        Structure: Chains laid out side by side so each is a separate polymer.
    """
    lines: list[str] = []
    serial = 1
    for chain_id, residues in chains.items():
        for index, resname in enumerate(residues, start=1):
            for line in _RESIDUE_TEMPLATE.splitlines():
                x = float(line[30:38]) + 4.0 * index
                y = float(line[38:46]) + 8.0 * (ord(chain_id) - ord("A"))
                lines.append(
                    f"{line[:6]}{serial:5d}{line[11:17]}{resname:>3s}{line[20:21]}"
                    f"{chain_id}{index:4d}{line[26:30]}{x:8.3f}{y:8.3f}{line[46:]}"
                )
                serial += 1
    return Structure(structure="\n".join([*lines, "END", ""]))


def _homooligomer_input(n_chains: int = 2, **kwargs) -> ProteinMPNNHomooligomerSampleInput:
    """Build a tool input over an ``n_chains``-mer of identical protomers."""
    chain_ids = [chr(ord("A") + i) for i in range(n_chains)]
    structure = _oligomer_structure(dict.fromkeys(chain_ids, _PROTOMER))
    kwargs.setdefault("protomer_chains", chain_ids)
    return ProteinMPNNHomooligomerSampleInput(
        inputs=[ProteinMPNNHomooligomerStructureInput(structure=structure, **kwargs)]
    )


def _fake_dispatch(sink: list[dict], seq: str = "AGSVL"):
    """Return a ToolInstance.dispatch stand-in that records payloads and returns ``seq``."""

    def dispatch(_tool, input_dict, **_kwargs):
        sink.append(input_dict)
        return {"seq": [seq], "score": [0.5], "seqid": [0.5]}

    return staticmethod(dispatch)


# ---------------------------------------------------------------------------
# proteinmpnn-sample-homooligomer: input validation
# ---------------------------------------------------------------------------
def test_homooligomer_input_rejects_single_chain():
    """One chain has nothing to tie to."""
    structure = _oligomer_structure({"A": _PROTOMER, "B": _PROTOMER})
    with pytest.raises(ValidationError, match="at least two chains"):
        ProteinMPNNHomooligomerStructureInput(structure=structure, protomer_chains=["A"])


def test_homooligomer_input_rejects_unequal_protomer_lengths():
    """Unequal protomers cannot be tied; the message names the offending lengths."""
    structure = _oligomer_structure({"A": _PROTOMER, "B": _PROTOMER[:3]})
    with pytest.raises(ValidationError, match=r"same length.*'A': 5.*'B': 3"):
        ProteinMPNNHomooligomerStructureInput(structure=structure, protomer_chains=["A", "B"])


def test_homooligomer_input_rejects_out_of_range_fixed_positions():
    """Fixed positions are 1-indexed within one protomer, not across the assembly."""
    structure = _oligomer_structure({"A": _PROTOMER, "B": _PROTOMER})
    with pytest.raises(ValidationError, match=r"\[7\] fall outside the 1\.\.5 protomer"):
        ProteinMPNNHomooligomerStructureInput(
            structure=structure, protomer_chains=["A", "B"], protomer_fixed_positions=[2, 7]
        )


def test_homooligomer_input_warns_but_accepts_pseudo_symmetric_protomers(caplog):
    """Equal-length protomers that differ in sequence are legal but worth flagging."""
    structure = _oligomer_structure({"A": _PROTOMER, "B": ["TRP"] * 5})
    with caplog.at_level(logging.WARNING):
        inp = ProteinMPNNHomooligomerStructureInput(structure=structure, protomer_chains=["A", "B"])
    assert inp.protomer_chain_ids == ["A", "B"]
    assert "pseudo-symmetric" in caplog.text


def test_homooligomer_fixed_positions_broadcast_to_every_protomer():
    """Per-protomer positions are the schema; broadcasting them is this property's job."""
    structure = _oligomer_structure(dict.fromkeys("ABC", _PROTOMER))
    inp = ProteinMPNNHomooligomerStructureInput(
        structure=structure, protomer_chains=["A", "B", "C"], protomer_fixed_positions=[3, 1, 3]
    )
    assert inp.fixed_positions.chains == {"A": [1, 3], "B": [1, 3], "C": [1, 3]}


def test_homooligomer_input_has_no_per_chain_fixed_positions_field():
    """The asymmetric-selection footgun is meant to be unrepresentable, not validated."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProteinMPNNHomooligomerStructureInput(
            structure=_oligomer_structure(dict.fromkeys("AB", _PROTOMER)),
            protomer_chains=["A", "B"],
            fixed_positions={"A": [2]},
        )


# ---------------------------------------------------------------------------
# proteinmpnn-sample-homooligomer: run behavior
# ---------------------------------------------------------------------------
def test_homooligomer_run_ties_and_broadcasts_the_protomer_sequence(monkeypatch):
    """Tied decoding returns one segment, which must land on every protomer chain."""
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _fake_dispatch(sent))

    output = run_proteinmpnn_sample_homooligomer(
        _homooligomer_input(3, protomer_fixed_positions=[2]),
        ProteinMPNNHomooligomerSampleConfig(num_sequences_per_structure=1, seed=42),
    )

    assert sent[0]["homooligomer"] is True
    assert sent[0]["fixed_positions"] == {"A": [2], "B": [2], "C": [2]}
    design = output.design_sets[0].complexes[0]
    assert [chain.sequence for chain in design.chains] == ["AGSVL"] * 3


def test_homooligomer_run_rejects_a_mismatched_segment_count(monkeypatch):
    """A worker returning neither one segment nor one per chain is a hard error."""
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _fake_dispatch(sent, seq="AGSVL/AGSVL"))

    with pytest.raises(ValueError, match="returned 2 sequence segment"):
        run_proteinmpnn_sample_homooligomer(
            _homooligomer_input(3),
            ProteinMPNNHomooligomerSampleConfig(num_sequences_per_structure=1, seed=42),
        )


# ---------------------------------------------------------------------------
# proteinmpnn-sample: the nudge toward the dedicated tool
# ---------------------------------------------------------------------------
def test_sample_warns_when_the_redesigned_chains_are_a_homooligomer(monkeypatch, caplog):
    """The general tool designs chains independently, which is rarely meant for an oligomer."""
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _fake_dispatch(sent, seq="AGSVL/AGSVL"))

    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=_oligomer_structure(dict.fromkeys("AB", _PROTOMER)),
                chains_to_redesign=["A", "B"],
            )
        ]
    )
    with caplog.at_level(logging.WARNING):
        run_proteinmpnn_sample(inp, ProteinMPNNSampleConfig(num_sequences_per_structure=1, seed=42))

    assert "proteinmpnn-sample-homooligomer" in caplog.text
    # The general tool must still design independently, never tie.
    assert sent[0]["homooligomer"] is False


def test_sample_does_not_warn_for_equal_length_but_different_chains(monkeypatch, caplog):
    """Matching lengths alone are not an oligomer; warning on them would be noise."""
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _fake_dispatch(sent, seq="AGSVL/AGSVL"))

    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(
                structure=_oligomer_structure({"A": _PROTOMER, "B": ["TRP"] * 5}),
                chains_to_redesign=["A", "B"],
            )
        ]
    )
    with caplog.at_level(logging.WARNING):
        run_proteinmpnn_sample(inp, ProteinMPNNSampleConfig(num_sequences_per_structure=1, seed=42))

    assert "proteinmpnn-sample-homooligomer" not in caplog.text


def test_proteinmpnn_sample_backbone_noise_uses_model_constructor(monkeypatch, tmp_path):
    """ColabDesign's mk_mpnn_model has no set_opt; backbone_noise is a constructor arg."""
    from proto_tools.tools.inverse_folding.proteinmpnn.standalone import inference as proteinmpnn_inference

    calls: list[float] = []
    prep_calls: list[dict] = []

    class FakeMpnnModel:
        def __init__(self, *, model_name, weights, backbone_noise=0.0):
            calls.append(backbone_noise)
            self._model = SimpleNamespace(params={"p": np.array([1.0])})

        def prep_inputs(self, *args, **kwargs):
            prep_calls.append(kwargs)
            return

        def sample_parallel(self, *args, **kwargs):
            return {
                "seq": np.array(["AGSVL"]),
                "score": np.array([0.0]),
                "seqid": np.array([1.0]),
                "logits": np.zeros((1, 5, len(PROTEINMPNN_AMINO_ACIDS_WITH_X))),
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

    noisy = model.sample(
        str(pdb_path),
        ["A"],
        batch_size=1,
        seed=1,
        device="cpu",
        backbone_noise=0.02,
        homooligomer=True,
    )
    assert noisy["seq"].tolist() == ["AGSVL"]
    assert calls == [0.02]
    assert prep_calls[0]["homooligomer"] is True

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
            SequenceStructurePair(
                sequence=original_sequence,
                structure=pdb_structure,
                fixed_positions=fixed_positions,
            ),
            SequenceStructurePair(
                sequence=modified_sequence,
                structure=pdb_structure,
                fixed_positions=fixed_positions,
            ),
        ]
    )
    config = ProteinMPNNScoringConfig(seed=42, return_logits=True)
    output = run_proteinmpnn_score(inp, config)
    assert output.success, f"Failed to score: {output}"

    validate_output(output)
    assert output.tool_id == "proteinmpnn-score"
    assert output.vocab == list(_CANONICAL_AA)

    assert len(output.scores) == 2
    assert all(isinstance(score, InverseFoldingScoringMetrics) for score in output.scores)
    scored_len = len(original_sequence) - len(fixed_positions["A"])
    assert all(
        np.isclose(score.log_likelihood, score.avg_log_likelihood * scored_len, rtol=1e-5) for score in output.scores
    )

    # Original sequence should have lower perplexity than the modified one
    assert output.scores[0].perplexity < output.scores[1].perplexity


def test_proteinmpnn_score_logits_land_in_proto_columns() -> None:
    """The native-to-proto permutation moves each residue's value to its own proto column.

    ColabDesign scores in AlphaFold order — the 20 residues sorted by three-letter code — with a
    trailing unknown column. That order is derived here from an external table rather than read
    from the toolkit, so transposing two residues in the toolkit's own alphabet fails this test
    instead of moving the expectation with it. Every value is distinguishable, so a swap surfaces
    as a wrong residue rather than a plausible-looking matrix.
    """
    from Bio.Data.IUPACData import protein_letters_1to3

    from proto_tools.tools.inverse_folding.proteinmpnn.standalone.inference import (
        PROTEINMPNN_TO_CANONICAL_INDICES,
    )

    native_order = [*sorted(_CANONICAL_AA, key=lambda residue: protein_letters_1to3[residue].upper()), "X"]
    assert native_order == PROTEINMPNN_AMINO_ACIDS_WITH_X, (
        f"the toolkit's alphabet {''.join(PROTEINMPNN_AMINO_ACIDS_WITH_X)} is not AlphaFold's "
        f"three-letter-code order {''.join(native_order)}"
    )

    # Column j of the model's output carries the value j, so a column's value names its origin.
    native = np.tile(np.arange(len(native_order), dtype=float), (3, 1))

    proto_ordered = native[..., PROTEINMPNN_TO_CANONICAL_INDICES]

    assert proto_ordered.shape == (3, len(_CANONICAL_AA))
    for column, residue in enumerate(_CANONICAL_AA):
        expected_native_column = native_order.index(residue)
        assert proto_ordered[0, column] == expected_native_column, (
            f"proto column {column} ({residue}) carries native column {proto_ordered[0, column]:.0f}, "
            f"expected {expected_native_column}"
        )
    assert float(native_order.index("X")) not in set(proto_ordered[0]), "the unknown column reached the output"


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
    assert logits_arr.shape == (seq_len, len(_CANONICAL_AA))

    # log_likelihood = avg_log_likelihood * seq_len
    assert np.isclose(score.log_likelihood, score.avg_log_likelihood * seq_len, rtol=1e-5)

    # perplexity = exp(-avg_log_likelihood)
    assert np.isclose(score.perplexity, np.exp(-score.avg_log_likelihood), rtol=1e-5)

    assert score.avg_log_likelihood <= 0
    assert score.perplexity >= 1.0


def test_proteinmpnn_gradient_dispatch_contract(monkeypatch):
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {
            "results": [
                {
                    "gradient": [[0.0] * 20] * len(item["logits"]),
                    "loss": 0.5,
                    "metrics": {"avg_log_likelihood": -0.5, "perplexity": np.exp(0.5)},
                }
                for item in payload["items"]
            ],
            "vocab": list(_CANONICAL_AA),
        }

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn.proteinmpnn_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_proteinmpnn_gradient(
        ProteinMPNNGradientInput(
            inputs=ProteinMPNNGradientCandidate(
                logits=[[0.0] * 20] * 5,
                structure=_small_structure(),
                chains_to_redesign=["A"],
                fixed_positions={"A": [1]},
            ),
        ),
        ProteinMPNNGradientConfig(hard=1.0, device="cpu"),
    )

    assert captured["toolkit"] == "proteinmpnn"
    assert captured["payload"]["operation"] == "compute_gradient"
    (item,) = captured["payload"]["items"]
    assert item["chain_ids"] == ["A"]
    assert item["fixed_positions"] == {"A": [1]}
    assert captured["payload"]["soft"] == 1.0
    assert captured["payload"]["hard"] == 1.0
    assert captured["payload"]["compute_gradient"] is True
    assert result.results[0].gradient == [[0.0] * 20] * 5


def test_proteinmpnn_sample_lays_design_over_structure_chains(monkeypatch):
    """The designed chain takes the model sequence; DNA context and modifications come from the structure."""

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        bs = payload["batch_size"]
        return {"seq": ["MKTAYIAK"] * bs, "score": [-1.0] * bs, "seqid": [1.0] * bs}

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn._sampling.ToolInstance.dispatch",
        fake_dispatch,
    )
    structure = Structure.from_file(Path(__file__).parent.parent / "dummy_data" / "modified_gapped_complex.pdb")
    output = run_proteinmpnn_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure)]),
        ProteinMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
    )

    design = output.design_sets[0].complexes[0]
    assert design.chain_sequence_map() == {"A": "MKTAYIAK", "B": "AGC"}
    assert design.chains[1] == structure.get_chain("B")
    assert design.designed == [True, False]


def test_proteinmpnn_sample_seeds_distinct_across_inputs(monkeypatch):
    """Per-input seed should differ — previously every input got base_seed + 0."""
    captured_seeds: list[int] = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_seeds.append(payload["seed"])
        bs = payload["batch_size"]
        return {"seq": ["A" * 5] * bs, "score": [-1.0] * bs, "seqid": [1.0] * bs}

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn._sampling.ToolInstance.dispatch",
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
        "proto_tools.tools.inverse_folding.proteinmpnn._sampling.ToolInstance.dispatch",
        fake_dispatch_again,
    )
    run_proteinmpnn_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure) for _ in range(3)]),
        ProteinMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=42, device="cpu"),
    )
    assert captured_seeds == captured_seeds_again


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_proteinmpnn_gradient_forward_only_returns_loss_without_a_gradient() -> None:
    """compute_gradient=False returns the loss, metrics and vocabulary with gradient=None."""
    logits = np.random.default_rng(0).normal(size=(5, 20)).tolist()

    result = run_proteinmpnn_gradient(
        ProteinMPNNGradientInput(
            inputs=ProteinMPNNGradientCandidate(
                logits=logits,
                structure=_small_structure(),
                chains_to_redesign=["A"],
            ),
            temperature=0.7,
        ),
        ProteinMPNNGradientConfig(seed=42, compute_gradient=False),
    )

    validate_output(result)
    assert result.results[0].gradient is None
    assert result.results[0].loss > 0.0
    assert result.results[0].metrics["avg_log_likelihood"] == pytest.approx(-result.results[0].loss, rel=1e-6)
    assert result.vocab == list(_CANONICAL_AA)


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("soft", "hard"),
    [(1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.5, 0.5)],
    ids=["default", "straight_through", "raw_input", "blended"],
)
def test_proteinmpnn_gradient_end_to_end(soft: float, hard: float):
    """ProteinMPNN gradient returns finite non-zero gradients at every relaxation setting.

    The raw-input and intermediate blends are covered because they must be functional, not
    because they are recommended: feeding unnormalized values to the model is outside its
    training distribution.
    """
    # Non-uniform, so argmax has no ties and the raw-input branch is not an all-zero state.
    logits = np.random.default_rng(0).normal(size=(5, 20)).tolist()

    result = run_proteinmpnn_gradient(
        ProteinMPNNGradientInput(
            inputs=ProteinMPNNGradientCandidate(
                logits=logits,
                structure=_small_structure(),
                chains_to_redesign=["A"],
            ),
            temperature=0.7,
        ),
        ProteinMPNNGradientConfig(seed=42, soft=soft, hard=hard),
    )

    validate_output(result)
    assert result.tool_id == "proteinmpnn-gradient"
    assert result.results[0].gradient is not None
    assert len(result.results[0].gradient) == 5
    assert all(len(row) == 20 for row in result.results[0].gradient)
    assert all(np.isfinite(v) for row in result.results[0].gradient for v in row)
    assert any(v != 0.0 for row in result.results[0].gradient for v in row)
    assert result.results[0].loss > 0.0
    assert result.results[0].metrics["avg_log_likelihood"] == pytest.approx(-result.results[0].loss, rel=1e-6)
    assert result.results[0].metrics["perplexity"] == pytest.approx(np.exp(result.results[0].loss), rel=1e-6)
    assert result.vocab == list(_CANONICAL_AA)


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


@pytest.mark.benchmark("proteinmpnn-sample-homooligomer")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_proteinmpnn_sample_homooligomer_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark proteinmpnn-sample-homooligomer: 50 tied designs of a 2x65 aa homodimer."""
    fixture = Path(proto_tools.tools.inverse_folding.proteinmpnn.__file__).parent / "example_homooligomer_fixture.pdb"
    structure = Structure.from_file(fixture)
    inputs = ProteinMPNNHomooligomerSampleInput(
        inputs=[ProteinMPNNHomooligomerStructureInput(structure=structure, protomer_chains=["A", "B"])]
    )
    config = ProteinMPNNHomooligomerSampleConfig(
        num_sequences_per_structure=50,
        batch_size=16,
        temperature=0.1,
        seed=0,
    )

    result = benchmark_twice(request, "proteinmpnn", lambda: run_proteinmpnn_sample_homooligomer(inputs, config))

    assert result.tool_id == "proteinmpnn-sample-homooligomer"
    assert len(result.design_sets) == 1, "Should have one design set per input structure"
    design_set = result.design_sets[0]
    assert len(design_set.complexes) == 50, "Should have 50 complexes"
    protomer_len = len(structure.get_chain_sequence("A"))
    for design in design_set.complexes:
        sequences = [chain.sequence for chain in design.chains]
        assert len(set(sequences)) == 1, "Tied decoding must give every protomer the same sequence"
        assert len(sequences[0]) == protomer_len, "Designed protomer should match the protomer length"


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


@pytest.mark.benchmark("proteinmpnn-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_proteinmpnn_gradient_benchmark(request: pytest.FixtureRequest, pdb_structure: Structure) -> None:
    """Benchmark proteinmpnn-gradient: 20 sequential backward passes over renin (~340 aa) (cold + warm)."""
    target_len = len(pdb_structure.get_chain_sequence("A"))
    sequence = random_protein_sequences(n=1, length=target_len, seed=7)[0]
    inputs = ProteinMPNNGradientInput(
        inputs=ProteinMPNNGradientCandidate(
            logits=one_hot_protein_logits(sequence, sharpness=2.0),
            structure=pdb_structure,
            chains_to_redesign=["A"],
        ),
        temperature=0.7,
    )
    config = ProteinMPNNGradientConfig(seed=42)

    def run_many() -> ProteinMPNNGradientOutput:
        result = None
        for _ in range(20):
            result = run_proteinmpnn_gradient(inputs, config)
        return result

    result = benchmark_twice(request, "proteinmpnn", run_many)

    validate_output(result)
    assert result.tool_id == "proteinmpnn-gradient"
    assert result.results[0].gradient is not None
    assert len(result.results[0].gradient) == target_len
    assert all(len(row) == 20 for row in result.results[0].gradient)
    assert all(np.isfinite(v) for row in result.results[0].gradient for v in row)
    assert result.results[0].loss > 0.0
