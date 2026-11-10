"""tests/inverse_folding_tests/test_readable_position_numbering.py.

Translating positions between the structure's numbering and each model's own.

A model that skips residues numbers what it reads from 1 with no gaps, so its numbering and
the structure's diverge after the first skipped residue. Every boundary that carries a
position across has to translate, and getting it wrong is silent: the neighbouring residue
is fixed, scored or reported instead of the one the caller named. These pin each boundary
with GFP, whose chromophore at position 64 is skipped by every model. CPU-only, stubbed
dispatch, no model execution.
"""

import logging
import re
from pathlib import Path

import pytest

from proto_tools.entities.structures.examples import get_gfp_structure
from proto_tools.tools.inverse_folding.esm_if1._readable import esm_if1_readable_positions
from proto_tools.tools.inverse_folding.fampnn._readable import fampnn_readable_positions
from proto_tools.tools.inverse_folding.shared_data_models import (
    flat_readable_positions,
    to_model_numbering,
    unreadable_positions,
)
from proto_tools.utils import ToolInstance

CHROMOPHORE = 64
GFP_LENGTH = 228


# ── the translations themselves ─────────────────────────────────────────────


def test_flat_positions_step_over_a_skipped_residue():
    """A structure-wide position counts every polymer residue, skipped ones included."""
    structure = get_gfp_structure()
    flat = flat_readable_positions(esm_if1_readable_positions(structure), structure)
    assert len(flat) == GFP_LENGTH - 1
    assert flat[:3] == [1, 2, 3]
    assert flat[62:65] == [63, 65, 66], "the chromophore is stepped over, not renumbered away"
    assert flat[-1] == GFP_LENGTH


def test_flat_positions_offset_each_chain_by_the_ones_before_it():
    """The second chain's positions continue the count rather than restarting at 1.

    Both FAMPNN scoring tools key results by a structure-wide position and accept multi-chain
    input, so an unoffset second chain would collide with the first.
    """
    from tests.inverse_folding_tests.test_readable_positions_all_toolkits import BACKBONE, _structure

    structure = _structure(
        {
            "A": [("ALA", False, BACKBONE), ("LIG", True, BACKBONE), ("GLY", False, BACKBONE)],
            "B": [("ALA", False, BACKBONE), ("LIG", True, BACKBONE), ("GLY", False, BACKBONE)],
        }
    )
    readable = esm_if1_readable_positions(structure)
    assert readable == {"A": [1, 3], "B": [1, 3]}, "each chain is numbered within itself"
    assert flat_readable_positions(readable, structure) == [1, 3, 4, 6], "chain B is offset by chain A"


def test_model_numbering_is_the_inverse_of_indexing_into_readable():
    """Round-tripping a position through both directions returns it unchanged."""
    structure = get_gfp_structure()
    readable = esm_if1_readable_positions(structure)["A"]
    for position in (1, 63, 65, 100, GFP_LENGTH):
        model_position = to_model_numbering(readable, [position])[0]
        assert readable[model_position - 1] == position


def test_model_numbering_drops_a_position_the_model_cannot_see():
    """There is no counterpart to name, so it is dropped rather than guessed at."""
    readable = esm_if1_readable_positions(get_gfp_structure())["A"]
    assert to_model_numbering(readable, [CHROMOPHORE]) == []
    assert to_model_numbering(readable, [10, CHROMOPHORE, 100]) == [10, 99]


def test_unreadable_positions_is_the_complement():
    """Together the two cover every polymer residue exactly once."""
    structure = get_gfp_structure()
    readable = esm_if1_readable_positions(structure)
    assert unreadable_positions(readable, structure) == {"A": [CHROMOPHORE]}


# ── esm_if1: the model indexes positionally, so positions are renumbered ────


def _esm_if1_dispatch(sent: list[dict], native: str, readable: list[int]):
    """A worker stand-in that honours fixed positions in the model's own numbering."""

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        post_filter = "".join(native[position - 1] for position in readable)
        fixed = set((input_dict.get("fixed_positions") or {}).get("A", ()))
        sequence = "".join(residue if (index + 1) in fixed else "W" for index, residue in enumerate(post_filter))
        metrics = {"log_likelihood": -80.0, "avg_log_likelihood": -0.35, "perplexity": 1.4}
        return {"sequences": [sequence] * input_dict["batch_size"], "metrics": [metrics] * input_dict["batch_size"]}

    return staticmethod(dispatch)


def _run_esm_if1(monkeypatch, fixed, sent):
    from proto_tools.tools.inverse_folding import (
        ESMIF1SampleConfig,
        InverseFoldingInput,
        InverseFoldingStructureInput,
        run_esm_if1_sample,
    )

    structure = get_gfp_structure()
    readable = esm_if1_readable_positions(structure)["A"]
    monkeypatch.setattr(ToolInstance, "dispatch", _esm_if1_dispatch(sent, structure.get_chain_sequence("A"), readable))
    output = run_esm_if1_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure, fixed_positions=fixed)]),
        ESMIF1SampleConfig(num_sequences_per_structure=1, seed=0),
    )
    return output.design_sets[0].complexes[0]


@pytest.mark.parametrize("position", [10, 63, 65, 100, GFP_LENGTH])
def test_esm_if1_fixes_the_residue_the_caller_named(monkeypatch, position):
    """The regression: a position past the chromophore must not pin its neighbour."""
    sent: list[dict] = []
    design = _run_esm_if1(monkeypatch, {"A": [position]}, sent)
    native = get_gfp_structure().get_chain_sequence("A")
    assert design.chains[0].sequence[position - 1] == native[position - 1]


# ── both: how a fixed position crosses into the model's numbering ──────────


def _run_ligandmpnn(monkeypatch, fixed, sent):
    """Run LigandMPNN against a worker stand-in, returning its design."""
    from proto_tools.tools.inverse_folding import (
        InverseFoldingInput,
        InverseFoldingStructureInput,
        LigandMPNNSampleConfig,
        run_ligandmpnn_sample,
    )
    from proto_tools.tools.inverse_folding.ligandmpnn._readable import ligandmpnn_readable_positions

    structure = get_gfp_structure()
    readable = ligandmpnn_readable_positions(structure)["A"]
    native = structure.get_chain_sequence("A")

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        fixed_ids = set((input_dict.get("fixed_positions") or {}).get("A", ()))
        sequence = "".join(native[p - 1] if p in fixed_ids else "W" for p in readable)
        return {
            "chain_sequences": [[{"id": "A", "sequence": sequence}]],
            "metrics": [{"sequence_recovery": 1.0, "pmpnn": 0.5}],
            "pdb_strings": [None],
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    output = run_ligandmpnn_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure, fixed_positions=fixed)]),
        LigandMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
    )
    return output.design_sets[0].complexes[0]


#: Per toolkit, the runner and the payload it should send for {"A": [10, 64, 100]}. The two
#: differ because ESM-IF1 addresses a residue by its ordinal among what it read, so 100
#: renumbers to 99, while AtomWorks matches by residue id, which staging leaves alone.
FIXED_POSITION_CASES = {
    "esm-if1-sample": (_run_esm_if1, {"A": [10, 99]}),
    "ligandmpnn-sample": (_run_ligandmpnn, {"A": [10, 100]}),
}


@pytest.mark.parametrize("tool", sorted(FIXED_POSITION_CASES))
def test_a_fixed_position_reaches_the_model_in_its_own_numbering(tool, monkeypatch, caplog):
    """Renumbered or filtered, a position the model cannot read is dropped and reported."""
    runner, expected = FIXED_POSITION_CASES[tool]
    sent: list[dict] = []
    with caplog.at_level(logging.WARNING):
        design = runner(monkeypatch, {"A": [10, CHROMOPHORE, 100]}, sent)

    assert sent[0]["fixed_positions"] == expected
    assert "64 (CRO)" in caplog.text, "the dropped position is reported, not silently ignored"
    assert design.chains[0].sequence[CHROMOPHORE - 1] == "X", "and preserved regardless"


def test_a_selection_of_only_unreadable_positions_sends_nothing():
    """With nothing the model can address, the worker is given no selection at all."""
    from proto_tools.entities.structures import ResidueSelection
    from proto_tools.tools.inverse_folding.shared_data_models import split_fixed_positions

    readable = esm_if1_readable_positions(get_gfp_structure())
    resolved, dropped = split_fixed_positions(ResidueSelection(chains={"A": [CHROMOPHORE]}), readable, renumber=True)
    assert resolved is None, "an empty selection must not reach the model as an empty dict"
    assert dropped == {"A": [CHROMOPHORE]}


# ── fampnn: scores carry positions in both directions ───────────────────────


def test_fampnn_score_keys_are_renumbered_into_the_structure(monkeypatch):
    """A key names the structure's residue, not the n-th residue the parser read."""
    from proto_tools.tools.inverse_folding import (
        FAMPNNScoreAllMutationsConfig,
        FAMPNNScoreAllMutationsInput,
        run_fampnn_score_all_mutations,
    )

    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    readable = flat_readable_positions(fampnn_readable_positions(structure), structure)

    def dispatch(_tool, _input_dict, **_kwargs):
        # The model numbers its own reads from 1, labelling each with that residue's letter.
        return {
            "scores": {f"{index}{native[position - 1]}": {"A": 0.0} for index, position in enumerate(readable, start=1)}
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    output = run_fampnn_score_all_mutations(
        FAMPNNScoreAllMutationsInput(inputs=[structure]), FAMPNNScoreAllMutationsConfig(seed=0)
    )

    scores = output.results[0].scores
    positions = sorted(int(re.fullmatch(r"(\d+)(.*)", key).group(1)) for key in scores)
    assert positions == [p for p in range(1, GFP_LENGTH + 1) if p != CHROMOPHORE]
    # Each key embeds its wild-type residue, so the numbering is self-checking.
    for key in scores:
        position, wild_type = re.fullmatch(r"(\d+)(.*)", key).groups()
        assert native[int(position) - 1] == wild_type


def test_fampnn_omits_a_mutation_on_a_residue_it_cannot_read(monkeypatch, caplog):
    """Scoring the chromophore's neighbour instead would be worse than returning nothing.

    The rest of the batch is still scored, and the echoed mutations stay aligned with the
    scores so a caller can tell which of their requests came back.
    """
    from proto_tools.tools.inverse_folding import FAMPNNScoreConfig, FAMPNNScoreInput, MutationInput, run_fampnn_score

    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    sent: list[dict] = []

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        return {"mutations": input_dict["mutations"], "scores": [0.0] * len(input_dict["mutations"])}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    requested = [f"{native[9]}10A", f"{native[63]}64A", f"{native[99]}100A"]
    with caplog.at_level(logging.WARNING):
        output = run_fampnn_score(
            FAMPNNScoreInput(inputs=[MutationInput(structure=structure, mutations=requested)]),
            FAMPNNScoreConfig(seed=0),
        )

    result = output.results[0]
    assert result.mutations == [requested[0], requested[2]], "the unreadable one is omitted"
    assert len(result.scores) == len(result.mutations), "scores stay aligned with mutations"
    assert sent[0]["mutations"] == [f"{native[9]}10A", f"{native[99]}99A"], "the rest are renumbered"
    assert "does not read" in caplog.text and "64" in caplog.text


def test_fampnn_returns_nothing_when_no_mutation_can_be_scored(monkeypatch, caplog):
    """An all-unreadable batch is reported and returns empty rather than reaching the model."""
    from proto_tools.tools.inverse_folding import FAMPNNScoreConfig, FAMPNNScoreInput, MutationInput, run_fampnn_score

    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    called = []
    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(lambda *a, **k: called.append(1)))

    with caplog.at_level(logging.WARNING):
        output = run_fampnn_score(
            FAMPNNScoreInput(inputs=[MutationInput(structure=structure, mutations=[f"{native[63]}64A"])]),
            FAMPNNScoreConfig(seed=0),
        )

    assert output.results[0].mutations == [] and output.results[0].scores == []
    assert not called, "the model is never dispatched with an empty mutation list"


def test_fampnn_psce_holds_a_place_for_a_chain_the_model_never_sees():
    """The design carries every polymer chain; FAMPNN is handed only the protein ones.

    Spreading pSCE over the chains the model was given would leave the list short, and
    shifted whenever a nucleic chain sorts before a protein one.
    """
    from proto_tools.entities.structures.structure import Structure
    from proto_tools.tools.inverse_folding.fampnn.fampnn_sample import _psce_per_design_residue

    structure = Structure.from_file(Path(__file__).parent.parent / "dummy_data" / "modified_gapped_complex.pdb")
    readable = fampnn_readable_positions(structure)
    protein_chains = [cid for cid, kind in structure.get_chain_types().items() if kind == "protein"]
    assert protein_chains == ["A"], "the fixture must have a non-protein polymer chain too"

    model_sequence = "".join("W" for _ in readable["A"])
    spread = _psce_per_design_residue(
        protein_chains, [model_sequence], [float(i) for i in range(1, len(readable["A"]) + 1)], readable, structure
    )

    total = sum(len(residues) for residues in structure.get_chain_residue_atoms().values())
    assert len(spread) == total, "one entry per polymer residue of the design, every chain"
    assert spread[-3:] == [None, None, None], "the nucleic chain holds its places"


def test_fampnn_psce_has_one_entry_per_design_residue(monkeypatch):
    """Per-residue pSCE is indexed by chain position, so a skipped residue holds a place."""
    from proto_tools.tools.inverse_folding import (
        FAMPNNSampleConfig,
        FAMPNNSampleInput,
        FAMPNNStructureInput,
        run_fampnn_sample,
    )

    structure = get_gfp_structure()
    readable = fampnn_readable_positions(structure)["A"]
    native = structure.get_chain_sequence("A")
    staged_pdb = {"text": None}

    def dispatch(_tool, input_dict, **_kwargs):
        # The model packs only the residues it read, so its structure is one residue shorter.
        import gemmi

        packed = gemmi.read_pdb(input_dict["pdb_path"])
        packed.setup_entities()
        chain = packed[0]["A"]
        for index, residue in enumerate(chain):
            if residue.seqid.num == CHROMOPHORE and residue.name != "HOH":
                del chain[index]
                break
        packed.setup_entities()
        staged_pdb["text"] = packed.make_pdb_string()
        sequence = "".join(native[p - 1] for p in readable)
        # One pSCE per residue read, ascending, so a shift would be visible in the values.
        return {
            "chain_sequences": [[sequence]],
            "psce": [[float(index) for index in range(1, len(readable) + 1)]],
            "pdb_strings": [staged_pdb["text"]],
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    output = run_fampnn_sample(
        FAMPNNSampleInput(inputs=[FAMPNNStructureInput(structure=structure, chains_to_redesign=["A"])]),
        FAMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
    )

    design = output.design_sets[0].complexes[0]
    psce = design.metrics.psce
    assert len(psce) == len(design.chains[0].sequence) == GFP_LENGTH
    assert psce[CHROMOPHORE - 1] is None, "the model never packed it, so there is no error to report"
    assert psce[CHROMOPHORE - 2] == 63.0 and psce[CHROMOPHORE] == 64.0, "values straddle the gap unshifted"
    assert design.metrics.avg_psce == pytest.approx(sum(range(1, len(readable) + 1)) / len(readable))
