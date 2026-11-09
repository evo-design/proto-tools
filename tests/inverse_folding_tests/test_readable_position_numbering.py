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


def test_flat_positions_number_across_chains_in_order():
    """A structure-wide position counts every polymer residue, skipped ones included."""
    structure = get_gfp_structure()
    flat = flat_readable_positions(esm_if1_readable_positions(structure), structure)
    assert len(flat) == GFP_LENGTH - 1
    assert flat[:3] == [1, 2, 3]
    assert flat[62:65] == [63, 65, 66], "the chromophore is stepped over, not renumbered away"
    assert flat[-1] == GFP_LENGTH


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


def test_esm_if1_renumbers_a_fixed_position_for_the_worker(monkeypatch):
    """Position 100 of the chain is position 99 of what the model reads."""
    sent: list[dict] = []
    _run_esm_if1(monkeypatch, {"A": [100]}, sent)
    assert sent[0]["fixed_positions"] == {"A": [99]}


def test_esm_if1_leaves_positions_before_the_skip_alone(monkeypatch):
    """Nothing is skipped yet, so the two numberings still agree."""
    sent: list[dict] = []
    _run_esm_if1(monkeypatch, {"A": [10]}, sent)
    assert sent[0]["fixed_positions"] == {"A": [10]}


def test_esm_if1_drops_an_unreadable_fixed_position_and_says_so(monkeypatch, caplog):
    """The model cannot fix a residue it never saw; reconstruction preserves it instead."""
    sent: list[dict] = []
    with caplog.at_level(logging.WARNING):
        design = _run_esm_if1(monkeypatch, {"A": [CHROMOPHORE]}, sent)

    assert sent[0]["fixed_positions"] is None
    assert "64 (CRO)" in caplog.text
    assert design.chains[0].sequence[CHROMOPHORE - 1] == "X", "preserved regardless"


# ── ligandmpnn: the model matches by residue id, so positions are filtered ──


def test_ligandmpnn_filters_rather_than_renumbers(monkeypatch, caplog):
    """AtomWorks keeps each designed residue's own number, and raises on one it cannot find."""
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
    sent: list[dict] = []

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        fixed = set((input_dict.get("fixed_positions") or {}).get("A", ()))
        sequence = "".join(native[p - 1] if p in fixed else "W" for p in readable)
        return {
            "chain_sequences": [[{"id": "A", "sequence": sequence}]],
            "metrics": [{"sequence_recovery": 1.0, "pmpnn": 0.5}],
            "pdb_strings": [None],
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    with caplog.at_level(logging.WARNING):
        run_ligandmpnn_sample(
            InverseFoldingInput(
                inputs=[
                    InverseFoldingStructureInput(structure=structure, fixed_positions={"A": [10, CHROMOPHORE, 100]})
                ]
            ),
            LigandMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
        )

    assert sent[0]["fixed_positions"] == {"A": [10, 100]}, "kept positions keep their own numbers"
    assert "64 (CRO)" in caplog.text


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


def test_fampnn_sends_mutations_in_the_models_numbering(monkeypatch):
    """A mutation at chain position 100 is position 99 to the model."""
    from proto_tools.tools.inverse_folding import FAMPNNScoreConfig, FAMPNNScoreInput, MutationInput, run_fampnn_score

    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    sent: list[dict] = []

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        return {"mutations": input_dict["mutations"], "scores": [0.0] * len(input_dict["mutations"])}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    requested = [f"{native[9]}10A", f"{native[99]}100A"]
    output = run_fampnn_score(
        FAMPNNScoreInput(inputs=[MutationInput(structure=structure, mutations=requested)]),
        FAMPNNScoreConfig(seed=0),
    )

    assert sent[0]["mutations"] == [f"{native[9]}10A", f"{native[99]}99A"]
    assert output.results[0].mutations == requested, "the caller is answered in their own numbering"


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
