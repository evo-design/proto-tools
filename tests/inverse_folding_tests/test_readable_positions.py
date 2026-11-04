"""tests/inverse_folding_tests/test_readable_positions.py.

Tests for ProteinMPNN's residue mapping and the constraints derived from it.

Parsers disagree about which residues they process, so the mapping is per model. These pin
ProteinMPNN's rule against the upstream behaviour it mirrors, the constraints resolved from
it, and the worker payload that results. CPU-only, no model execution.
"""

import logging
from pathlib import Path

import pytest

from proto_tools.entities.complex import Complex
from proto_tools.entities.structures.examples import get_gfp_structure
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
    ProteinMPNNSampleConfig,
    run_proteinmpnn_sample,
)
from proto_tools.tools.inverse_folding.proteinmpnn._constraints import resolve_constraints
from proto_tools.tools.inverse_folding.proteinmpnn._readable import (
    COLABDESIGN_MODRES,
    proteinmpnn_readable_positions,
)
from proto_tools.tools.inverse_folding.shared_data_models import DesignedComplex
from proto_tools.utils import ToolInstance

MODIFIED_GAPPED_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "modified_gapped_complex.pdb"
_TOOL = "proteinmpnn-sample"

_ATOM_TEMPLATE = {
    "N": "{rec}{ser:5d}  N   {rn:>3s} A{num:4d}    {x:8.3f}   0.000   0.000  1.00  0.00           N",
    "CA": "{rec}{ser:5d}  CA  {rn:>3s} A{num:4d}    {x:8.3f}   1.458   0.000  1.00  0.00           C",
    "C": "{rec}{ser:5d}  C   {rn:>3s} A{num:4d}    {x:8.3f}   2.009   1.420  1.00  0.00           C",
    "O": "{rec}{ser:5d}  O   {rn:>3s} A{num:4d}    {x:8.3f}   1.246   2.390  1.00  0.00           O",
}
_BACKBONE = ["N", "CA", "C", "O"]


def _structure(spec: list[tuple[str, bool, list[str]]]) -> Structure:
    """Build a single-chain structure from ``(residue name, is_hetatm, atom names)`` per residue."""
    lines: list[str] = []
    serial = 1
    for number, (resname, is_hetatm, atoms) in enumerate(spec, start=1):
        record = "HETATM" if is_hetatm else "ATOM  "
        for atom in atoms:
            lines.append(_ATOM_TEMPLATE[atom].format(rec=record, ser=serial, rn=resname, num=number, x=4.0 * number))
            serial += 1
    return Structure(structure="\n".join([*lines, "END", ""]))


def _resolve(structure, chains=("A",), fixed=None):
    """Resolve constraints for one structure, the way the tool does."""
    selection = InverseFoldingStructureInput(structure=structure, fixed_positions=fixed).fixed_positions
    return resolve_constraints(structure, list(chains), selection, tool=_TOOL)


# ── the residue mapping ─────────────────────────────────────────────────────


def test_a_modres_residue_is_processed_but_an_unlisted_one_is_not():
    """ColabDesign renames its MODRES table from HETATM to ATOM; everything else it never sees."""
    structure = _structure(
        [
            ("ALA", False, _BACKBONE),
            ("MSE", True, _BACKBONE),
            ("SEP", True, _BACKBONE),
            ("LIG", True, _BACKBONE),
            ("GLY", False, _BACKBONE),
        ]
    )
    assert _resolve(structure).readable["A"] == [1, 2, 3, 5]
    assert {"MSE", "SEP"} <= COLABDESIGN_MODRES and "LIG" not in COLABDESIGN_MODRES


def test_a_residue_without_a_backbone_nitrogen_is_excluded():
    """``ignore_missing=True`` keeps only residues whose atom-37 slot 0 (N) is resolved."""
    structure = _structure([("ALA", False, _BACKBONE), ("LYS", False, ["CA", "C", "O"]), ("LEU", False, _BACKBONE)])
    assert _resolve(structure).readable["A"] == [1, 3]


def test_the_gfp_chromophore_is_the_only_excluded_position():
    """The motivating case: 228 residues, the chromophore at 64 excluded."""
    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    assert native[63] == "X"
    assert _resolve(structure).readable["A"] == [p for p in range(1, len(native) + 1) if p != 64]


def test_the_mapping_is_the_same_before_and_after_staging():
    """The model reads the staged file, so a mapping taken from the input must agree with it."""
    from proto_tools.tools.inverse_folding.structure_staging import staged_structure

    structure = get_gfp_structure()
    before = proteinmpnn_readable_positions(structure)
    with staged_structure(structure) as staged:
        restaged = Structure(structure=staged.path.read_text(), structure_format="pdb", source="staged")
        after = proteinmpnn_readable_positions(restaged)
    assert before["A"] == after["A"]


def test_the_mapping_survives_an_mmcif_round_trip():
    """CIF input is converted before staging; the mapping must not shift."""
    structure = get_gfp_structure()
    as_cif = Structure(structure=structure.structure_cif, structure_format="cif", source="cif")
    assert proteinmpnn_readable_positions(structure)["A"] == proteinmpnn_readable_positions(as_cif)["A"]


# ── resolved constraints and warnings ───────────────────────────────────────


def test_an_unfixed_excluded_residue_is_preserved_with_one_warning(caplog):
    """The caller did not ask for this, so it has to be reported."""
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(get_gfp_structure())

    assert 64 in constraints.effective_fixed["A"]
    assert constraints.worker_fixed is None, "an excluded position cannot be fixed at the model"
    assert caplog.text.count("position 64 (CRO)") == 1
    assert "cannot redesign" in caplog.text


def test_explicitly_fixing_an_excluded_residue_warns_nothing(caplog):
    """The constraint is already satisfied, so there is nothing to report."""
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(get_gfp_structure(), fixed={"A": [64]})

    assert caplog.text == ""
    assert 64 in constraints.effective_fixed["A"]


def test_an_unfixed_modified_residue_stays_redesignable_and_is_silent(caplog):
    """A chemical modification alone must not fix a position the parser can read."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(structure)

    assert constraints.worker_fixed is None, "nothing is fixed unless the caller asks"
    assert 2 not in constraints.effective_fixed["A"], "MSE at 2 is free to be redesigned"
    assert caplog.text == ""


def test_explicitly_fixing_a_modified_residue_explains_the_parent_amino_acid(caplog):
    """The sequence shows M while the annotation stays MSE, so say so once."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(structure, fixed={"A": [2]})

    assert constraints.worker_fixed.chains["A"] == [2]
    assert "position 2 (MSE) is fixed" in caplog.text
    assert "contains M at this position" in caplog.text
    assert "chains[].modifications" in caplog.text


def test_a_residue_missing_its_backbone_nitrogen_warns_with_the_reason(caplog):
    """A standard residue excluded for a structural reason gets its own message."""
    structure = _structure([("ALA", False, _BACKBONE), ("LYS", False, ["CA", "C", "O"]), ("LEU", False, _BACKBONE)])
    with caplog.at_level(logging.WARNING):
        _resolve(structure)

    assert "position 2 (LYS)" in caplog.text
    assert "backbone nitrogen is missing" in caplog.text


def test_excluded_positions_are_dropped_from_worker_constraints_without_shifting(caplog):
    """ColabDesign matches fixed positions by residue number, so the rest keep their numbers."""
    structure = _structure(
        [
            ("ALA", False, _BACKBONE),
            ("LIG", True, _BACKBONE),
            ("GLY", False, _BACKBONE),
            ("LEU", False, _BACKBONE),
        ]
    )
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(structure, fixed={"A": [1, 2, 4]})

    assert constraints.worker_fixed.chains["A"] == [1, 4], "2 removed, 4 keeps its number"
    assert 2 in constraints.effective_fixed["A"], "still preserved in the output"


# ── reconstruction ──────────────────────────────────────────────────────────


def test_a_short_design_is_placed_on_the_processed_positions():
    """The PR #360 regression: the design must land where the model actually read."""
    structure = get_gfp_structure()
    constraints = _resolve(structure)

    design = DesignedComplex.from_design(
        structure,
        {"A": "W" * len(constraints.readable["A"])},
        {"A"},
        tool=_TOOL,
        include_ligands=False,
        readable_positions=constraints.readable,
        fixed_positions=constraints.effective_fixed,
    )

    chain = design.chains[0]
    assert len(chain.sequence) == 228
    assert chain.sequence[63] == "X", "the chromophore is untouched"
    assert [(m.position, m.modification_code) for m in chain.modifications] == [(64, "CRO")]


def test_a_full_length_sequence_is_rejected_when_the_mapping_excludes_a_position():
    """A length matching the chain is not evidence the mapping was honoured."""
    structure = get_gfp_structure()
    constraints = _resolve(structure)
    with pytest.raises(ValueError, match="returned 228 residues"):
        DesignedComplex.from_design(
            structure,
            {"A": "W" * 228},
            {"A"},
            tool=_TOOL,
            include_ligands=False,
            readable_positions=constraints.readable,
        )


def test_a_redesigned_chain_missing_from_the_mapping_is_rejected():
    """A missing entry must not fall back to plain length matching."""
    with pytest.raises(ValueError, match="missing from the residue mapping"):
        DesignedComplex.from_design(
            get_gfp_structure(),
            {"A": "W" * 227},
            {"A"},
            tool=_TOOL,
            include_ligands=False,
            readable_positions={},
        )


def test_a_fixed_modified_residue_keeps_its_annotation_but_a_redesigned_one_does_not():
    """An annotation may only stay where the output residue still matches the input."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    native = Complex.from_structure(structure, include_ligands=False).chains[0].sequence
    constraints = _resolve(structure, fixed={"A": [2]})
    readable = constraints.readable["A"]

    designed = "".join(native[p - 1] if p == 2 else "W" for p in readable)
    design = DesignedComplex.from_design(
        structure,
        {"A": designed},
        {"A"},
        tool=_TOOL,
        include_ligands=False,
        readable_positions=constraints.readable,
        fixed_positions={"A": [2]},
    )

    codes = {m.position: m.modification_code for m in design.chains[0].modifications}
    assert codes.get(2) == "MSE", "a fixed modified residue keeps its annotation"
    assert 6 not in codes, "a redesigned position drops its stale annotation"


def test_an_unmigrated_toolkit_keeps_strict_length_matching():
    """Declaring no mapping must behave exactly as before this feature."""
    with pytest.raises(ValueError, match="returned 227 residues"):
        DesignedComplex.from_design(
            get_gfp_structure(), {"A": "W" * 227}, {"A"}, tool="esm-if1-sample", include_ligands=False
        )


# ── the worker payload ──────────────────────────────────────────────────────


def _record_dispatch(sent: list[dict], native: str | None = None, readable: list[int] | None = None):
    """A dispatch stand-in that honours the fixed positions it is handed.

    Reconstruction rejects a design that changed a fixed residue, so a stand-in that returned
    a constant string would fail for the wrong reason.
    """

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        count = input_dict["batch_size"]
        fixed = set((input_dict.get("fixed_positions") or {}).get("A", ()))
        positions = readable or []
        seq = "".join(native[p - 1] if (native and p in fixed) else "W" for p in positions)
        return {"seq": [seq] * count, "score": [0.5] * count, "seqid": [0.5] * count}

    return staticmethod(dispatch)


def test_one_mapping_is_reused_across_samples_and_warns_once(monkeypatch, caplog):
    """Warnings belong to the structure, not to each generated sequence."""
    structure = get_gfp_structure()
    readable = _resolve(structure).readable["A"]
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _record_dispatch(sent, structure.get_chain_sequence("A"), readable))

    inputs = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure)])
    caplog.clear()  # the mapping was resolved above; count only the run's own warnings
    with caplog.at_level(logging.WARNING):
        output = run_proteinmpnn_sample(inputs, ProteinMPNNSampleConfig(num_sequences_per_structure=3, seed=1))

    assert len(output.design_sets[0].complexes) == 3
    assert caplog.text.count("position 64 (CRO)") == 1, "once per structure, not per design"


def test_the_worker_receives_only_processed_fixed_positions(monkeypatch):
    """The excluded chromophore must not reach ColabDesign, which cannot resolve it."""
    structure = get_gfp_structure()
    readable = _resolve(structure).readable["A"]
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _record_dispatch(sent, structure.get_chain_sequence("A"), readable))

    inputs = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=structure, fixed_positions={"A": [10, 64]})]
    )
    run_proteinmpnn_sample(inputs, ProteinMPNNSampleConfig(num_sequences_per_structure=1, seed=1))

    assert sent[0]["fixed_positions"] == {"A": [10]}, "64 dropped, 10 keeps its number"


# ── homooligomer ────────────────────────────────────────────────────────────


def _protomer(chain_id: str, spec: list[tuple[str, bool, list[str]]], offset: float) -> list[str]:
    """Atom lines for one protomer chain, laid out clear of the others."""
    lines: list[str] = []
    for number, (resname, is_hetatm, atoms) in enumerate(spec, start=1):
        record = "HETATM" if is_hetatm else "ATOM  "
        for atom in atoms:
            line = _ATOM_TEMPLATE[atom].format(rec=record, ser=1, rn=resname, num=number, x=4.0 * number)
            lines.append(f"{line[:21]}{chain_id}{line[22:38]}{float(line[38:46]) + offset:8.3f}{line[46:]}")
    return lines


def _two_protomers(spec_a, spec_b) -> Structure:
    """Build a two-chain structure from explicit per-protomer residue specs."""
    lines = _protomer("A", spec_a, 0.0) + _protomer("B", spec_b, 30.0)
    return Structure(structure="\n".join([*lines, "END", ""]))


def test_protomers_excluding_different_positions_are_rejected():
    """Equal length and numbering are not enough: tying would pair different residues."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import require_matching_protomers

    spec_a = [("ALA", False, _BACKBONE), ("LIG", True, _BACKBONE), ("GLY", False, _BACKBONE)]
    spec_b = [("ALA", False, _BACKBONE), ("GLY", False, _BACKBONE), ("LIG", True, _BACKBONE)]
    structure = _two_protomers(spec_a, spec_b)
    readable = proteinmpnn_readable_positions(structure)
    assert readable["A"] != readable["B"], "the fixture must actually differ"

    with pytest.raises(ValueError, match="processes different residue positions in each chain"):
        require_matching_protomers(structure, ["A", "B"], readable, tool="proteinmpnn-sample-homooligomer")


def test_protomers_excluding_the_same_positions_are_accepted():
    """A genuine homooligomer excludes the same position in every protomer."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import require_matching_protomers

    spec = [("ALA", False, _BACKBONE), ("LIG", True, _BACKBONE), ("GLY", False, _BACKBONE)]
    structure = _two_protomers(spec, spec)
    readable = proteinmpnn_readable_positions(structure)
    assert readable["A"] == readable["B"] == [1, 3]

    require_matching_protomers(structure, ["A", "B"], readable, tool="proteinmpnn-sample-homooligomer")


def test_protomer_fixed_positions_reach_the_worker_without_excluded_ones(monkeypatch):
    """`protomer_fixed_positions` broadcasts, then the excluded ones are dropped per chain."""
    from proto_tools.tools.inverse_folding import (
        ProteinMPNNHomooligomerSampleConfig,
        ProteinMPNNHomooligomerSampleInput,
        ProteinMPNNHomooligomerStructureInput,
        run_proteinmpnn_sample_homooligomer,
    )

    spec = [
        ("ALA", False, _BACKBONE),
        ("LIG", True, _BACKBONE),
        ("GLY", False, _BACKBONE),
        ("LEU", False, _BACKBONE),
    ]
    structure = _two_protomers(spec, spec)

    sent: list[dict] = []

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        fixed = set((input_dict.get("fixed_positions") or {}).get("A", ()))
        native = structure.get_chain_sequence("A")
        readable = proteinmpnn_readable_positions(structure)["A"]
        seq = "".join(native[p - 1] if p in fixed else "W" for p in readable)
        return {"seq": [seq], "score": [0.5], "seqid": [0.5]}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))

    inputs = ProteinMPNNHomooligomerSampleInput(
        inputs=[
            ProteinMPNNHomooligomerStructureInput(
                structure=structure, protomer_chains=["A", "B"], protomer_fixed_positions=[1, 2]
            )
        ]
    )
    run_proteinmpnn_sample_homooligomer(
        inputs, ProteinMPNNHomooligomerSampleConfig(num_sequences_per_structure=1, seed=1)
    )

    assert sent[0]["fixed_positions"] == {"A": [1], "B": [1]}, "position 2 is excluded in both"


def test_an_unfixed_modified_residue_loses_its_annotation_even_when_the_letter_matches():
    """Returning M does not make a designed methionine a selenomethionine.

    The regression Prompt 4 calls out: annotation survival must follow whether the residue
    was preserved, not whether the output letter happens to equal the input's.
    """
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    native = Complex.from_structure(structure, include_ligands=False).chains[0].sequence
    constraints = _resolve(structure)  # nothing fixed
    readable = constraints.readable["A"]
    assert 2 in readable and native[1] == "M", "MSE at 2 reads as M and is redesignable"

    # the model happens to return the native letter at the unfixed MSE position
    designed = "".join(native[p - 1] if p == 2 else "W" for p in readable)
    design = DesignedComplex.from_design(
        structure,
        {"A": designed},
        {"A"},
        tool=_TOOL,
        include_ligands=False,
        readable_positions=constraints.readable,
        fixed_positions=constraints.effective_fixed,
    )

    assert design.chains[0].sequence[1] == "M"
    assert all(m.position != 2 for m in design.chains[0].modifications), "MSE must not be claimed"


def test_a_design_that_changes_a_fixed_residue_is_rejected():
    """A fixed position coming back different is a violated constraint, not a dropped annotation."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    constraints = _resolve(structure, fixed={"A": [2]})
    readable = constraints.readable["A"]

    with pytest.raises(ValueError, match="was fixed but the model returned a different residue"):
        DesignedComplex.from_design(
            structure,
            {"A": "W" * len(readable)},
            {"A"},
            tool=_TOOL,
            include_ligands=False,
            readable_positions=constraints.readable,
            fixed_positions=constraints.effective_fixed,
        )


def test_tied_protomers_fixed_to_different_amino_acids_are_rejected():
    """One shared sequence cannot be methionine in A and serine in B at the same position."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import require_compatible_fixed_residues

    spec_a = [("ALA", 0), ("MET", 0), ("GLY", 0)]
    spec_b = [("ALA", 0), ("SER", 0), ("GLY", 0)]
    structure = _two_protomers([(n, h, _BACKBONE) for n, h in spec_a], [(n, h, _BACKBONE) for n, h in spec_b])
    assert structure.get_chain_sequence("A")[1] != structure.get_chain_sequence("B")[1]

    with pytest.raises(ValueError, match="position 2"):
        require_compatible_fixed_residues(
            structure, ["A", "B"], {"A": [2], "B": [2]}, tool="proteinmpnn-sample-homooligomer"
        )


def test_tied_protomers_fixed_to_the_same_amino_acid_are_accepted():
    """Pseudo-symmetric inputs stay supported when the fixed residues agree."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import require_compatible_fixed_residues

    spec_a = [("ALA", 0), ("MET", 0), ("GLY", 0)]
    spec_b = [("TRP", 0), ("MET", 0), ("GLY", 0)]  # differs at 1, which is not fixed
    structure = _two_protomers([(n, h, _BACKBONE) for n, h in spec_a], [(n, h, _BACKBONE) for n, h in spec_b])
    require_compatible_fixed_residues(
        structure, ["A", "B"], {"A": [2], "B": [2]}, tool="proteinmpnn-sample-homooligomer"
    )


# ── scoring ─────────────────────────────────────────────────────────────────


def test_scoring_trims_the_sequence_to_the_processed_positions(caplog):
    """A scored sequence covers every observed residue; the model parses fewer."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import restrict_sequence_to_readable

    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    with caplog.at_level(logging.WARNING):
        trimmed = restrict_sequence_to_readable(structure, ["A"], native, tool="proteinmpnn-score")

    assert len(native) == 228 and len(trimmed) == 227
    assert trimmed == native[:63] + native[64:], "only the chromophore is dropped"
    assert "position 64 (CRO)" in caplog.text


def test_scoring_refuses_logits_it_cannot_align():
    """Logits are declared one row per sequence position, so a trimmed sequence breaks that."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import restrict_sequence_to_readable

    structure = get_gfp_structure()
    with pytest.raises(ValueError, match="per-position logits cannot be aligned"):
        restrict_sequence_to_readable(
            structure,
            ["A"],
            structure.get_chain_sequence("A"),
            tool="proteinmpnn-score",
            allow_excluded=False,
        )


def test_scoring_rejects_a_sequence_that_does_not_cover_the_chain():
    """A length mismatch against the structure is the caller's error, not the parser's."""
    from proto_tools.tools.inverse_folding.proteinmpnn._constraints import restrict_sequence_to_readable

    with pytest.raises(ValueError, match="but chains"):
        restrict_sequence_to_readable(get_gfp_structure(), ["A"], "MVLS", tool="proteinmpnn-score")


# ── scoring: the worker payload ─────────────────────────────────────────────


def _score_payload(monkeypatch, structure, excluded=None, return_logits=False):
    """Run scoring against a stubbed worker and return the dispatched payload."""
    from proto_tools.tools.inverse_folding import (
        ProteinMPNNScoringConfig,
        ProteinMPNNScoringInput,
        SequenceStructurePair,
        run_proteinmpnn_score,
    )

    sent: list[dict] = []

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        return {
            "metrics": {"log_likelihood": -1.0, "avg_log_likelihood": -0.5, "perplexity": 1.6},
            "logits": None,
            "vocab": None,
        }

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(dispatch))
    sequence = "".join(structure.get_chain_sequence(c) for c in structure.get_chain_ids())
    run_proteinmpnn_score(
        ProteinMPNNScoringInput(
            sequence_structure_pairs=[
                SequenceStructurePair(sequence=sequence, structure=structure, excluded_positions=excluded)
            ]
        ),
        ProteinMPNNScoringConfig(return_logits=return_logits),
    )
    return sent[0]


def test_scoring_sends_the_trimmed_sequence_and_warns(monkeypatch, caplog):
    """GFP with no user exclusions: 227 residues reach the worker, with one warning."""
    with caplog.at_level(logging.WARNING):
        payload = _score_payload(monkeypatch, get_gfp_structure())

    assert len(payload["sequence"]) == 227
    assert payload["fixed_positions"] is None
    assert caplog.text.count("position 64 (CRO)") == 1


def test_scoring_drops_a_parser_excluded_position_from_the_worker_selection(monkeypatch, caplog):
    """Explicitly excluding CRO is honoured without sending a position the model lacks."""
    with caplog.at_level(logging.WARNING):
        payload = _score_payload(monkeypatch, get_gfp_structure(), excluded={"A": [10, 64]})

    worker = payload["fixed_positions"]
    assert list(worker) == ["A"] and len(worker["A"]) == 1, "only the readable exclusion is sent"
    assert "position 64 (CRO)" in caplog.text, "the parser exclusion is still reported"


def test_scoring_does_not_warn_about_a_user_excluded_readable_position(monkeypatch, caplog):
    """A readable residue the caller excluded is exactly what they asked for."""
    structure = _structure([("ALA", False, _BACKBONE), ("GLY", False, _BACKBONE), ("LEU", False, _BACKBONE)])
    with caplog.at_level(logging.WARNING):
        payload = _score_payload(monkeypatch, structure, excluded={"A": [2]})

    assert caplog.text == ""
    assert payload["fixed_positions"] is not None


def test_scoring_translates_positions_through_nonsequential_numbering(monkeypatch):
    """GFP's residues do not start at 1, so ordinal positions must be translated."""
    structure = get_gfp_structure()
    first_number = structure.get_chain_residue_ids()["A"][0][0]
    assert first_number != 1, "the fixture must actually be offset"

    payload = _score_payload(monkeypatch, structure, excluded={"A": [10]})
    assert payload["fixed_positions"]["A"] == [structure.get_chain_residue_ids()["A"][9][0]]


def test_scoring_refuses_logits_when_a_position_is_parser_excluded(monkeypatch):
    """Logits would not align to the caller's sequence, so this is rejected before dispatch."""
    with pytest.raises(ValueError, match="per-position logits cannot be aligned"):
        _score_payload(monkeypatch, get_gfp_structure(), return_logits=True)
