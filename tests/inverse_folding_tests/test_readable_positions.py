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


def test_a_readable_modified_residue_is_fixed_by_default_with_a_warning(caplog):
    """MSE is readable, so fixing it changes what the caller would expect to be designed."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(structure)

    assert 2 in constraints.worker_fixed.chains["A"], "fixed at the model, not only in the output"
    assert "position 2 (MSE)" in caplog.text
    assert "kept fixed during design" in caplog.text


def test_explicitly_fixing_a_readable_modified_residue_warns_nothing(caplog):
    """Fixing MSE is exactly what the default does, so it is silent."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    modified = [m.position for m in structure.get_chain("A").modifications]
    with caplog.at_level(logging.WARNING):
        constraints = _resolve(structure, fixed={"A": modified})

    assert caplog.text == ""
    assert 2 in constraints.worker_fixed.chains["A"]


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


def _record_dispatch(sent: list[dict], length: int = 227):
    """A ToolInstance.dispatch stand-in returning ``batch_size`` sequences of ``length``."""

    def dispatch(_tool, input_dict, **_kwargs):
        sent.append(input_dict)
        count = input_dict["batch_size"]
        return {"seq": ["W" * length] * count, "score": [0.5] * count, "seqid": [0.5] * count}

    return staticmethod(dispatch)


def test_one_mapping_is_reused_across_samples_and_warns_once(monkeypatch, caplog):
    """Warnings belong to the structure, not to each generated sequence."""
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _record_dispatch(sent))

    inputs = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=get_gfp_structure())])
    with caplog.at_level(logging.WARNING):
        output = run_proteinmpnn_sample(inputs, ProteinMPNNSampleConfig(num_sequences_per_structure=3, seed=1))

    assert len(output.design_sets[0].complexes) == 3
    assert caplog.text.count("position 64 (CRO)") == 1, "once per structure, not per design"


def test_the_worker_receives_only_processed_fixed_positions(monkeypatch):
    """The excluded chromophore must not reach ColabDesign, which cannot resolve it."""
    sent: list[dict] = []
    monkeypatch.setattr(ToolInstance, "dispatch", _record_dispatch(sent))

    inputs = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=get_gfp_structure(), fixed_positions={"A": [10, 64]})]
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
        return {"seq": ["WWW"], "score": [0.5], "seqid": [0.5]}

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
