"""tests/inverse_folding_tests/test_readable_positions.py.

Tests for the per-model readable-residue hook.

Models disagree about which residues they parse, so a single repository-wide rule is wrong
for everyone. These pin each toolkit's declared rule against the upstream behaviour it
mirrors, and pin the validation that runs in ``preprocess``. CPU-only, no model execution.
"""

from pathlib import Path

import pytest

from proto_tools.entities.structures.examples import get_gfp_structure
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding import (
    InverseFoldingInput,
    InverseFoldingStructureInput,
    ProteinMPNNSampleConfig,
)
from proto_tools.tools.inverse_folding.proteinmpnn._readable import COLABDESIGN_MODRES

MODIFIED_GAPPED_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "modified_gapped_complex.pdb"

_ATOM_TEMPLATE = {
    "N": "{rec}{ser:5d}  N   {rn:>3s} A{num:4d}    {x:8.3f}   0.000   0.000  1.00  0.00           N",
    "CA": "{rec}{ser:5d}  CA  {rn:>3s} A{num:4d}    {x:8.3f}   1.458   0.000  1.00  0.00           C",
    "C": "{rec}{ser:5d}  C   {rn:>3s} A{num:4d}    {x:8.3f}   2.009   1.420  1.00  0.00           C",
    "O": "{rec}{ser:5d}  O   {rn:>3s} A{num:4d}    {x:8.3f}   1.246   2.390  1.00  0.00           O",
}


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


_BACKBONE = ["N", "CA", "C", "O"]


def test_proteinmpnn_reads_a_modres_residue_but_not_an_unlisted_one():
    """ColabDesign rewrites its MODRES table from HETATM to ATOM; everything else it never sees."""
    structure = _structure(
        [
            ("ALA", False, _BACKBONE),
            ("MSE", True, _BACKBONE),  # in MODRES -> read as MET
            ("SEP", True, _BACKBONE),  # in MODRES -> read as SER
            ("LIG", True, _BACKBONE),  # not in MODRES -> never parsed
            ("GLY", False, _BACKBONE),
        ]
    )
    readable = ProteinMPNNSampleConfig().readable_positions(InverseFoldingStructureInput(structure=structure))
    assert readable["A"] == [1, 2, 3, 5]
    assert {"MSE", "SEP"} <= COLABDESIGN_MODRES and "LIG" not in COLABDESIGN_MODRES


def test_proteinmpnn_drops_a_residue_without_a_backbone_nitrogen():
    """``ignore_missing=True`` keeps only residues whose atom-37 slot 0 (N) is resolved."""
    structure = _structure(
        [
            ("ALA", False, _BACKBONE),
            ("LYS", False, ["CA", "C", "O"]),  # no N -> dropped despite being an ATOM record
            ("LEU", False, _BACKBONE),
        ]
    )
    readable = ProteinMPNNSampleConfig().readable_positions(InverseFoldingStructureInput(structure=structure))
    assert readable["A"] == [1, 3]


def test_proteinmpnn_skips_the_gfp_chromophore_only():
    """GFP is the motivating case: 228 residues, the chromophore at 64 unreadable."""
    structure = get_gfp_structure()
    native = structure.get_chain_sequence("A")
    readable = ProteinMPNNSampleConfig().readable_positions(InverseFoldingStructureInput(structure=structure))

    assert native[63] == "X", "the chromophore is position 64"
    assert readable["A"] == [p for p in range(1, len(native) + 1) if p != 64]


def test_proteinmpnn_reads_selenomethionine_so_it_can_be_held_fixed():
    """Rejecting a fixable MSE was a real regression; this pins that it stays allowed."""
    structure = Structure.from_file(MODIFIED_GAPPED_PDB_FILE)
    assert structure.get_chain_sequence("A")[1] == "M", "position 2 is the selenomethionine"

    config = ProteinMPNNSampleConfig()
    inputs = InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=structure, fixed_positions={"A": [2]})])
    config.preprocess(inputs)  # must not raise


def test_preprocess_rejects_a_fixed_position_the_model_cannot_read():
    """The error names the field, the chain and the position, before anything is dispatched."""
    structure = get_gfp_structure()
    config = ProteinMPNNSampleConfig()
    inputs = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=structure, fixed_positions={"A": [64]})]
    )
    with pytest.raises(ValueError, match=r"cannot read chain A residue\(s\) 64 \(CRO\)"):
        config.preprocess(inputs)


def test_a_toolkit_declaring_no_rule_validates_nothing():
    """The default keeps un-migrated toolkits behaving exactly as they do today."""
    from proto_tools.tools.inverse_folding.shared_data_models import InverseFoldingConfig

    class _Unmigrated(InverseFoldingConfig):
        pass

    structure = get_gfp_structure()
    config = _Unmigrated()
    assert config.readable_positions(InverseFoldingStructureInput(structure=structure)) is None
    inputs = InverseFoldingInput(
        inputs=[InverseFoldingStructureInput(structure=structure, fixed_positions={"A": [64]})]
    )
    config.preprocess(inputs)  # must not raise
