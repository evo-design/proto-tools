"""tests/inverse_folding_tests/test_readable_positions_all_toolkits.py.

Every inverse-folding toolkit's residue mapping, against the same awkward structures.

Each model ships its own structure parser and they disagree about what they will read, so a
rule is declared per toolkit. These tests pin all of them side by side: the same fixture goes
to every rule and each is asserted against what its own upstream parser does, which is what
makes a disagreement visible rather than plausible. CPU-only, no model execution.

``test_every_migrated_toolkit_declares_a_rule`` is the guard that matters most: a toolkit
whose config extends :class:`InverseFoldingConfig` without declaring a rule reverts to strict
length matching and fails on any structure containing a residue its parser skips.
"""

import pytest

from proto_tools.entities.structures.examples import get_gfp_structure
from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.esm_if1._readable import esm_if1_readable_positions
from proto_tools.tools.inverse_folding.fampnn._readable import fampnn_readable_positions
from proto_tools.tools.inverse_folding.ligandmpnn._readable import ligandmpnn_readable_positions
from proto_tools.tools.inverse_folding.proteinmpnn._readable import proteinmpnn_readable_positions
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedComplex,
    unreadable_positions,
    validate_residue_mapping,
)

#: Every toolkit that declares a rule, by the tool key it is declared on.
RULES = {
    "esm-if1-sample": esm_if1_readable_positions,
    "fampnn-sample": fampnn_readable_positions,
    "ligandmpnn-sample": ligandmpnn_readable_positions,
    "proteinmpnn-sample": proteinmpnn_readable_positions,
}

CHROMOPHORE = 64
GFP_LENGTH = 228

BACKBONE = ("N", "CA", "C", "O")
#: A nucleotide's atoms share no name with any protein backbone or AlphaFold heavy atom.
NUCLEIC_ATOMS = ("P", "O5'", "C5'", "N9")


def _structure(chains: dict[str, list[tuple[str, bool, tuple[str, ...]]]]) -> Structure:
    """Build a structure from ``(residue name, is_hetatm, atom names)`` per residue, per chain.

    Atom names are written verbatim, so a residue can carry names no parser recognizes.

    Args:
        chains (dict[str, list[tuple[str, bool, tuple[str, ...]]]]): Residue specs per chain ID.

    Returns:
        Structure: A single-model structure holding exactly those residues.
    """
    lines: list[str] = []
    serial = 1
    for offset, (chain_id, residues) in enumerate(chains.items()):
        for number, (name, is_hetatm, atoms) in enumerate(residues, start=1):
            record = "HETATM" if is_hetatm else "ATOM  "
            for index, atom in enumerate(atoms):
                # Atom names of three characters or fewer start in column 14 by convention.
                field = f"{atom:<4s}" if len(atom) == 4 else f" {atom:<3s}"
                x = 4.0 * number + 30.0 * offset
                # Columns: 13-16 atom name, 17 altLoc, 18-20 residue name, 22 chain, 23-26 number.
                lines.append(
                    f"{record}{serial:5d} {field} {name:>3s} {chain_id}{number:4d}    "
                    f"{x:8.3f}{index * 1.2:8.3f}{0.0:8.3f}  1.00  0.00          "
                    f"{atom[0]:>2s}"
                )
                serial += 1
    return Structure(structure="\n".join([*lines, "END", ""]))


def _one_chain(residues: list[tuple[str, bool, tuple[str, ...]]]) -> Structure:
    """Build a single-chain structure named ``A``."""
    return _structure({"A": residues})


# Each case names a structure and the positions every toolkit's parser reads from it. The
# expectations are derived from each upstream parser's own source, not from observed output.
CASES: dict[str, tuple[Structure, dict[str, dict[str, list[int]]]]] = {
    # Every model reads selenomethionine, by a different route: ColabDesign rewrites its
    # MODRES table to ATOM, biotite maps MSE to M, FAMPNN lists it as an NCAA, and LigandMPNN
    # relies on AtomWorks' own MSE-to-MET conversion, which only applies once
    # LIGANDMPNN_ATOM_RECORD_RESIDUES has staged it as ATOM so the HETATM split leaves it in
    # the polymer chain.
    "mse_written_as_hetatm": (
        _one_chain([("ALA", False, BACKBONE), ("MSE", True, BACKBONE), ("GLY", False, BACKBONE)]),
        {
            "esm-if1-sample": {"A": [1, 2, 3]},
            "fampnn-sample": {"A": [1, 2, 3]},
            "ligandmpnn-sample": {"A": [1, 2, 3]},
            "proteinmpnn-sample": {"A": [1, 2, 3]},
        },
    ),
    # Phosphoserine is in ColabDesign's MODRES table and FAMPNN's NCAA table, but biotite has
    # no one-letter code for it.
    "sep_written_as_hetatm": (
        _one_chain([("ALA", False, BACKBONE), ("SEP", True, BACKBONE), ("GLY", False, BACKBONE)]),
        {
            "esm-if1-sample": {"A": [1, 3]},
            "fampnn-sample": {"A": [1, 2, 3]},
            "ligandmpnn-sample": {"A": [1, 3]},
            "proteinmpnn-sample": {"A": [1, 2, 3]},
        },
    ),
    # A residue no table lists is skipped by every parser.
    "unlisted_hetatm_residue": (
        _one_chain([("ALA", False, BACKBONE), ("LIG", True, BACKBONE), ("GLY", False, BACKBONE)]),
        {
            "esm-if1-sample": {"A": [1, 3]},
            "fampnn-sample": {"A": [1, 3]},
            "ligandmpnn-sample": {"A": [1, 3]},
            "proteinmpnn-sample": {"A": [1, 3]},
        },
    ),
    # ProteinMPNN requires a backbone nitrogen and LigandMPNN requires all four atoms;
    # ESM-IF1 holds the position on any one of them and FAMPNN on any recognized atom.
    "missing_backbone_nitrogen": (
        _one_chain([("ALA", False, BACKBONE), ("LYS", False, ("CA", "C", "O")), ("GLY", False, BACKBONE)]),
        {
            "esm-if1-sample": {"A": [1, 2, 3]},
            "fampnn-sample": {"A": [1, 2, 3]},
            "ligandmpnn-sample": {"A": [1, 3]},
            "proteinmpnn-sample": {"A": [1, 3]},
        },
    ),
    # Only LigandMPNN asks for the carbonyl oxygen.
    "missing_carbonyl_oxygen": (
        _one_chain([("ALA", False, BACKBONE), ("LYS", False, ("N", "CA", "C")), ("GLY", False, BACKBONE)]),
        {
            "esm-if1-sample": {"A": [1, 2, 3]},
            "fampnn-sample": {"A": [1, 2, 3]},
            "ligandmpnn-sample": {"A": [1, 3]},
            "proteinmpnn-sample": {"A": [1, 2, 3]},
        },
    ),
    # UNK is a position every parser keeps, so an unknown residue is designed, not skipped.
    "unknown_residue_written_as_atom": (
        _one_chain([("ALA", False, BACKBONE), ("UNK", False, BACKBONE), ("GLY", False, BACKBONE)]),
        {
            "esm-if1-sample": {"A": [1, 2, 3]},
            "fampnn-sample": {"A": [1, 2, 3]},
            "ligandmpnn-sample": {"A": [1, 2, 3]},
            "proteinmpnn-sample": {"A": [1, 2, 3]},
        },
    ),
    # A nucleic chain contributes no design positions to any protein inverse-folding model,
    # even though its residues are ordinary ATOM records.
    "nucleic_acid_chain": (
        _structure(
            {
                "A": [("ALA", False, BACKBONE), ("GLY", False, BACKBONE)],
                "B": [("DA", False, NUCLEIC_ATOMS), ("DG", False, NUCLEIC_ATOMS)],
            }
        ),
        {
            "esm-if1-sample": {"A": [1, 2], "B": []},
            "fampnn-sample": {"A": [1, 2], "B": []},
            "ligandmpnn-sample": {"A": [1, 2], "B": []},
            "proteinmpnn-sample": {"A": [1, 2], "B": []},
        },
    ),
}


@pytest.mark.parametrize("tool", sorted(RULES))
@pytest.mark.parametrize("case", sorted(CASES))
def test_each_toolkit_reads_the_positions_its_parser_reads(case: str, tool: str):
    """Pin every rule against the same awkward structures, chain by chain.

    The expectation names every polymer chain, so a chain the model reads none of has to come
    back present and empty rather than missing, and the mapping has to be well formed.
    """
    structure, expected = CASES[case]
    readable = RULES[tool](structure)
    validate_residue_mapping(readable, structure)
    assert readable == expected[tool]


@pytest.mark.parametrize("tool", sorted(RULES))
def test_the_gfp_chromophore_is_the_only_position_no_toolkit_reads(tool: str):
    """The motivating case: every model skips CRO at 64 and reads the other 227 residues."""
    structure = get_gfp_structure()
    assert len(structure.get_chain_residue_atoms()["A"]) == GFP_LENGTH
    assert RULES[tool](structure)["A"] == [p for p in range(1, GFP_LENGTH + 1) if p != CHROMOPHORE]


@pytest.mark.parametrize("tool", sorted(RULES))
def test_a_mapping_survives_staging_and_a_cif_round_trip(tool: str):
    """The model reads the staged file, so a mapping taken from the input must agree with it."""
    from proto_tools.tools.inverse_folding.structure_staging import staged_structure

    structure = get_gfp_structure()
    before = RULES[tool](structure)["A"]

    with staged_structure(structure) as staged:
        restaged = Structure(structure=staged.path.read_text(), structure_format="pdb", source="staged")
    assert RULES[tool](restaged)["A"] == before, "staging must not shift the mapping"

    as_cif = Structure(structure=structure.structure_cif, structure_format="cif", source="cif")
    assert RULES[tool](as_cif)["A"] == before, "a CIF round trip must not shift the mapping"


@pytest.mark.parametrize("tool", sorted(RULES))
def test_a_short_design_is_placed_back_around_the_skipped_residue(tool: str):
    """The end-to-end contract: a 227-residue design lands on a 228-residue chain."""
    structure = get_gfp_structure()
    readable = RULES[tool](structure)
    design = DesignedComplex.from_design(
        structure,
        {"A": "W" * len(readable["A"])},
        {"A"},
        tool=tool,
        include_ligands=False,
        readable_positions=readable,
        fixed_positions=unreadable_positions(readable, structure),
    )

    chain = design.chains[0]
    assert len(chain.sequence) == 228
    assert chain.sequence[63] == "X", "the chromophore keeps its input identity"
    assert [(m.position, m.modification_code) for m in chain.modifications] == [(64, "CRO")]


def _inverse_folding_config_subclasses() -> list[type]:
    """Every config that inherits the readable_positions hook, however deeply."""
    from proto_tools.tools.inverse_folding import InverseFoldingConfig

    found: list[type] = []
    pending = [InverseFoldingConfig]
    while pending:
        for subclass in pending.pop().__subclasses__():
            found.append(subclass)
            pending.append(subclass)
    return found


def test_every_migrated_toolkit_declares_a_rule():
    """A config extending InverseFoldingConfig without a rule silently fails on real inputs.

    ``InverseFoldingConfig.readable_positions`` returns ``None`` by default, which keeps
    strict length matching. That is a safe default for a toolkit nobody has mirrored yet, but
    for one that ships it means any structure with a skipped residue raises instead of
    designing. Discovering the subclasses rather than listing them is what makes this catch a
    toolkit added later.
    """
    from proto_tools.tools.inverse_folding import InverseFoldingStructureInput, LigandMPNNSampleConfig

    inp = InverseFoldingStructureInput(structure=get_gfp_structure())
    expected = [p for p in range(1, GFP_LENGTH + 1) if p != CHROMOPHORE]
    subclasses = _inverse_folding_config_subclasses()
    assert len(subclasses) >= 4, f"expected every sample config to be discovered, got {subclasses}"

    for config_class in subclasses:
        config = config_class()
        if isinstance(config, LigandMPNNSampleConfig) and config.implementation == "original":
            continue
        readable = config.readable_positions(inp)
        assert readable is not None, f"{config_class.__name__} declares no residue mapping"
        assert readable["A"] == expected, f"{config_class.__name__} disagrees about GFP"


def test_the_legacy_ligandmpnn_implementation_declares_no_rule():
    """``implementation='original'`` runs a different parser, so it must not borrow Foundry's rule."""
    from proto_tools.tools.inverse_folding import InverseFoldingStructureInput, LigandMPNNSampleConfig

    inp = InverseFoldingStructureInput(structure=get_gfp_structure())
    assert LigandMPNNSampleConfig(implementation="original").readable_positions(inp) is None
