"""tests/inverse_folding_tests/test_caliby.py.

Tests for Caliby sequence design, Potts-energy scoring, and sidechain packing.
"""

import math
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.caliby import (
    CalibyPackConfig,
    CalibyPackInput,
    CalibySampleConfig,
    CalibySampleInput,
    CalibyScoreConfig,
    CalibyScoreInput,
    CalibyScoreOutput,
    CalibyScoreStructureInput,
    CalibyScoringMetrics,
    CalibyStructureInput,
    run_caliby_pack,
    run_caliby_sample,
    run_caliby_score,
)
from proto_tools.tools.inverse_folding.caliby.caliby_sample import (
    DESIGN_CHECKPOINTS,
    PACKER_CHECKPOINTS,
    chain_residue_ids,
    restore_numbering,
    sequential_pdb,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "test_structure_similarity.pdb"
MULTI_CHAIN_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"
BENCHMARK_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"
PROTEIN_DNA_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "protein_dna_complex.pdb"

_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O"})
_ALL_CONFIG_CLASSES = (CalibySampleConfig, CalibyScoreConfig, CalibyPackConfig)


_persistent_tool = make_persistent_fixture("caliby")


def _atom_count(structure: Structure) -> int:
    """Total number of atoms across every chain of ``structure``."""
    return sum(len(residue) for model in structure.gemmi_struct for chain in model for residue in chain)


def _backbone_only(structure: Structure) -> Structure:
    """Return ``structure`` with every non-backbone atom dropped (N, CA, C, O retained)."""
    lines = [
        line
        for line in structure.structure_pdb.splitlines()
        if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() in _BACKBONE_ATOM_NAMES
    ]
    return Structure(structure="\n".join(lines) + "\n", structure_format="pdb", source="test")


def _renumber(structure: Structure, offset: int, gap_after: int, gap_size: int) -> Structure:
    """Shift residue numbering by ``offset`` and open a ``gap_size`` gap after residue ``gap_after``.

    Produces a structure whose numbering neither starts at 1 nor runs contiguously, so a
    1-indexed ``fixed_positions`` selection only lands on the intended residues when it is
    resolved through the chain's actual residue numbers.
    """
    lines = []
    for line in structure.structure_pdb.splitlines():
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) >= 26:
            number = int(line[22:26])
            shifted = number + offset + (gap_size if number > gap_after else 0)
            lines.append(line[:22] + f"{shifted:>4}" + line[26:])
        else:
            lines.append(line)
    return Structure(structure="\n".join(lines) + "\n", structure_format="pdb", source="test")


def _with_insertion_code(structure: Structure, after: int, code: str = "A") -> Structure:
    """Relabel residue ``after + 1`` as ``<after><code>`` and shift every later residue down by one.

    The chain keeps its residue count, but residue ``after`` and its insertion-coded neighbour
    now share a number, which only an insertion-code-aware mapping keeps apart.
    """
    lines = []
    for line in structure.structure_pdb.splitlines():
        relabelled = line
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) >= 27:
            number = int(line[22:26])
            if number == after + 1:
                relabelled = line[:22] + f"{after:>4}{code}" + line[27:]
            elif number > after + 1:
                relabelled = line[:22] + f"{number - 1:>4}" + line[26:]
        lines.append(relabelled)
    return Structure(structure="\n".join(lines) + "\n", structure_format="pdb", source="test")


@pytest.fixture(scope="module")
def pdb_structure():
    return Structure.from_file(TEST_PDB_FILE)


@pytest.fixture(scope="module")
def multi_chain_structure():
    return Structure.from_file(MULTI_CHAIN_PDB_FILE)


@pytest.fixture(scope="module")
def offset_numbered_structure(pdb_structure):
    """The test backbone renumbered to 101-130, 151-185: offset start plus an internal gap."""
    return _renumber(pdb_structure, offset=100, gap_after=30, gap_size=20)


@pytest.fixture(scope="module")
def insertion_coded_structure(pdb_structure):
    """The test backbone numbered 1..5, 5A, 6..64: position 6 is residue ``5A``."""
    return _with_insertion_code(pdb_structure, after=5)


@pytest.fixture(scope="module")
def benchmark_pdb_structure():
    return Structure.from_file(BENCHMARK_PDB_FILE)


@pytest.fixture(scope="module")
def benchmark_backbone_structure(benchmark_pdb_structure):
    return _backbone_only(benchmark_pdb_structure)


# ============================================================================
# Unit tests (no GPU required, schema and data model validation)
# ============================================================================
def test_caliby_structure_input_rejects_conformer_with_different_chains(multi_chain_structure):
    """A conformer missing one of the primary conformer's chains is rejected."""
    with pytest.raises(ValidationError, match="every conformer must match the primary conformer"):
        CalibyStructureInput(
            structure=multi_chain_structure,
            conformers=[multi_chain_structure.select_chains(["A"])],
        )


def test_caliby_structure_input_rejects_conformer_with_different_residue_count(pdb_structure, multi_chain_structure):
    """A conformer whose chain has a different residue count is rejected, and the message names it."""
    with pytest.raises(ValidationError, match="does not align with the primary structure") as exc:
        CalibyStructureInput(
            structure=pdb_structure,
            conformers=[multi_chain_structure.select_chains(["A"])],
        )
    assert "'A'" in str(exc.value)


def test_caliby_structure_input_accepts_renumbered_conformer(pdb_structure):
    """Upstream aligns on a per-chain ordinal, so absolute numbering may differ.

    A generative sampler renumbering conformers from 1 is the common case, and Caliby
    aggregates positionally, so matching residue counts are enough on their own.
    """
    renumbered = _renumber(pdb_structure, offset=100, gap_after=30, gap_size=20)
    assert renumbered.get_chain_positions("A") != pdb_structure.get_chain_positions("A")
    assert len(renumbered.get_chain_positions("A")) == len(pdb_structure.get_chain_positions("A"))

    inp = CalibyStructureInput(structure=pdb_structure, conformers=[renumbered])
    assert len(inp.conformers) == 1


def test_caliby_structure_input_accepts_renumbered_conformer_with_position_constraints(pdb_structure):
    """Every conformer is staged numbered 1..N, so a position names the same ordinal in each."""
    renumbered = _renumber(pdb_structure, offset=100, gap_after=30, gap_size=20)

    CalibyStructureInput(structure=pdb_structure, conformers=[renumbered], fixed_positions={"A": [5]})
    CalibyStructureInput(structure=pdb_structure, conformers=[renumbered], pinned_residues={"A": {5: "W"}})


def test_caliby_score_structure_input_rejects_misaligned_conformer(pdb_structure, multi_chain_structure):
    """The scoring input enforces the same conformer alignment contract as the design input."""
    with pytest.raises(ValidationError, match="does not align with the primary structure"):
        CalibyScoreStructureInput(
            structure=pdb_structure,
            conformers=[multi_chain_structure.select_chains(["A"])],
        )


def test_caliby_ensemble_structures_puts_primary_conformer_first(pdb_structure):
    """``ensemble_structures`` is the primary conformer followed by the supplied conformers."""
    assert CalibyStructureInput(structure=pdb_structure).ensemble_structures == [pdb_structure]

    conformer = Structure.from_file(TEST_PDB_FILE)
    inp = CalibyStructureInput(structure=pdb_structure, conformers=[conformer])
    assert inp.ensemble_structures[0] is pdb_structure
    assert inp.ensemble_structures[1] is conformer

    score_inp = CalibyScoreStructureInput(structure=pdb_structure, conformers=[conformer])
    assert score_inp.ensemble_structures[0] is pdb_structure


def test_caliby_sample_config_rejects_packer_checkpoint():
    """Design needs a seq_des model, so a packer checkpoint is rejected by name."""
    with pytest.raises(ValidationError, match="is not a Caliby design checkpoint"):
        CalibySampleConfig(model_checkpoint="caliby_packer_010")


def test_caliby_score_config_rejects_packer_checkpoint():
    """Scoring runs the design models, so a packer checkpoint is rejected by name."""
    with pytest.raises(ValidationError, match="is not a Caliby scoring checkpoint"):
        CalibyScoreConfig(model_checkpoint="caliby_packer_010")


def test_caliby_pack_config_rejects_design_checkpoint():
    """Packing needs a scn_pack model, so a design checkpoint is rejected by name."""
    with pytest.raises(ValidationError, match="is not a Caliby packing checkpoint"):
        CalibyPackConfig(model_checkpoint="caliby")


@pytest.mark.parametrize("config_class", _ALL_CONFIG_CLASSES)
def test_caliby_configs_reject_unknown_checkpoint(config_class):
    """A name belonging to neither checkpoint family is rejected by every config."""
    with pytest.raises(ValidationError, match="Unknown Caliby checkpoint"):
        config_class(model_checkpoint="caliby_v99")


@pytest.mark.parametrize("config_class", _ALL_CONFIG_CLASSES)
def test_caliby_configs_accept_custom_ckpt_path(config_class):
    """An explicit .ckpt path bypasses the bundled-name check on every config."""
    config = config_class(model_checkpoint="/models/custom_caliby.ckpt")
    assert config.model_checkpoint == "/models/custom_caliby.ckpt"


def test_setup_script_downloads_every_bundled_checkpoint():
    """setup.sh must fetch exactly the checkpoints the configs accept by name."""
    setup_sh = (
        Path(__file__).parents[2] / "proto_tools" / "tools" / "inverse_folding" / "caliby" / "standalone" / "setup.sh"
    ).read_text()
    downloaded = set(re.findall(r'^\s*"([a-z0-9_]+)",\s*$', setup_sh, flags=re.MULTILINE))
    assert downloaded == set(DESIGN_CHECKPOINTS) | set(PACKER_CHECKPOINTS)


@pytest.mark.parametrize("config_class", _ALL_CONFIG_CLASSES)
def test_caliby_configs_block_remote_execution_for_custom_checkpoint(config_class):
    """A local .ckpt file is not present on a hosted worker, so remote execution is refused."""
    reason = config_class(model_checkpoint="/models/custom_caliby.ckpt").remote_unsupported_reason("proto")
    assert reason is not None
    assert ".ckpt" in reason


@pytest.mark.parametrize("config_class", _ALL_CONFIG_CLASSES)
def test_caliby_configs_allow_remote_execution_for_bundled_checkpoint(config_class):
    """A bundled checkpoint ships with the hosted worker, so remote execution is allowed."""
    assert config_class().remote_unsupported_reason("proto") is None


def test_caliby_sample_config_deduplicates_excluded_amino_acids():
    """Repeated exclusions collapse to one entry each, in first-seen order."""
    config = CalibySampleConfig(excluded_amino_acids=["C", "M", "C"])
    assert config.excluded_amino_acids == ["C", "M"]


def test_caliby_sample_config_folds_empty_exclusions_to_none():
    """An empty exclusion list means no exclusions at all."""
    assert CalibySampleConfig(excluded_amino_acids=[]).excluded_amino_acids is None


@pytest.mark.parametrize("as_structure", [False, True], ids=["path", "structure"])
def test_caliby_pack_input_wraps_a_bare_item(as_structure, pdb_structure):
    """A bare path or Structure is wrapped into a one-item structure list."""
    inp = CalibyPackInput(structures=pdb_structure if as_structure else str(TEST_PDB_FILE))
    assert len(inp.structures) == 1
    assert inp.structures[0].get_chain_ids() == ["A"]


@pytest.mark.parametrize("file_format", ["csv", "json"])
def test_caliby_score_output_export(file_format, tmp_path):
    """Scoring output exports to csv and json, and the csv carries the sequence column."""
    output = CalibyScoreOutput(
        scores=[CalibyScoringMetrics(potts_energy=-120.5, avg_potts_energy=-1.85, sequence="AGSM:KTV")]
    )
    output.export("scores", export_path=tmp_path, file_format=file_format)

    exported = tmp_path / f"scores.{file_format}"
    assert exported.stat().st_size > 0
    if file_format == "csv":
        text = exported.read_text()
        assert text.splitlines()[0].startswith("sequence,")
        assert "AGSM:KTV" in text


def _write_homodimer(tmp_path, *, waters: bool = False) -> Structure:
    """Two identical copies of the benchmark monomer, offset so the chains do not clash."""
    lines = [line for line in BENCHMARK_PDB_FILE.read_text().splitlines(keepends=True) if line.startswith("ATOM")]
    doubled = list(lines)
    for line in lines:
        shifted = float(line[30:38]) + 60.0
        doubled.append(f"{line[:21]}B{line[22:30]}{shifted:8.3f}{line[38:]}")
    if waters:
        # Deposited structures number waters independently per chain, which must not be read
        # as the polymer chains carrying different residue numbering.
        serial = 9000
        for chain_id, first in (("A", 501), ("B", 701)):
            for offset in range(3):
                serial += 1
                doubled.append(
                    f"HETATM{serial:5d}  O   HOH {chain_id}{first + offset:4d}"
                    "      50.000  50.000  50.000  1.00 30.00           O  \n"
                )
    path = tmp_path / "homodimer.pdb"
    path.write_text("".join(doubled) + "END\n")
    return Structure(structure=str(path))


def test_symmetric_chains_accepts_a_single_bare_group(tmp_path):
    """``["A", "B"]`` is one group, not two groups of one chain."""
    inp = CalibyStructureInput(structure=_write_homodimer(tmp_path), symmetric_chains=["A", "B"])
    assert inp.symmetric_chains is not None
    assert [list(group.chains) for group in inp.symmetric_chains] == [["A", "B"]]


def test_symmetric_chains_rejects_a_group_of_one(tmp_path):
    with pytest.raises(ValidationError, match="must tie at least two chains"):
        CalibyStructureInput(structure=_write_homodimer(tmp_path), symmetric_chains=[["A"]])


def test_symmetric_chains_rejects_unknown_chain(tmp_path):
    """Each group is validated against the structure under its own field label."""
    with pytest.raises(ValidationError, match=r"symmetric_chains: chain\(s\) \['Z'\] not in structure"):
        CalibyStructureInput(structure=_write_homodimer(tmp_path), symmetric_chains=[["A", "Z"]])


def test_symmetric_chains_rejects_chain_in_two_groups(tmp_path):
    structure = _write_homodimer(tmp_path)
    with pytest.raises(ValidationError, match="more than one symmetric_chains group"):
        CalibyStructureInput(structure=structure, symmetric_chains=[["A", "B"], ["B", "A"]])


def test_symmetric_chains_rejects_chain_held_fixed_by_chains_to_redesign(tmp_path):
    """A tied chain that is not redesigned keeps its input sequence, so the tie would not hold."""
    structure = _write_homodimer(tmp_path)
    with pytest.raises(ValidationError, match="that chains_to_redesign holds fixed"):
        CalibyStructureInput(structure=structure, symmetric_chains=["A", "B"], chains_to_redesign=["B"])


def test_symmetric_chains_rejects_position_constraints_on_a_tied_chain(tmp_path):
    """Conditioned positions are not rewritten with the tied residue, so the two conflict."""
    structure = _write_homodimer(tmp_path)
    with pytest.raises(ValidationError, match="that symmetric_chains ties"):
        CalibyStructureInput(structure=structure, symmetric_chains=["A", "B"], fixed_positions={"A": [5]})
    with pytest.raises(ValidationError, match="that symmetric_chains ties"):
        CalibyStructureInput(structure=structure, symmetric_chains=["A", "B"], pinned_residues={"A": {5: "W"}})


def test_symmetric_chains_rejects_mismatched_residue_numbering():
    """Caliby ties by residue number, so chains of differing length cannot be tied."""
    with pytest.raises(ValidationError, match="different residue numbering"):
        CalibyStructureInput(structure=str(MULTI_CHAIN_PDB_FILE), symmetric_chains=[["A", "B"]])


def test_pinned_residues_accepts_per_chain_shorthand(pdb_structure):
    """The bare ``{chain: {position: residue}}`` form coerces to the canonical shape."""
    inp = CalibyStructureInput(structure=pdb_structure, pinned_residues={"A": {3: "W"}})
    assert inp.pinned_residues is not None
    assert inp.pinned_residues.chains == {"A": {3: "W"}}


def test_pinned_residues_rejects_unknown_chain(pdb_structure):
    """The selection is validated against the structure under its own field label."""
    with pytest.raises(ValidationError, match="pinned_residues: chain 'Z' not in structure"):
        CalibyStructureInput(structure=pdb_structure, pinned_residues={"Z": {3: "W"}})


def test_pinned_residues_rejects_position_past_chain_end(pdb_structure):
    beyond = len(pdb_structure.get_chain_positions("A")) + 1
    with pytest.raises(ValidationError, match="invalid positions"):
        CalibyStructureInput(structure=pdb_structure, pinned_residues={"A": {beyond: "W"}})


def test_pinned_residues_rejects_non_positive_position(pdb_structure):
    with pytest.raises(ValidationError, match="must be 1-indexed"):
        CalibyStructureInput(structure=pdb_structure, pinned_residues={"A": {0: "W"}})


def test_pinned_residues_rejects_overlap_with_fixed_positions(pdb_structure):
    """Caliby conditions on overridden positions, so naming one as fixed too is contradictory."""
    with pytest.raises(ValidationError, match="appear in both pinned_residues and fixed_positions"):
        CalibyStructureInput(structure=pdb_structure, pinned_residues={"A": {3: "W"}}, fixed_positions={"A": [3]})


def test_chain_residue_ids_keep_insertion_codes(insertion_coded_structure):
    """``5`` and ``5A`` are distinct residues, where integer numbering collapses them."""
    residue_ids = chain_residue_ids(insertion_coded_structure)["A"]
    assert residue_ids[4:7] == [(5, ""), (5, "A"), (6, "")]
    assert insertion_coded_structure.get_chain_positions("A")[4:6] == [5, 5]


def test_sequential_pdb_numbers_each_residue_by_its_position(insertion_coded_structure):
    """The staged copy numbers residues 1..N with no insertion codes, sequence unchanged."""
    staged = Structure(structure=sequential_pdb(insertion_coded_structure), structure_format="pdb", source="test")
    assert chain_residue_ids(staged)["A"] == [(n, "") for n in range(1, 66)]
    assert staged.get_chain_sequence("A") == insertion_coded_structure.get_chain_sequence("A")


def test_restore_numbering_round_trips_insertion_codes(insertion_coded_structure):
    """An output written in staged numbering gets the input residue IDs back."""
    residue_ids = chain_residue_ids(insertion_coded_structure)
    staged = Structure(structure=sequential_pdb(insertion_coded_structure), structure_format="pdb", source="test")
    restored = restore_numbering(staged.structure_cif, residue_ids, source="test")
    assert chain_residue_ids(restored) == residue_ids


def test_restore_numbering_rejects_a_chain_of_different_length(pdb_structure, insertion_coded_structure):
    """A residue-count mismatch cannot be mapped back by position, so it raises."""
    shortened = {"A": chain_residue_ids(insertion_coded_structure)["A"][:-1]}
    with pytest.raises(ValueError, match="numbering cannot be restored"):
        restore_numbering(pdb_structure.structure_cif, shortened, source="test")


def test_symmetric_chains_rejects_numbering_that_differs_only_by_insertion_code(tmp_path):
    """``5A`` on one chain and ``5B`` on the other are different residues, so they cannot be tied."""
    homodimer = _write_homodimer(tmp_path)
    lines = []
    for line in homodimer.structure_pdb.splitlines():
        if line.startswith("ATOM") and int(line[22:26]) == 6:
            # Chain A's residue 6 becomes 5A, chain B's becomes 5B.
            lines.append(line[:22] + f"{5:>4}" + line[21] + line[27:])
        else:
            lines.append(line)
    structure = Structure(structure="\n".join(lines) + "\n", structure_format="pdb", source="test")
    # Integer numbering cannot tell the chains apart.
    assert structure.get_chain_positions("A") == structure.get_chain_positions("B")

    with pytest.raises(ValidationError, match="different residue numbering"):
        CalibyStructureInput(structure=structure, symmetric_chains=["A", "B"])


# ============================================================================
# Integration tests (GPU required, run with --integration or --all)
# ============================================================================
@pytest.mark.uses_gpu
def test_caliby_score_batch_size_does_not_change_the_ensemble_energy(pdb_structure):
    """batch_size only chunks conformers through the model; the aggregate must be unchanged."""
    ensemble = CalibyScoreStructureInput(structure=pdb_structure, conformers=[pdb_structure, pdb_structure])
    energies = []
    for batch_size in (1, 4):
        output = run_caliby_score(CalibyScoreInput(inputs=[ensemble]), CalibyScoreConfig(batch_size=batch_size))
        assert output.success, f"Failed to score with batch_size={batch_size}: {output}"
        energies.append(float(output.scores[0]["potts_energy"]))
    assert energies[0] == pytest.approx(energies[1], rel=1e-5)


@pytest.mark.uses_gpu
def test_caliby_sample_pins_supplied_residue_identities(pdb_structure):
    """A pinned identity appears in the design even where the structure lacks it."""
    inp = CalibySampleInput(
        inputs=[CalibyStructureInput(structure=pdb_structure, pinned_residues={"A": {3: "W", 4: "W", 5: "W"}})]
    )
    output = run_caliby_sample(inp, CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=20, seed=0))
    assert output.success, f"Failed to design with pinned residues: {output}"

    designed = output.design_sets[0].complexes[0].chains[0].sequence
    assert designed[2:5] == "WWW"
    assert len(designed) == len(pdb_structure.get_chain_sequence("A"))


@pytest.mark.uses_gpu
@pytest.mark.parametrize("waters", [False, True], ids=["dry", "solvated"])
def test_caliby_sample_ties_symmetric_chains(tmp_path, waters):
    """Tied chains come out identical, including when per-chain waters are present.

    Waters are numbered independently per chain, so reading them as polymer numbering
    would reject a solvated homooligomer outright.
    """
    structure = _write_homodimer(tmp_path, waters=waters)
    output = run_caliby_sample(
        CalibySampleInput(inputs=[CalibyStructureInput(structure=structure, symmetric_chains=["A", "B"])]),
        CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=20, seed=0),
    )
    assert output.success, f"Failed to design with symmetric_chains: {output}"
    chains = output.design_sets[0].complexes[0].chain_sequence_map()
    assert chains["A"] == chains["B"]


@pytest.mark.uses_gpu
def test_caliby_sample_designs_untied_chains_independently(tmp_path):
    """Two identical untied chains diverge, so the tie above is not holding vacuously."""
    structure = _write_homodimer(tmp_path)
    output = run_caliby_sample(
        CalibySampleInput(inputs=[CalibyStructureInput(structure=structure)]),
        CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=50, seed=0),
    )
    assert output.success, f"Failed to design: {output}"
    chains = output.design_sets[0].complexes[0].chain_sequence_map()
    assert chains["A"] != chains["B"]


@pytest.mark.uses_gpu
def test_caliby_sample_accepts_chains_with_unmodelled_gaps(tmp_path):
    """A chain break must not be mistaken for residues dropped by preprocessing."""
    # Drop an interior stretch to leave a chain break, as an experimental structure would have.
    kept = [
        line
        for line in BENCHMARK_PDB_FILE.read_text().splitlines(keepends=True)
        if not (line.startswith("ATOM") and 100 <= int(line[22:26]) <= 103)
    ]
    gapped = tmp_path / "gapped.pdb"
    gapped.write_text("".join(kept))

    structure = Structure(structure=str(gapped))
    # get_chain_sequences() pads each break with "-", so it runs longer than the residue count.
    assert len(structure.get_chain_sequences()["A"]) > len(structure.get_chain_positions("A"))

    inp = CalibySampleInput(inputs=[CalibyStructureInput(structure=structure)])
    output = run_caliby_sample(inp, CalibySampleConfig(potts_sweeps=20, seed=0))
    assert output.success, f"Failed to design a gapped structure: {output}"

    design = output.design_sets[0].complexes[0]
    for chain in design.chains:
        assert len(chain.sequence) == len(structure.get_chain_positions(chain.id))


@pytest.mark.uses_gpu
def test_caliby_sample_rejects_nucleic_acid_chains():
    """A nucleic acid chain is dropped by the featurizer, so the chain mismatch is reported."""
    structure = Structure(structure=str(PROTEIN_DNA_PDB_FILE))
    # Caliby keeps peptide polymers only, so the two DNA strands never reach the model.
    assert len(structure.get_chain_ids()) == 3

    inp = CalibySampleInput(inputs=[CalibyStructureInput(structure=structure)])
    with pytest.raises(Exception, match="Supply protein chains only"):
        run_caliby_sample(inp, CalibySampleConfig(potts_sweeps=10, seed=0))


@pytest.mark.uses_gpu
def test_caliby_sample_simple(pdb_structure):
    """Basic Caliby sequence design with a single structure."""
    inp = CalibySampleInput(inputs=[CalibyStructureInput(structure=pdb_structure)])
    config = CalibySampleConfig(num_sequences_per_structure=2, potts_sweeps=100, seed=42)
    output = run_caliby_sample(inp, config)
    assert output.success, f"Failed to design: {output}"
    validate_output(output)
    assert output.tool_id == "caliby-sample"

    assert len(output.design_sets) == 1
    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 2

    native = pdb_structure.get_chain_sequence("A")
    for design in design_set.complexes:
        assert design.chain_sequence_map() == {"A": design.chains[0].sequence}
        assert len(design.chains[0].sequence) == len(native)
        # Designs are backbone-only CIF, so parsing must recover the input topology.
        assert design.structure.structure_format == "cif"
        assert design.structure.structure.strip()
        assert design.structure.get_chain_ids() == ["A"]
        assert isinstance(design.metrics["potts_energy"], float)
        assert isinstance(design.metrics["avg_potts_energy"], float)
        assert isinstance(design.metrics["sequence_recovery"], float)
        assert math.isfinite(design.metrics["potts_energy"])
        assert 0.0 <= design.metrics["sequence_recovery"] <= 1.0
    assert_metrics_in_spec(output)


@pytest.mark.uses_gpu
def test_caliby_sample_improves_on_native_potts_energy(pdb_structure):
    """A Caliby design reaches a lower Potts energy than the structure's native sequence."""
    native = run_caliby_score(CalibyScoreInput(inputs=[CalibyScoreStructureInput(structure=pdb_structure)]))
    assert native.success, f"Failed to score native sequence: {native}"

    designed = run_caliby_sample(
        CalibySampleInput(inputs=[CalibyStructureInput(structure=pdb_structure)]),
        CalibySampleConfig(num_sequences_per_structure=1, seed=42),
    )
    assert designed.success, f"Failed to design: {designed}"

    design_energy = designed.design_sets[0].complexes[0].metrics["potts_energy"]
    native_energy = native.scores[0]["potts_energy"]
    assert design_energy < native_energy, (
        f"Designed sequence energy {design_energy} is not below native energy {native_energy}"
    )


@pytest.mark.uses_gpu
def test_caliby_sample_ensemble_conditioned(pdb_structure):
    """Ensemble-conditioned design aggregates Potts parameters over the supplied conformers."""
    inp = CalibySampleInput(
        inputs=[
            CalibyStructureInput(
                structure=pdb_structure,
                conformers=[Structure.from_file(TEST_PDB_FILE)],
            )
        ]
    )
    config = CalibySampleConfig(num_sequences_per_structure=2, potts_sweeps=100, seed=42)
    output = run_caliby_sample(inp, config)
    assert output.success, f"Ensemble-conditioned design failed: {output}"
    validate_output(output)
    assert output.tool_id == "caliby-sample"

    assert len(output.design_sets) == 1
    design_set = output.design_sets[0]
    assert len(design_set.complexes) == 2
    native_length = len(pdb_structure.get_chain_sequence("A"))
    for design in design_set.complexes:
        assert len(design.chains[0].sequence) == native_length
        assert design.structure.structure_format == "cif"
        assert design.structure.get_chain_ids() == ["A"]
    assert_metrics_in_spec(output)


@pytest.mark.uses_gpu
def test_caliby_sample_respects_fixed_positions(offset_numbered_structure):
    """Fixed positions keep their native residue on a structure with offset, gapped numbering."""
    native = offset_numbered_structure.get_chain_sequence("A")
    fixed = [1, 5, 33, 65]

    inp = CalibySampleInput(
        inputs=[CalibyStructureInput(structure=offset_numbered_structure, fixed_positions={"A": fixed})]
    )
    config = CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=100, seed=42)
    output = run_caliby_sample(inp, config)
    assert output.success, f"Fixed-position design failed: {output}"

    designed = output.design_sets[0].complexes[0].chains[0].sequence
    assert len(designed) == len(native)
    for position in fixed:
        assert designed[position - 1] == native[position - 1], (
            f"Fixed position {position} changed from {native[position - 1]} to {designed[position - 1]}"
        )


@pytest.mark.uses_gpu
def test_caliby_sample_resolves_positions_past_an_insertion_code(insertion_coded_structure):
    """Pins and fixed positions land on their own residue when a neighbour shares its number."""
    native = insertion_coded_structure.get_chain_sequence("A")
    # Positions 5 and 6 are residues 5 and 5A; integer numbering would map both to residue 5.
    inp = CalibySampleInput(
        inputs=[
            CalibyStructureInput(
                structure=insertion_coded_structure,
                pinned_residues={"A": {5: "W", 6: "Y"}},
                fixed_positions={"A": [7, 40]},
            )
        ]
    )
    output = run_caliby_sample(inp, CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=20, seed=0))
    assert output.success, f"Failed to design an insertion-coded structure: {output}"

    design = output.design_sets[0].complexes[0]
    designed = design.chains[0].sequence
    assert len(designed) == len(native)
    assert designed[4:6] == "WY"
    assert designed[6] == native[6]
    assert designed[39] == native[39]
    assert chain_residue_ids(design.structure) == chain_residue_ids(insertion_coded_structure)
    assert design.structure.get_chain_sequence("A") == designed


@pytest.mark.uses_gpu
def test_caliby_sample_redesigns_selected_chain_only(multi_chain_structure):
    """With chains_to_redesign=['A'], chain B keeps its native sequence and is flagged undesigned."""
    inp = CalibySampleInput(inputs=[CalibyStructureInput(structure=multi_chain_structure, chains_to_redesign=["A"])])
    config = CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=100, seed=42)
    output = run_caliby_sample(inp, config)
    assert output.success, f"Chain-selective design failed: {output}"

    design = output.design_sets[0].complexes[0]
    native = multi_chain_structure.get_chain_sequences()
    assert [chain.id for chain in design.chains] == ["A", "B"]
    assert design.designed == [True, False]
    assert design.chain_sequence_map()["B"] == native["B"]
    assert len(design.chain_sequence_map()["A"]) == len(native["A"])


@pytest.mark.uses_gpu
def test_caliby_sample_unseeded_divergence(pdb_structure):
    """Unseeded runs draw a fresh base seed, so they diverge."""
    inp = CalibySampleInput(inputs=[CalibyStructureInput(structure=pdb_structure)])
    # Sampling anneals down to `temperature`, so a warmer final temperature keeps the two
    # unseeded runs from collapsing onto the same near-greedy sequence.
    config = CalibySampleConfig(num_sequences_per_structure=1, potts_sweeps=100, temperature=0.5)

    first = run_caliby_sample(inp, config)
    second = run_caliby_sample(inp, config)
    assert first.success and second.success

    assert first.design_sets[0].complexes[0].chain_sequences != second.design_sets[0].complexes[0].chain_sequences


@pytest.mark.uses_gpu
def test_caliby_score_native_sequence(pdb_structure):
    """Scoring reports the Potts energy of the sequence the structure carries."""
    inp = CalibyScoreInput(inputs=[CalibyScoreStructureInput(structure=pdb_structure)])
    output = run_caliby_score(inp, CalibyScoreConfig())
    assert output.success, f"Failed to score: {output}"
    validate_output(output)
    assert output.tool_id == "caliby-score"

    assert len(output.scores) == 1
    score = output.scores[0]
    assert score.sequence.replace(":", "") == pdb_structure.get_chain_sequence("A")
    assert isinstance(score["potts_energy"], float)
    assert math.isfinite(score["potts_energy"])
    assert score["avg_potts_energy"] == pytest.approx(score["potts_energy"] / len(score.sequence.replace(":", "")))
    assert score.local_conditional_energies is None
    assert_metrics_in_spec(output)


@pytest.mark.uses_gpu
def test_caliby_score_local_energies_matrix(multi_chain_structure):
    """``return_local_energies`` adds a (seq_len, len(potts_tokens)) matrix."""
    inp = CalibyScoreInput(inputs=[CalibyScoreStructureInput(structure=multi_chain_structure)])
    output = run_caliby_score(inp, CalibyScoreConfig(return_local_energies=True))
    assert output.success, f"Failed to score with local energies: {output}"

    score = output.scores[0]
    assert score.potts_tokens
    assert output.potts_tokens == score.potts_tokens
    assert score.local_conditional_energies is not None
    # Chain separators are not scored positions, so they do not get a row.
    assert len(score.local_conditional_energies) == len(score.sequence.replace(":", ""))
    assert all(len(row) == len(score.potts_tokens) for row in score.local_conditional_energies)


@pytest.mark.uses_gpu
def test_caliby_pack_adds_sidechains(pdb_structure):
    """Packing a backbone-only structure places sidechain atoms without changing the sequence."""
    backbone = _backbone_only(pdb_structure)
    inp = CalibyPackInput(structures=[backbone])
    output = run_caliby_pack(inp, CalibyPackConfig(diffusion_steps=10, seed=42))
    assert output.success, f"Failed to pack: {output}"
    validate_output(output)
    assert output.tool_id == "caliby-pack"

    assert len(output.packed_structures) == 1
    packed = output.packed_structures[0]
    assert packed.structure_format == "cif"
    assert packed.structure.strip()
    assert packed.get_chain_ids() == ["A"]
    assert packed.get_chain_sequence("A") == pdb_structure.get_chain_sequence("A")
    assert _atom_count(packed) > _atom_count(backbone)


@pytest.mark.uses_gpu
def test_caliby_pack_keeps_insertion_coded_numbering(insertion_coded_structure):
    """The packed structure comes back in the input's numbering, insertion codes included."""
    output = run_caliby_pack(
        CalibyPackInput(structures=[_backbone_only(insertion_coded_structure)]),
        CalibyPackConfig(diffusion_steps=10, seed=42),
    )
    assert output.success, f"Failed to pack an insertion-coded structure: {output}"

    packed = output.packed_structures[0]
    assert chain_residue_ids(packed) == chain_residue_ids(insertion_coded_structure)
    assert packed.get_chain_sequence("A") == insertion_coded_structure.get_chain_sequence("A")


# ============================================================================
# Benchmarks
# ============================================================================
@pytest.mark.benchmark("caliby-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_caliby_sample_benchmark(request: pytest.FixtureRequest, benchmark_pdb_structure: Structure) -> None:
    """Benchmark caliby-sample: 8 designs of renin (~340 aa), single structure (cold + warm).

    Keeps the default 500 Potts sweeps and the default ``caliby`` checkpoint, so each
    design pays the full featurization plus MCMC cost on a realistic monomer.
    """
    inputs = CalibySampleInput(inputs=[CalibyStructureInput(structure=benchmark_pdb_structure)])
    config = CalibySampleConfig(num_sequences_per_structure=8, seed=0)

    result = benchmark_twice(request, "caliby", lambda: run_caliby_sample(inputs, config))

    assert result.tool_id == "caliby-sample"
    assert len(result.design_sets) == 1
    design_set = result.design_sets[0]
    assert len(design_set.complexes) == 8
    target_len = len(benchmark_pdb_structure.get_chain_sequence("A"))
    for design in design_set.complexes:
        assert len(design.chains[0].sequence) == target_len
        assert design.structure.structure.strip()


@pytest.mark.benchmark("caliby-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_caliby_score_benchmark(
    request: pytest.FixtureRequest, benchmark_pdb_structure: Structure, multi_chain_structure: Structure
) -> None:
    """Benchmark caliby-score: renin (~340 aa) plus a 2-chain complex (233 aa) with local energies.

    Two distinct structures rather than repeats, so per-item dedup cannot collapse the
    workload, and ``return_local_energies=True`` includes the full per-position matrix.
    """
    inputs = CalibyScoreInput(
        inputs=[
            CalibyScoreStructureInput(structure=benchmark_pdb_structure),
            CalibyScoreStructureInput(structure=multi_chain_structure),
        ]
    )
    config = CalibyScoreConfig(return_local_energies=True)

    result = benchmark_twice(request, "caliby", lambda: run_caliby_score(inputs, config))

    assert result.tool_id == "caliby-score"
    assert len(result.scores) == 2
    for score in result.scores:
        assert isinstance(score["potts_energy"], float)
        assert score.local_conditional_energies is not None
        assert len(score.local_conditional_energies) == len(score.sequence.replace(":", ""))


@pytest.mark.benchmark("caliby-pack")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_caliby_pack_benchmark(request: pytest.FixtureRequest, benchmark_backbone_structure: Structure) -> None:
    """Benchmark caliby-pack: 4 sidechain packs of a backbone-only renin (~340 aa) (cold + warm).

    Keeps the default 50 diffusion steps and the recommended ``caliby_packer_010``
    checkpoint, so each pack runs the full denoising trajectory.
    """
    inputs = CalibyPackInput(structures=[benchmark_backbone_structure] * 4)
    config = CalibyPackConfig(seed=0)

    result = benchmark_twice(request, "caliby", lambda: run_caliby_pack(inputs, config))

    assert result.tool_id == "caliby-pack"
    assert len(result.packed_structures) == 4
    backbone_atoms = _atom_count(benchmark_backbone_structure)
    for packed in result.packed_structures:
        assert packed.structure.strip()
        assert _atom_count(packed) > backbone_atoms
