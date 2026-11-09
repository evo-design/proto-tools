"""tests/inverse_folding_tests/test_structure_staging.py.

Tests for staging a structure for an inverse-folding model and reading its output back.

Pure structure bookkeeping: CPU-only, environment-free, no model execution.
"""

from pathlib import Path

import pytest

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.structure_staging import staged_ensemble, staged_structure

# ── StagedStructure ─────────────────────────────────────────────────────────


def test_staged_structure_exposes_the_file_and_the_maps_to_read_output_back(insertion_coded_structure):
    """Every residue reaches the model, each numbered by its position."""
    with staged_structure(insertion_coded_structure) as staged:
        assert staged.path.exists()
        written = Structure(structure=staged.path.read_text(), structure_format="pdb", source="test")
        assert written.get_chain_residue_ids()["A"] == [(n, "") for n in range(1, 66)]
        assert staged.residue_ids == insertion_coded_structure.get_chain_residue_ids()
        assert staged.chain_map == {"A": "A"}


def test_staged_structure_translates_chain_ids_both_ways(two_character_chain_structure):
    """A chain ID shortened for PDB staging resolves in and back out."""
    with staged_structure(two_character_chain_structure) as staged:
        [short_id] = staged.chains(["AB"])
        assert short_id == staged.chain_map["AB"]
        assert staged.original_chain(short_id) == "AB"
        assert staged.original_chain("Z") == "Z"


def test_staged_structure_positions_are_ordinals_on_staged_chains(two_character_chain_structure):
    """A 1-indexed position is already the staged residue number, so only chain IDs change."""
    from proto_tools.entities.structures import ResidueSelection

    with staged_structure(two_character_chain_structure) as staged:
        resolved = staged.positions(ResidueSelection(chains={"AB": [1, 7]}))
        assert resolved == {staged.chain_map["AB"]: [1, 7]}
        assert staged.positions(None) is None


def test_staged_structure_restores_output_written_in_staged_numbering(insertion_coded_structure):
    """The structure a model writes comes back in the input's chain IDs and numbering."""
    with staged_structure(insertion_coded_structure) as staged:
        model_output = staged.path.read_text()
    restored = staged.restore(model_output, source="test")
    assert restored.get_chain_residue_ids() == insertion_coded_structure.get_chain_residue_ids()


def _staged_output_without_residue(staged, chain_id: str, number: int) -> str:
    """The staged file as a model would write it back, minus one polymer residue."""
    import gemmi

    written = gemmi.read_pdb_string(staged.path.read_text())
    written.setup_entities()
    chain = written[0][chain_id]
    for index, residue in enumerate(chain):
        if residue.seqid.num == number and residue.name != "HOH":
            del chain[index]
            break
    written.setup_entities()
    return written.make_pdb_string()


def test_restoring_output_that_skips_a_residue_matches_the_positions_the_model_read():
    """A model that skips a residue writes a shorter structure, which still restores.

    FAMPNN and LigandMPNN pack only the residues their parser reads, so the output has one
    residue fewer than the input. Matching it against the whole chain's IDs by position
    would silently shift every residue past the skipped one.
    """
    from proto_tools.entities.structures.examples import get_gfp_structure
    from proto_tools.tools.inverse_folding.ligandmpnn._readable import ligandmpnn_readable_positions

    structure = get_gfp_structure()
    readable = ligandmpnn_readable_positions(structure)
    assert readable["A"] == [p for p in range(1, 229) if p != 64], "GFP's chromophore is skipped"

    with staged_structure(structure) as staged:
        model_output = _staged_output_without_residue(staged, "A", 64)

    restored = staged.restore(model_output, source="test", readable_positions=readable)

    expected = [ids for position, ids in enumerate(structure.get_chain_residue_ids()["A"], start=1) if position != 64]
    assert restored.get_chain_residue_ids()["A"] == expected, "every residue keeps its own input ID"


def test_restoring_a_short_output_without_the_mapping_is_rejected():
    """Without the mapping there is no way to tell which residue the model dropped."""
    from proto_tools.entities.structures.examples import get_gfp_structure

    with staged_structure(get_gfp_structure()) as staged:
        model_output = _staged_output_without_residue(staged, "A", 64)

    with pytest.raises(ValueError, match="residue IDs were given for it"):
        staged.restore(model_output, source="test")


def test_staging_can_write_a_modified_residue_as_an_atom_record():
    """A parser that keys on the record type loses a modified residue unless it is staged ATOM.

    LigandMPNN is that parser: AtomWorks moves every ``HETATM`` into a non-polymer chain
    before its own ``MSE``-to-``MET`` conversion can run, so selenomethionine is read as
    ligand context and the packed structure comes back with a chain the input cannot name.
    """
    from proto_tools.entities.structures.structure import Structure as _Structure

    source = Path(__file__).parent.parent / "dummy_data" / "modified_gapped_complex.pdb"
    structure = _Structure.from_file(source)

    with staged_structure(structure) as staged:
        default = staged.path.read_text()
    with staged_structure(structure, atom_record_residues=frozenset({"MSE"})) as staged:
        rewritten = staged.path.read_text()

    def record_types(text: str, residue: str) -> set[str]:
        return {
            line[:6].strip()
            for line in text.splitlines()
            if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == residue
        }

    assert record_types(default, "MSE") == {"HETATM"}, "the input's record type is kept by default"
    assert record_types(rewritten, "MSE") == {"ATOM"}, "and rewritten only for the named residues"
    assert record_types(rewritten, "SEP") == {"HETATM"}, "an unnamed modified residue is untouched"


def test_staged_ensemble_stages_every_conformer(insertion_coded_structure, staging_structure):
    """Conformers are staged alongside the primary, which carries the maps."""
    with staged_ensemble([insertion_coded_structure, staging_structure]) as (staged, conformer_paths):
        assert len(conformer_paths) == 1
        assert all(path.exists() for path in [staged.path, *conformer_paths])
        assert staged.residue_ids == insertion_coded_structure.get_chain_residue_ids()


# ── Every sample tool stages its input ──────────────────────────────────────


def _proteinmpnn_case(native):
    from proto_tools.tools.inverse_folding import ProteinMPNNSampleConfig, run_proteinmpnn_sample

    return (
        "proteinmpnn._sampling",
        run_proteinmpnn_sample,
        ProteinMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
        {"seq": [native], "score": [-1.0], "seqid": [1.0]},
    )


def _esm_if1_case(native):
    from proto_tools.tools.inverse_folding import ESMIF1SampleConfig, run_esm_if1_sample

    return (
        "esm_if1.esm_if1_sample",
        run_esm_if1_sample,
        ESMIF1SampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
        {"sequences": [native], "metrics": [{"avg_log_likelihood": -1.0}]},
    )


def _ligandmpnn_case(native):
    from proto_tools.tools.inverse_folding import LigandMPNNSampleConfig, run_ligandmpnn_sample

    return (
        "ligandmpnn.ligandmpnn_sample",
        run_ligandmpnn_sample,
        LigandMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
        {
            "chain_sequences": [[{"id": "A", "sequence": native}]],
            "metrics": [{"sequence_recovery": 1.0}],
            "pdb_strings": [None],
        },
    )


@pytest.mark.parametrize(
    "case", [_proteinmpnn_case, _esm_if1_case, _ligandmpnn_case], ids=["proteinmpnn", "esm_if1", "ligandmpnn"]
)
def test_sample_tools_hand_the_model_a_sequentially_numbered_structure(monkeypatch, insertion_coded_structure, case):
    """Staging is what lets these models see both residues of an insertion-coded pair."""
    from proto_tools.tools.inverse_folding import InverseFoldingInput, InverseFoldingStructureInput

    native = insertion_coded_structure.get_chain_sequence("A")
    module, run, config, payload = case(native)
    captured: dict[str, str] = {}

    def fake_dispatch(toolkit, input_dict, *, instance=None, config=None):
        captured["pdb"] = Path(input_dict["pdb_path"]).read_text()
        captured["chain_ids"] = input_dict["chain_ids"]
        return payload

    monkeypatch.setattr(
        f"proto_tools.tools.inverse_folding.{module}.ToolInstance.dispatch", staticmethod(fake_dispatch)
    )
    design = (
        run(
            InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=insertion_coded_structure)]),
            config,
        )
        .design_sets[0]
        .complexes[0]
    )

    staged = Structure(structure=captured["pdb"], structure_format="pdb", source="test")
    assert staged.get_chain_residue_ids()["A"] == [(n, "") for n in range(1, len(native) + 1)]
    assert captured["chain_ids"] == ["A"]
    assert design.chain_sequence_map() == {"A": native}


def test_staging_a_real_antibody_keeps_every_cdr_residue(antibody_structure):
    """PDB 1IGY's Kabat insertions are the case this staging exists for: all 40 residues reach the model."""
    with staged_structure(antibody_structure) as staged:
        written = Structure(structure=staged.path.read_text(), structure_format="pdb", source="test")
        assert written.get_chain_residue_ids("B")["B"] == [(n, "") for n in range(1, 41)]
        assert written.get_chain_sequence("B") == antibody_structure.get_chain_sequence("B")
        model_output = staged.path.read_text()

    restored = staged.restore(model_output, source="test")
    assert restored.get_chain_residue_ids() == antibody_structure.get_chain_residue_ids()


def test_sample_tool_designs_every_residue_of_a_real_antibody(monkeypatch, antibody_structure):
    """A design over an insertion-coded antibody covers all 40 residues, not the 36 distinct numbers."""
    from proto_tools.tools.inverse_folding import (
        InverseFoldingInput,
        InverseFoldingStructureInput,
        ProteinMPNNSampleConfig,
        run_proteinmpnn_sample,
    )

    native = {cid: antibody_structure.get_chain_sequence(cid) for cid in ("A", "B")}

    def fake_dispatch(toolkit, input_dict, *, instance=None, config=None):
        staged = Structure(structure=Path(input_dict["pdb_path"]).read_text(), structure_format="pdb", source="test")
        # The model answers with one residue per residue it was given, in chain order.
        return {
            "seq": ["/".join(staged.get_chain_sequence(cid) for cid in input_dict["chain_ids"])],
            "score": [-1.0],
            "seqid": [1.0],
        }

    monkeypatch.setattr(
        "proto_tools.tools.inverse_folding.proteinmpnn._sampling.ToolInstance.dispatch",
        staticmethod(fake_dispatch),
    )
    output = run_proteinmpnn_sample(
        InverseFoldingInput(inputs=[InverseFoldingStructureInput(structure=antibody_structure)]),
        ProteinMPNNSampleConfig(num_sequences_per_structure=1, batch_size=1, seed=0, device="cpu"),
    )

    design = output.design_sets[0].complexes[0]
    assert design.chain_sequence_map() == native
    assert len(design.chain_sequence_map()["B"]) == 40
