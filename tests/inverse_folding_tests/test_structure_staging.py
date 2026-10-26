"""tests/inverse_folding_tests/test_structure_staging.py.

Tests for staging a structure for an inverse-folding model and reading its output back.

Pure structure bookkeeping: CPU-only, environment-free, no model execution.
"""

from pathlib import Path

import pytest

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.structure_staging import (
    chain_residue_ids,
    restore_numbering,
    sequential_pdb,
    staged_ensemble,
    staged_structure,
)


def test_chain_residue_ids_keep_insertion_codes(insertion_coded_structure):
    """``5`` and ``5A`` are distinct residues, where integer numbering collapses them."""
    residue_ids = chain_residue_ids(insertion_coded_structure)["A"]
    assert residue_ids[4:7] == [(5, ""), (5, "A"), (6, "")]
    assert insertion_coded_structure.get_chain_positions("A")[4:6] == [5, 5]


def test_sequential_pdb_numbers_each_residue_by_its_position(insertion_coded_structure):
    """The staged copy numbers residues 1..N with no insertion codes, sequence unchanged."""
    staged = Structure(structure=sequential_pdb(insertion_coded_structure)[0], structure_format="pdb", source="test")
    assert chain_residue_ids(staged)["A"] == [(n, "") for n in range(1, 66)]
    assert staged.get_chain_sequence("A") == insertion_coded_structure.get_chain_sequence("A")


def test_restore_numbering_round_trips_insertion_codes(insertion_coded_structure):
    """An output written in staged numbering gets the input residue IDs back."""
    residue_ids = chain_residue_ids(insertion_coded_structure)
    staged = Structure(structure=sequential_pdb(insertion_coded_structure)[0], structure_format="pdb", source="test")
    restored = restore_numbering(staged.structure_cif, residue_ids, source="test")
    assert chain_residue_ids(restored) == residue_ids


def test_restore_numbering_reads_pdb_output(insertion_coded_structure):
    """Models that write PDB are restored the same way as those that write CIF."""
    residue_ids = chain_residue_ids(insertion_coded_structure)
    pdb_content, _ = sequential_pdb(insertion_coded_structure)
    restored = restore_numbering(pdb_content, residue_ids, source="test", structure_format="pdb")
    assert chain_residue_ids(restored) == residue_ids


def test_restore_numbering_maps_shortened_chain_ids_back(two_character_chain_structure):
    """A CIF chain ID too long for PDB is staged shortened and restored on the output."""
    pdb_content, chain_map = sequential_pdb(two_character_chain_structure)
    staged = Structure(structure=pdb_content, structure_format="pdb", source="test")
    assert staged.get_chain_ids() == [chain_map["AB"]]
    assert len(chain_map["AB"]) == 1

    residue_ids = chain_residue_ids(two_character_chain_structure)
    restored = restore_numbering(staged.structure_cif, residue_ids, source="test", chain_map=chain_map)
    assert restored.get_chain_ids() == ["AB"]
    assert chain_residue_ids(restored) == residue_ids


def test_restore_numbering_rejects_a_chain_of_different_length(staging_structure, insertion_coded_structure):
    """A residue-count mismatch cannot be mapped back by position, so it raises."""
    shortened = {"A": chain_residue_ids(insertion_coded_structure)["A"][:-1]}
    with pytest.raises(ValueError, match="numbering cannot be restored"):
        restore_numbering(staging_structure.structure_cif, shortened, source="test")


# ── StagedStructure ─────────────────────────────────────────────────────────


def test_staged_structure_exposes_the_file_and_the_maps_to_read_output_back(insertion_coded_structure):
    """Every residue reaches the model, each numbered by its position."""
    with staged_structure(insertion_coded_structure) as staged:
        assert staged.path.exists()
        written = Structure(structure=staged.path.read_text(), structure_format="pdb", source="test")
        assert chain_residue_ids(written)["A"] == [(n, "") for n in range(1, 66)]
        assert staged.residue_ids == chain_residue_ids(insertion_coded_structure)
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
    assert chain_residue_ids(restored) == chain_residue_ids(insertion_coded_structure)


def test_staged_ensemble_stages_every_conformer(insertion_coded_structure, staging_structure):
    """Conformers are staged alongside the primary, which carries the maps."""
    with staged_ensemble([insertion_coded_structure, staging_structure]) as (staged, conformer_paths):
        assert len(conformer_paths) == 1
        assert all(path.exists() for path in [staged.path, *conformer_paths])
        assert staged.residue_ids == chain_residue_ids(insertion_coded_structure)


# ── Every sample tool stages its input ──────────────────────────────────────


def _proteinmpnn_case(native):
    from proto_tools.tools.inverse_folding import ProteinMPNNSampleConfig, run_proteinmpnn_sample

    return (
        "proteinmpnn.proteinmpnn_sample",
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
    assert chain_residue_ids(staged)["A"] == [(n, "") for n in range(1, len(native) + 1)]
    assert captured["chain_ids"] == ["A"]
    assert design.chain_sequence_map() == {"A": native}


def test_restore_numbering_keeps_pdb_output_as_pdb(insertion_coded_structure):
    """A model that writes PDB gets a PDB back, so restoring numbering does not change format."""
    pdb_content, _ = sequential_pdb(insertion_coded_structure)
    restored = restore_numbering(
        pdb_content, chain_residue_ids(insertion_coded_structure), source="test", structure_format="pdb"
    )
    assert restored.structure_format == "pdb"


def test_restore_numbering_falls_back_to_cif_for_a_chain_id_pdb_cannot_hold(two_character_chain_structure):
    """``AB`` does not fit a PDB chain column, so the restored structure is CIF."""
    pdb_content, chain_map = sequential_pdb(two_character_chain_structure)
    restored = restore_numbering(
        pdb_content,
        chain_residue_ids(two_character_chain_structure),
        source="test",
        chain_map=chain_map,
        structure_format="pdb",
    )
    assert restored.structure_format == "cif"
    assert restored.get_chain_ids() == ["AB"]
