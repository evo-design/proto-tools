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
