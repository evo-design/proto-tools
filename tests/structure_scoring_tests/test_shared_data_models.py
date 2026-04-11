"""Tests for ScoringStructureInput shared by all PyRosetta scoring tools."""

import pytest

from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    MAX_CHAINS_FOR_PDB,
    ScoringStructureInput,
    prepare_pdb_and_chain_maps,
    remap_per_residue_chain_ids,
)


def _synthetic_cif(chain_names: list[str]) -> str:
    """Build a minimal valid mmCIF with one glycine per named chain.

    Chains are placed 100 A apart on the x-axis to avoid any overlap.
    """
    header = (
        "data_synthetic\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
    )
    rows = []
    atom_id = 1
    for chain_idx, name in enumerate(chain_names):
        base_x = chain_idx * 100.0
        label_asym = chr(ord("A") + chain_idx % 26)
        for atom_name, dx, dy in [("N", 0.0, 0.0), ("CA", 1.5, 0.0), ("C", 2.0, 1.5), ("O", 1.3, 2.5)]:
            rows.append(
                f"ATOM {atom_id} {atom_name[0]} {atom_name} . GLY {label_asym} 1 1 ? "
                f"{base_x + dx:.3f} {dy:.3f} 0.000 1.00 20.00 1 GLY {name} {atom_name} 1"
            )
            atom_id += 1
    return header + "\n".join(rows) + "\n"


def test_scoring_input_accepts_mmcif_multichar_chain_ids():
    """Users can pass mmCIF chains with multi-character labels (e.g. 'Heavy')."""
    cif = _synthetic_cif(["Heavy", "Light"])

    inp = ScoringStructureInput(structure=cif, chain_ids=["Heavy"])

    # chain_ids are preserved as the original mmCIF labels (not shortened).
    assert inp.chain_ids == ["Heavy"]
    assert inp.structure.get_chain_ids() == ["Heavy", "Light"]


def test_helpers_round_trip_mmcif_chain_ids_without_mutating_input():
    """Forward-translate to PDB labels, remap back to mmCIF labels, verify input is untouched.

    Guards the end-to-end contract: the chain label a user supplies is what they see in
    the output, and nothing on the ScoringStructureInput or its Structure is mutated
    along the way.
    """
    cif = _synthetic_cif(["Heavy", "Light"])
    inp = ScoringStructureInput(structure=cif, chain_ids=["Heavy"])

    # ── Forward: prepare translates mmCIF labels → single-char PDB labels ──
    pdb_contents, pdb_chain_ids_list, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps([inp])

    assert len(pdb_contents) == 1
    assert len(pdb_chain_ids_list) == 1
    assert len(pdb_to_mmcif_maps) == 1

    # The translated chain_ids are single characters — what PyRosetta sees.
    translated = pdb_chain_ids_list[0]
    assert translated is not None
    assert len(translated) == 1
    assert len(translated[0]) == 1
    pdb_label_for_heavy = translated[0]

    # The reverse map restores the original mmCIF label.
    assert pdb_to_mmcif_maps[0][pdb_label_for_heavy] == "Heavy"

    # The input ScoringStructureInput was NOT mutated by prepare.
    assert inp.chain_ids == ["Heavy"]
    assert inp.structure.get_chain_ids() == ["Heavy", "Light"]

    # ── Reverse: remap rewrites PDB labels → mmCIF labels in-place ──
    fake_results = [
        {
            "per_residue": [
                {"chain_id": pdb_label_for_heavy, "residue_index": 1},
                {"chain_id": pdb_label_for_heavy, "residue_index": 2},
            ],
        }
    ]
    remap_per_residue_chain_ids(fake_results, pdb_to_mmcif_maps)

    assert all(res["chain_id"] == "Heavy" for res in fake_results[0]["per_residue"])

    # The input ScoringStructureInput is STILL not mutated after remap.
    assert inp.chain_ids == ["Heavy"]
    assert inp.structure.get_chain_ids() == ["Heavy", "Light"]


def test_scoring_input_rejects_too_many_chains():
    """Structures with more chains than PDB can represent are rejected up front."""
    chain_names = [f"chain{i}" for i in range(MAX_CHAINS_FOR_PDB + 1)]
    cif = _synthetic_cif(chain_names)

    with pytest.raises(ValueError, match=f"at most {MAX_CHAINS_FOR_PDB}"):
        ScoringStructureInput(structure=cif)
