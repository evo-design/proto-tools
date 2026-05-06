"""Tests for ScoringStructureInput shared by all PyRosetta scoring tools."""

import pytest

from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_energy import (
    PyRosettaEnergyConfig,
    PyRosettaEnergyInput,
    run_pyrosetta_energy,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_relax import (
    PyRosettaRelaxConfig,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sap import (
    PyRosettaSAPConfig,
    PyRosettaSAPInput,
    run_pyrosetta_sap,
)
from proto_tools.tools.structure_scoring.pyrosetta.pyrosetta_sasa import (
    PyRosettaSASAConfig,
    PyRosettaSASAInput,
    run_pyrosetta_sasa,
)
from proto_tools.tools.structure_scoring.pyrosetta.shared_data_models import (
    MAX_CHAINS_FOR_PDB,
    ScoringStructureInput,
    prepare_pdb_and_chain_maps,
    remap_per_residue_chain_ids,
)
from tests._structure_fixtures import synthetic_cif


def test_scoring_input_accepts_mmcif_multichar_chain_ids():
    """Users can pass mmCIF chains with multi-character labels (e.g. 'Heavy')."""
    cif = synthetic_cif(["Heavy", "Light"])

    inp = ScoringStructureInput(structure=cif, chains_to_score=["Heavy"])

    # Selected chains are preserved as the original mmCIF labels (not shortened).
    assert inp.chains_to_score is not None
    assert inp.chains_to_score.chains == ["Heavy"]
    assert inp.structure.get_chain_ids() == ["Heavy", "Light"]


def test_helpers_round_trip_mmcif_chain_ids_without_mutating_input():
    """Forward-translate to PDB labels, remap back to mmCIF labels, verify input is untouched.

    Guards the end-to-end contract: the chain label a user supplies is what they see in
    the output, and nothing on the ScoringStructureInput or its Structure is mutated
    along the way.
    """
    cif = synthetic_cif(["Heavy", "Light"])
    inp = ScoringStructureInput(structure=cif, chains_to_score=["Heavy"])

    # ── Forward: prepare translates mmCIF labels → single-char PDB labels ──
    pdb_contents, pdb_chain_ids_list, pdb_to_mmcif_maps = prepare_pdb_and_chain_maps([inp])

    assert len(pdb_contents) == 1
    assert len(pdb_chain_ids_list) == 1
    assert len(pdb_to_mmcif_maps) == 1

    # The translated chain IDs are single characters — what PyRosetta sees.
    translated = pdb_chain_ids_list[0]
    assert translated is not None
    assert len(translated) == 1
    assert len(translated[0]) == 1
    pdb_label_for_heavy = translated[0]

    # The reverse map restores the original mmCIF label.
    assert pdb_to_mmcif_maps[0][pdb_label_for_heavy] == "Heavy"

    # The input ScoringStructureInput was NOT mutated by prepare.
    assert inp.chains_to_score is not None
    assert inp.chains_to_score.chains == ["Heavy"]
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
    assert inp.chains_to_score is not None
    assert inp.chains_to_score.chains == ["Heavy"]
    assert inp.structure.get_chain_ids() == ["Heavy", "Light"]


def test_scoring_input_rejects_too_many_chains():
    """Structures with more chains than PDB can represent are rejected up front."""
    chain_names = [f"chain{i}" for i in range(MAX_CHAINS_FOR_PDB + 1)]
    cif = synthetic_cif(chain_names)

    with pytest.raises(ValueError, match=f"at most {MAX_CHAINS_FOR_PDB}"):
        ScoringStructureInput(structure=cif)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("input_cls", "config_cls", "runner"),
    [
        pytest.param(PyRosettaEnergyInput, PyRosettaEnergyConfig, run_pyrosetta_energy, id="energy"),
        pytest.param(PyRosettaSAPInput, PyRosettaSAPConfig, run_pyrosetta_sap, id="sap"),
        pytest.param(PyRosettaSASAInput, PyRosettaSASAConfig, run_pyrosetta_sasa, id="sasa"),
    ],
)
def test_preprocess_preserves_multichar_chain_ids(input_cls, config_cls, runner):
    """pre_relax_structures preprocess preserves multi-char chain IDs across all scoring tools.

    Every scoring tool routes pre_relax_structures=True through the same shared
    helpers (relax_inputs_via_pyrosetta + remap_per_residue_chain_ids), so this
    parametrization covers all three tools in one definition. With
    ``chains_to_score=["Heavy"]`` selected from a ``["Heavy", "Light"]`` CIF, the relaxed
    Structure retains its original mmCIF labels and per-residue output reports
    the user's label, not PyRosetta's single-char substitute.
    """
    cif = synthetic_cif(["Heavy", "Light"])
    result = runner(
        input_cls(inputs=[{"structure": cif, "chains_to_score": ["Heavy"]}]),
        config_cls(
            pre_relax_structures=True,
            relax_config=PyRosettaRelaxConfig(relax_cycles=1, seed=42),
        ),
    )
    assert result.success, f"multichar+preprocess failed: {result.errors}"
    assert all(r.chain_id == "Heavy" for r in result.results[0].per_residue)
