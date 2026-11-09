"""Predicted structures take their chain types from the input complex, not from the predictor's file.

Every structure predictor routes its output through ``align_with_input``, which types
each chain from the input chain it came from. The fixtures are real outputs for the MfnG + L-
tyrosine benchmark complex, kept as written: AlphaFold3's mmCIF (entities declared correctly), a
Biopython-converted AlphaFold3 PDB (no entity records, ``TER`` after the ligand), ESMFold2's
mmCIF (the ligand declared a
``polypeptide(L)`` entity), and RF3's mmCIF (the ligand written as ATOM residues, no entity
records).
"""

import inspect
import logging
import sys
from pathlib import Path

import gemmi
import pytest

from proto_tools.entities.structures import Structure
from proto_tools.tools import ToolRegistry
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionOutput,
    align_with_input,
)
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.structure_prediction_tests.test_structure_prediction import _STRUCTURE_PREDICTORS

_OUTPUTS = Path(__file__).resolve().parents[1] / "dummy_data" / "predicted_ligand_chains"
_PREDICTED = ["alphafold3_mfng_tyr.cif", "alphafold3_mfng_tyr.pdb", "esmfold2_mfng_tyr.cif", "rf3_mfng_tyr.cif"]


@pytest.fixture(scope="module")
def mfng_and_tyrosine():
    """The input complex the fixtures were predicted from: MfnG (protein) + L-tyrosine (ligand)."""
    return load_benchmark_complex("MfnG_and_ligand")


def _file_chain_polymers(cif: str) -> dict[str, bool]:
    """Whether each chain is a polymer by the mmCIF's own records, read with plain gemmi (no proto-tools fixes)."""
    struct = gemmi.make_structure_from_block(gemmi.cif.read_string(cif)[0])
    struct.setup_entities()
    return {chain.name: bool(chain.get_polymer()) for chain in struct[0]}


@pytest.mark.parametrize("filename", _PREDICTED)
def test_predicted_chains_are_typed_from_the_input(filename: str, mfng_and_tyrosine) -> None:
    """Each predictor's real output comes back with a protein chain A and a ligand chain B."""
    structure = align_with_input(Structure.from_file(_OUTPUTS / filename), mfng_and_tyrosine.chains)

    assert structure.get_chain_types() == {"A": "protein", "B": "ligand"}
    assert structure.num_chains == 1
    assert structure.get_chain_ligands("B") == [("TYR", 1)]
    # The written mmCIF itself declares B a non-polymer, for readers other than proto-tools.
    assert _file_chain_polymers(structure.structure_cif) == {"A": True, "B": False}


def test_a_correctly_typed_file_is_returned_unchanged(mfng_and_tyrosine) -> None:
    """A prediction whose own file already types every chain right keeps its file and format."""
    protein_only = Structure.from_file(_OUTPUTS / "esmfold2_mfng_tyr.cif").select_chain("A")
    pdb = Structure(structure=protein_only.structure_pdb, structure_format="pdb")

    assert align_with_input(pdb, mfng_and_tyrosine.chains[:1]) is pdb


def test_chains_renamed_in_order_are_matched_by_position(mfng_and_tyrosine) -> None:
    """A predictor that renames the chains but keeps their order still gets them typed and renamed back."""
    renamed = Structure.from_file(_OUTPUTS / "rf3_mfng_tyr.cif").with_renamed_chains({"A": "P", "B": "Q"})
    structure = align_with_input(renamed, mfng_and_tyrosine.chains)

    assert structure.get_chain_types() == {"A": "protein", "B": "ligand"}


def test_mismatched_output_is_kept_as_written_with_a_warning(mfng_and_tyrosine, caplog) -> None:
    """A completed prediction is never discarded: when chains cannot be matched safely, it comes back as written."""
    structure = Structure.from_file(_OUTPUTS / "rf3_mfng_tyr.cif")
    renamed = structure.with_renamed_chains({"A": "P", "B": "Q"})
    protein, tyrosine = mfng_and_tyrosine.chains
    cases = {
        "returned 2 chain(s) for 1 input chain(s)": (structure, [protein]),
        # Renamed and reordered: the 383-residue protein would be paired with the ligand.
        "has 1 residue(s) but input protein chain 'B' has": (renamed, [tyrosine, protein]),
        "was given as rna but reads as protein": (
            structure,
            [protein.model_copy(update={"entity_type": "rna"}), tyrosine],
        ),
    }
    for message, (output, chains) in cases.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert align_with_input(output, chains) is output
        assert message in caplog.text, message


def test_with_chain_types_rejects_bad_requests() -> None:
    """``with_chain_types`` itself is strict; ``align_with_input`` turns its errors into warnings."""
    structure = Structure.from_file(_OUTPUTS / "esmfold2_mfng_tyr.cif")
    with pytest.raises(ValueError, match="was given as rna but reads as protein"):
        structure.with_chain_types({"A": "rna", "B": "ligand"})
    with pytest.raises(ValueError, match="not found"):
        structure.with_chain_types({"Z": "ligand"})
    with pytest.raises(ValueError, match="must be one of"):
        structure.with_chain_types({"B": "small molecule"})


# ── Every predictor is covered ──────────────────────────────────────────────


def _structure_predictor_specs():
    return [spec for spec in ToolRegistry.list_all() if issubclass(spec.output_model, StructurePredictionOutput)]


# Structure predictors whose inputs ``test_folding``'s generic complexes cannot satisfy, and the test that folds them.
_FOLDED_ELSEWHERE = {
    "boltz2-affinity": "needs one ligand binder per complex; folded by test_boltz2.py::test_boltz2_affinity_end_to_end",
}


def test_every_structure_predictor_is_folded_in_test_folding() -> None:
    """A registered structure predictor must be in ``test_folding``, whose GPU runs check its chain types."""
    tested = {run.__name__ for run, *_ in _STRUCTURE_PREDICTORS.values()}
    missing = [
        spec.key
        for spec in _structure_predictor_specs()
        if spec.function.__name__ not in tested and spec.key not in _FOLDED_ELSEWHERE
    ]
    assert missing == [], f"add {missing} to test_folding's _STRUCTURE_PREDICTORS"


def test_every_structure_predictor_types_its_chains_from_the_input() -> None:
    """Every structure predictor's module routes its output through ``align_with_input``."""
    skipping = [
        spec.key
        for spec in _structure_predictor_specs()
        if "align_with_input(" not in inspect.getsource(sys.modules[spec.function.__module__])
    ]
    assert skipping == [], f"{skipping} must pass their predicted structures through align_with_input"
