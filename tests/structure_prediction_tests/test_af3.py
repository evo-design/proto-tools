"""
test_af3.py

Focused tests for AlphaFold3 specialized functionality, such as input JSON construction
and modifications.
"""

from unittest.mock import patch

import pytest

from bio_programming_tools.tools.structure_prediction import (
    AlphaFold3Config,
    AlphaFold3Input,
    Chain,
    StructurePredictionComplex,
    run_alphafold3,
)
from bio_programming_tools.tools.structure_prediction.alphafold3 import (
    alphafold3 as alphafold3_module,
)


@pytest.fixture
def mock_af3_inference(tmp_path):
    dummy_pdb_file = tmp_path / "dummy.pdb"
    dummy_pdb_file.write_text(
        "HEADER    DUMMY PDB\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.95           C\n"
        "END\n"
    )
    dummy_pdb_path = str(dummy_pdb_file)
    mock_metrics = {"avg_plddt": 0.95, "ptm": 0.8}

    # Patch where it's used (alphafold3.py), not where it's defined (inference.py)
    with patch.object(
        alphafold3_module, "alphafold3_inference", return_value=(dummy_pdb_path, mock_metrics)
    ) as mock:
        yield mock


def test_af3_ligand_and_nucleic_acids(mock_af3_inference):
    """
    Verifies that DNA, RNA, and Ligands are correctly formatted in the AF3 JSON dialect.
    """
    chains = [
        Chain(sequence="MVLSPADKTN", entity_type="protein"),
        Chain(sequence="ACGT", entity_type="dna"),
        Chain(sequence="CCO", entity_type="ligand"),
    ]

    complexes = [StructurePredictionComplex(chains=chains)]

    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_entities", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    # Extract arguments.
    input_json = mock_af3_inference.call_args.kwargs["input_json"]
    sequences = input_json["sequences"]

    assert len(sequences) == 3

    assert sequences[0]["protein"]["id"] == "A"
    assert sequences[0]["protein"]["sequence"] == "MVLSPADKTN"

    assert sequences[1]["dna"]["id"] == "B"
    assert sequences[1]["dna"]["sequence"] == "ACGT"

    assert sequences[2]["ligand"]["id"] == "C"
    assert sequences[2]["ligand"]["ccdCodes"] == ["EOH"]


def test_af3_ligand_smile_to_ccd_conversion(mock_af3_inference):
    """
    Test automatic conversion of SMILES to CCD codes.
    """
    chains = [
        Chain(sequence="CNC[C@@H](c1ccc(c(c1)O)O)O", entity_type="ligand"),  # L-epinephrine.
        Chain(sequence="c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N", entity_type="ligand"),  # ATP.
    ]

    complexes = [StructurePredictionComplex(chains=chains)]

    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_entities", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    # Extract arguments.
    input_json = mock_af3_inference.call_args.kwargs["input_json"]
    sequences = input_json["sequences"]

    assert len(sequences) == 2

    assert sequences[0]["ligand"]["id"] == "A"
    assert sequences[0]["ligand"]["ccdCodes"] == ["ALE"]

    assert sequences[1]["ligand"]["id"] == "B"
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]