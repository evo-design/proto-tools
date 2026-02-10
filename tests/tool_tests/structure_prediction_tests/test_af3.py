"""
test_af3.py

Focused tests for AlphaFold3 specialized functionality, such as input JSON construction
and modifications.
"""

from unittest.mock import patch

import pytest

from bio_programming.bio_tools.tools.structure_prediction import (
    AlphaFold3Config,
    AlphaFold3Input,
    Chain,
    StructurePredictionComplex,
    run_alphafold3,
)
from bio_programming.bio_tools.tools.structure_prediction.alphafold3 import (
    alphafold3 as alphafold3_module,
)


@pytest.fixture
def mock_af3_inference(tmp_path):
    dummy_pdb_file = tmp_path / "dummy.pdb"
    dummy_pdb_file.write_text("HEADER    DUMMY PDB")
    dummy_pdb_path = str(dummy_pdb_file)
    mock_metrics = {"avg_plddt": 0.95, "ptm": 0.8}

    # Mock use_cloud_gpu to return False so tests run as if on local GPU
    with patch.object(alphafold3_module, "use_cloud_gpu", return_value=False):
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

    run_alphafold3(inputs, config)

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

    run_alphafold3(inputs, config)

    # Extract arguments.
    input_json = mock_af3_inference.call_args.kwargs["input_json"]
    sequences = input_json["sequences"]

    assert len(sequences) == 2

    assert sequences[0]["ligand"]["id"] == "A"
    assert sequences[0]["ligand"]["ccdCodes"] == ["ALE"]

    assert sequences[1]["ligand"]["id"] == "B"
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]


@pytest.mark.skip(reason="Mock does not work with full test suite")
def test_af3_modifications_application(mock_af3_inference):
    """
    Verifies that Chain modifications are correctly converted to AF3 JSON format.
    """
    chains = [
        Chain(
            sequence="MVLSPADKTN",
            entity_type="protein",
            modifications=[(1, "HY3"), (5, "P1L")]
        ),
        Chain(
            sequence="ACGT",
            entity_type="dna",
            modifications=[(1, "6OG"), (2, "6MA")]
        ),
        Chain(
            sequence="ACGU",
            entity_type="rna",
            modifications=[(1, "2MG"), (4, "5MC")]
        ),
    ]

    complexes = [StructurePredictionComplex(chains=chains)]

    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(
        name="test_mods",
        use_msa=False,
    )

    run_alphafold3(inputs, config)

    # Inspect generated JSON.
    input_json = mock_af3_inference.call_args.kwargs["input_json"]
    sequences = input_json["sequences"]

    # Verify protein modifications use ptmType and ptmPosition
    prot_entry = sequences[0]["protein"]
    assert "modifications" in prot_entry
    assert prot_entry["modifications"] == [
        {"ptmType": "HY3", "ptmPosition": 1},
        {"ptmType": "P1L", "ptmPosition": 5},
    ]

    # Verify DNA modifications use modificationType and basePosition
    dna_entry = sequences[1]["dna"]
    assert "modifications" in dna_entry
    assert dna_entry["modifications"] == [
        {"modificationType": "6OG", "basePosition": 1},
        {"modificationType": "6MA", "basePosition": 2},
    ]

    # Verify RNA modifications use modificationType and basePosition
    rna_entry = sequences[2]["rna"]
    assert "modifications" in rna_entry
    assert rna_entry["modifications"] == [
        {"modificationType": "2MG", "basePosition": 1},
        {"modificationType": "5MC", "basePosition": 4},
    ]


@pytest.mark.skip(reason="Mock does not work with full test suite")
def test_af3_ligand_modifications_ignored(mock_af3_inference):
    """
    Verifies that modifications on ligand chains are correctly ignored in AF3 JSON.
    """
    chains = [
        Chain(
            sequence="CCO",
            entity_type="ligand",
            modifications=[(1, "HY3")]
        )
    ]
    complexes = [StructurePredictionComplex(chains=chains)]

    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(
        name="test_ligand_mods",
        use_msa=False,
    )

    run_alphafold3(inputs, config)

    # Check JSON has no modifications field for ligand
    input_json = mock_af3_inference.call_args.kwargs["input_json"]
    assert "modifications" not in input_json["sequences"][0]["ligand"]
