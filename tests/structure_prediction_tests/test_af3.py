"""
test_af3.py

Focused tests for AlphaFold3 specialized functionality, such as input JSON construction
and modifications.
"""

import json
from unittest.mock import patch

import pytest

from bio_programming_tools.tools.structure_prediction import (
    AlphaFold3Config,
    AlphaFold3Input,
    Chain,
    StructurePredictionComplex,
    run_alphafold3,
)


@pytest.fixture
def mock_af3_inference(tmp_path):
    """Mock ToolInstance.dispatch to capture and verify input JSON format."""
    dummy_pdb_file = tmp_path / "dummy.pdb"
    dummy_pdb_file.write_text(
        "HEADER    DUMMY PDB\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.95           C\n"
        "END\n"
    )
    dummy_pdb_path = str(dummy_pdb_file)
    mock_metrics = {"avg_plddt": 0.95, "ptm": 0.8}

    captured_data = {}

    def mock_dispatch(tool_name, input_data, **kwargs):
        """Mock dispatch that reads input JSON and returns expected format."""
        # Read the input JSON file to capture it for test verification
        with open(input_data["input_json_path"], "r") as f:
            captured_data["input_json"] = json.load(f)

        # Return dict format (not tuple) as per worker protocol
        return {
            "structure_pdb": dummy_pdb_path,
            "metrics": mock_metrics,
        }

    # Patch ToolInstance.dispatch as a static/class method
    with patch("bio_programming_tools.tools.structure_prediction.alphafold3.alphafold3.ToolInstance") as mock_ti:
        mock_ti.dispatch = mock_dispatch
        captured_data["mock"] = mock_dispatch
        yield captured_data


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

    # Extract captured input JSON
    input_json = mock_af3_inference["input_json"]
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

    # Extract captured input JSON
    input_json = mock_af3_inference["input_json"]
    sequences = input_json["sequences"]

    assert len(sequences) == 2

    assert sequences[0]["ligand"]["id"] == "A"
    assert sequences[0]["ligand"]["ccdCodes"] == ["ALE"]

    assert sequences[1]["ligand"]["id"] == "B"
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]