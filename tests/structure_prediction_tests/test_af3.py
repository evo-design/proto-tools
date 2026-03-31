"""tests/structure_prediction_tests/test_af3.py

Tests for AlphaFold3."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from proto_tools.tools.structure_prediction import (
    AlphaFold3Config,
    AlphaFold3Input,
    Chain,
    StructurePredictionComplex,
    run_alphafold3,
)

# ── Module-level constants ────────────────────────────────────────────────────

_EPINEPHRINE_SMILES = "CNC[C@@H](c1ccc(c(c1)O)O)O"  # L-epinephrine → ALE
_ATP_SMILES = (
    "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)"
    "O[P@](=O)(O)OP(=O)(O)O)O)O)N"
)  # ATP → ATP


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_af3_inference(tmp_path):
    """Patch ToolInstance.dispatch to capture the input JSON written to disk.

    The mock runs synchronously inside run_alphafold3's tempfile.TemporaryDirectory
    context, so input_json_path is still live when mock_dispatch reads it.
    """
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
        with open(input_data["input_json_path"], "r") as f:
            captured_data["input_json"] = json.load(f)
        return {
            "structure_pdb": dummy_pdb_path,
            "metrics": mock_metrics,
        }

    with patch(
        "proto_tools.tools.structure_prediction.alphafold3.alphafold3.ToolInstance"
    ) as mock_ti:
        mock_ti.dispatch = mock_dispatch
        yield captured_data


# ── JSON structure tests ──────────────────────────────────────────────────────


def test_af3_ligand_and_nucleic_acids(mock_af3_inference):
    """DNA, RNA, and ligands are correctly formatted in the AF3 JSON dialect."""
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

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 3

    assert sequences[0]["protein"]["id"] == "A"
    assert sequences[0]["protein"]["sequence"] == "MVLSPADKTN"

    assert sequences[1]["dna"]["id"] == "B"
    assert sequences[1]["dna"]["sequence"] == "ACGT"

    assert sequences[2]["ligand"]["id"] == "C"
    assert sequences[2]["ligand"]["ccdCodes"] == ["EOH"]


def test_af3_rna_entity(mock_af3_inference):
    """RNA chains are correctly formatted in the AF3 JSON dialect."""
    chains = [
        Chain(sequence="AUGC", entity_type="rna"),
    ]
    complexes = [StructurePredictionComplex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_rna", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 1
    assert sequences[0]["rna"]["id"] == "A"
    assert sequences[0]["rna"]["sequence"] == "AUGC"


def test_af3_protein_modifications_in_json(mock_af3_inference):
    """Protein PTMs are serialised as ptmType/ptmPosition in the AF3 JSON."""
    chains = [
        Chain(
            sequence="MVLSPADKTN",
            entity_type="protein",
            modifications=[(4, "SEP")],  # position 4 is 'S' (serine)
        ),
    ]
    complexes = [StructurePredictionComplex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_ptm", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    protein_entry = mock_af3_inference["input_json"]["sequences"][0]["protein"]
    assert "modifications" in protein_entry
    assert len(protein_entry["modifications"]) == 1
    mod = protein_entry["modifications"][0]
    assert mod["ptmType"] == "SEP"
    assert mod["ptmPosition"] == 4


def test_af3_nucleic_acid_modifications_in_json(mock_af3_inference):
    """RNA modifications are serialised as modificationType/basePosition in the AF3 JSON."""
    chains = [
        Chain(
            sequence="AUGCAUGC",
            entity_type="rna",
            modifications=[(3, "2MG")],  # position 3 is 'G' (guanosine)
        ),
    ]
    complexes = [StructurePredictionComplex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_rna_mod", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    rna_entry = mock_af3_inference["input_json"]["sequences"][0]["rna"]
    assert "modifications" in rna_entry
    assert len(rna_entry["modifications"]) == 1
    mod = rna_entry["modifications"][0]
    assert mod["modificationType"] == "2MG"
    assert mod["basePosition"] == 3


# ── SMILES-to-CCD conversion tests ───────────────────────────────────────────


def test_af3_ligand_smile_to_ccd_conversion(mock_af3_inference):
    """Known SMILES strings are automatically mapped to their CCD codes."""
    chains = [
        Chain(sequence=_EPINEPHRINE_SMILES, entity_type="ligand"),
        Chain(sequence=_ATP_SMILES, entity_type="ligand"),
    ]
    complexes = [StructurePredictionComplex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_entities", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 2

    assert sequences[0]["ligand"]["id"] == "A"
    assert sequences[0]["ligand"]["ccdCodes"] == ["ALE"]

    assert sequences[1]["ligand"]["id"] == "B"
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]


