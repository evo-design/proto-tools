"""tests/inverse_folding_tests/test_if_schemas.py

Tests for inverse folding shared data models."""

from pathlib import Path
from typing import List

import pytest
from pydantic import Field

from proto_tools.entities.structures.structure import Structure
from proto_tools.tools.inverse_folding.shared_data_models import (
    DesignedSequences,
    InverseFoldingInput,
    InverseFoldingOutput,
    InverseFoldingStructureInput,
)

TEST_PDB_FILE = Path(__file__).parent.parent / "dummy_data" / "renin_af3.pdb"


@pytest.fixture(scope="module")
def pdb_file_content() -> str:
    with open(TEST_PDB_FILE, "r") as f:
        return f.read()


# ── Structure input validation ───────────────────────────────────────────


def test_structure_from_pdb_filepath():
    structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE)
    assert isinstance(structure.structure, Structure)


def test_structure_from_pdb_content(pdb_file_content: str):
    structure = InverseFoldingStructureInput(structure=pdb_file_content)
    assert isinstance(structure.structure, Structure)


def test_structure_with_chain_ids():
    structure = InverseFoldingStructureInput(
        structure=TEST_PDB_FILE, chain_ids=["A"]
    )
    assert structure.chain_ids == ["A"]


def test_structure_without_chain_ids_defaults_to_all():
    """When chain_ids is None, should default to all chains in structure."""
    structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE)
    assert structure.chain_ids is not None
    assert len(structure.chain_ids) > 0
    expected_chains = structure.structure.get_chain_ids()
    assert structure.chain_ids == expected_chains


def test_structure_with_fixed_positions():
    structure = InverseFoldingStructureInput(
        structure=TEST_PDB_FILE, fixed_positions={"A": [1, 2, 3]}
    )
    assert structure.fixed_positions == {"A": [1, 2, 3]}


def test_structure_rejects_invalid_pdb_content():
    with pytest.raises(ValueError):
        InverseFoldingStructureInput(structure="not a pdb file")


def test_structure_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        InverseFoldingStructureInput(structure="/not/a/real/file.pdb")


# ── InverseFoldingInput ──────────────────────────────────────────────────


def test_input_from_pdb_filepath():
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=TEST_PDB_FILE),
            InverseFoldingStructureInput(structure=TEST_PDB_FILE),
            InverseFoldingStructureInput(structure=TEST_PDB_FILE),
        ]
    )
    assert len(inp.inputs) == 3
    assert all(isinstance(i.structure, Structure) for i in inp.inputs)


def test_input_from_pdb_content(pdb_file_content: str):
    inp = InverseFoldingInput(
        inputs=[
            InverseFoldingStructureInput(structure=pdb_file_content),
            InverseFoldingStructureInput(structure=pdb_file_content),
            InverseFoldingStructureInput(structure=pdb_file_content),
        ]
    )
    assert len(inp.inputs) == 3
    assert all(isinstance(i.structure, Structure) for i in inp.inputs)


# ── DesignedSequences ────────────────────────────────────────────────────


class _MockDesignedSequences(DesignedSequences):
    custom_metric: List[float] = Field(
        description="Custom metric for the designed sequences"
    )


def test_designed_sequences_len():
    sequences = _MockDesignedSequences(
        sequences=["MVLSP", "GGGS"], custom_metric=[0.1, 0.2]
    )
    assert len(sequences) == 2


def test_designed_sequences_getitem_and_metrics():
    sequences = _MockDesignedSequences(
        sequences=["MVLSP", "GGGS"], custom_metric=[0.1, 0.2]
    )
    assert sequences[0] == "MVLSP"
    assert sequences[1] == "GGGS"
    assert sequences.get_sequence_metrics(0) == {"custom_metric": 0.1}
    assert sequences.get_sequence_metrics(1) == {"custom_metric": 0.2}


# ── InverseFoldingOutput ─────────────────────────────────────────────────


def test_output_len():
    output = InverseFoldingOutput(
        designed_sequences=[DesignedSequences(sequences=["MVLSP", "GGGS"])]
    )
    assert len(output) == 1


def test_output_getitem():
    output = InverseFoldingOutput(
        designed_sequences=[DesignedSequences(sequences=["MVLSP", "GGGS"])]
    )
    assert output[0] == DesignedSequences(sequences=["MVLSP", "GGGS"])


def test_output_iter():
    output = InverseFoldingOutput(
        designed_sequences=[DesignedSequences(sequences=["MVLSP", "GGGS"])]
    )
    assert list(output) == [DesignedSequences(sequences=["MVLSP", "GGGS"])]
