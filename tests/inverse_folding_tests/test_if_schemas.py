"""
test_if_inputs.py

Tests for inverse folding input validation.
"""

from pathlib import Path
from typing import List

import pytest
from pydantic import Field

from bio_programming_tools.entities.structures.structure import Structure
from bio_programming_tools.tools.inverse_folding.shared_data_models import (
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


class TestInverseFoldingStructure:

    def test_structure_from_pdb_filepath(self):
        structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE)
        assert isinstance(structure.structure, Structure)

    def test_structure_from_pdb_content(self, pdb_file_content: str):
        structure = InverseFoldingStructureInput(structure=pdb_file_content)
        assert isinstance(structure.structure, Structure)

    def test_structure_with_chain_ids(self):
        structure = InverseFoldingStructureInput(
            structure=TEST_PDB_FILE, chain_ids=["A"]
        )
        assert structure.chain_ids == ["A"]

    def test_structure_without_chain_ids_defaults_to_all(self):
        """When chain_ids is None, should default to all chains in structure."""
        structure = InverseFoldingStructureInput(structure=TEST_PDB_FILE)
        # chain_ids should be populated with all available chains
        assert structure.chain_ids is not None
        assert len(structure.chain_ids) > 0
        # Should match the chains in the structure
        expected_chains = structure.structure.get_chain_ids()
        assert structure.chain_ids == expected_chains

    def test_structure_with_fixed_positions(self):
        structure = InverseFoldingStructureInput(
            structure=TEST_PDB_FILE, fixed_positions={"A": [1, 2, 3]}
        )
        assert structure.fixed_positions == {"A": [1, 2, 3]}

    def test_bad_structure_raise_error(self):
        with pytest.raises(ValueError):
            _ = InverseFoldingStructureInput(structure="not a pdb file")

        with pytest.raises(FileNotFoundError):
            _ = InverseFoldingStructureInput(structure="/not/a/real/file.pdb")


class TestInverseFoldingInput:

    def test_input_from_pdb_filepath(self):
        input = InverseFoldingInput(
            inputs=[
                InverseFoldingStructureInput(structure=TEST_PDB_FILE),
                InverseFoldingStructureInput(structure=TEST_PDB_FILE),
                InverseFoldingStructureInput(structure=TEST_PDB_FILE),
            ]
        )
        assert len(input.inputs) == 3
        assert all(isinstance(inp.structure, Structure) for inp in input.inputs)

    def test_input_from_pdb_content(self, pdb_file_content: str):
        input = InverseFoldingInput(
            inputs=[
                InverseFoldingStructureInput(structure=pdb_file_content),
                InverseFoldingStructureInput(structure=pdb_file_content),
                InverseFoldingStructureInput(structure=pdb_file_content),
            ]
        )
        assert len(input.inputs) == 3
        assert all(isinstance(inp.structure, Structure) for inp in input.inputs)


class MockCustomDesignedSequences(DesignedSequences):
    custom_metric: List[float] = Field(
        description="Custom metric for the designed sequences"
    )


class TestDesignedSequences:

    def test_len(self):
        sequences = MockCustomDesignedSequences(
            sequences=["MVLSP", "GGGS"], custom_metric=[0.1, 0.2]
        )
        assert len(sequences) == 2

    def test_getitem(self):
        sequences = MockCustomDesignedSequences(
            sequences=["MVLSP", "GGGS"], custom_metric=[0.1, 0.2]
        )
        assert sequences[0] == "MVLSP"
        assert sequences[1] == "GGGS"

        # Test get_sequence_metrics
        metrics = sequences.get_sequence_metrics(0)
        assert metrics == {"custom_metric": 0.1}
        metrics = sequences.get_sequence_metrics(1)
        assert metrics == {"custom_metric": 0.2}


class TestInverseFoldingOutput:

    def test_len(self):
        output = InverseFoldingOutput(
            designed_sequences=[DesignedSequences(sequences=["MVLSP", "GGGS"])]
        )
        assert len(output) == 1

    def test_getitem(self):
        output = InverseFoldingOutput(
            designed_sequences=[DesignedSequences(sequences=["MVLSP", "GGGS"])]
        )
        assert output[0] == DesignedSequences(sequences=["MVLSP", "GGGS"])

    def test_iter(self):
        output = InverseFoldingOutput(
            designed_sequences=[DesignedSequences(sequences=["MVLSP", "GGGS"])]
        )
        assert list(output) == [DesignedSequences(sequences=["MVLSP", "GGGS"])]
