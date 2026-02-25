"""Tests for the BioEmu conformational ensemble sampling tool."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from bio_programming_tools.entities.structures import Structure
from bio_programming_tools.tools.structure_dynamics.bioemu import (
    BioEmuConfig,
    BioEmuInput,
    run_bioemu,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
)
from tests.tool_infra_tests.test_export_functionality import validate_output


@pytest.fixture
def sample_sequence() -> str:
    """A short valid protein sequence."""
    return "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"


@pytest.fixture
def sample_pdb_content() -> str:
    """Minimal valid PDB content for one residue."""
    return (
        "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C\n"
        "END\n"
    )


class TestBioEmuInput:
    """Tests for BioEmuInput validation."""

    def test_rejects_multi_chain_complex(self, sample_sequence: str):
        """Test that multi-chain complexes are rejected."""
        with pytest.raises(ValueError, match="single-chain"):
            BioEmuInput(
                complexes=[
                    StructurePredictionComplex(
                        chains=[
                            {"sequence": sample_sequence, "entity_type": "protein"},
                            {"sequence": sample_sequence, "entity_type": "protein"},
                        ]
                    )
                ]
            )

    def test_rejects_non_protein_entity(self):
        """Test that non-protein entities are rejected."""
        with pytest.raises(ValueError, match="only supports: protein"):
            BioEmuInput(
                complexes=[
                    StructurePredictionComplex(
                        chains=[{"sequence": "ACGT", "entity_type": "dna"}]
                    )
                ]
            )

    def test_rejects_invalid_amino_acids(self):
        """Test that invalid amino acid characters are rejected."""
        with patch(
            "bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample.return_invalid_protein_chars",
            return_value={"1", "2", "3"},
        ):
            with pytest.raises(ValueError, match="Invalid protein characters"):
                BioEmuInput(
                    complexes=[
                        StructurePredictionComplex(
                            chains=[
                                {
                                    "sequence": "MVLSPADKTNVKAAW123",
                                    "entity_type": "protein",
                                }
                            ]
                        )
                    ]
                )

    def test_warns_on_long_sequence(self, caplog):
        """Test that long sequences log a warning."""
        long_sequence = "A" * 600
        with patch(
            "bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample.return_invalid_protein_chars",
            return_value=set(),
        ):
            with caplog.at_level("WARNING"):
                BioEmuInput(
                    complexes=[
                        StructurePredictionComplex(
                            chains=[
                                {
                                    "sequence": long_sequence,
                                    "entity_type": "protein",
                                }
                            ]
                        )
                    ]
                )
        assert "500 residues" in caplog.text


class TestBioEmuConfig:
    """Tests for BioEmuConfig validation."""

    def test_invalid_values(self):
        """Test config validation for invalid values."""
        with pytest.raises(ValueError):
            BioEmuConfig(num_samples=0)
        with pytest.raises(ValueError):
            BioEmuConfig(batch_size=0)
        with pytest.raises(ValueError):
            BioEmuConfig(model_name="invalid-model")


class TestRunBioEmu:
    """Tests for run_bioemu."""

    @pytest.mark.include_in_env_report
    @pytest.mark.uses_gpu
    def test_bioemu_sample_tool(self):
        """Test BioEmu conformational ensemble sampling end-to-end."""
        sequence = "MKTAYIAKQRQISFVKSHFSRQLE"
        inputs = BioEmuInput(
            complexes=[
                StructurePredictionComplex(
                    chains=[{"sequence": sequence, "entity_type": "protein"}]
                )
            ]
        )
        config = BioEmuConfig(num_samples=5, verbose=False)

        result = run_bioemu(inputs, config)
        validate_output(result)

        assert result.tool_id == "bioemu-sample"
        assert len(result.ensembles) == 1
        assert len(result.ensembles[0].structures) >= 1
        assert result.metadata["num_complexes"] == 1
        assert result.metadata["model_name"] == "bioemu-v1.1"

        for structure in result.ensembles[0].structures:
            assert isinstance(structure, Structure)
            assert structure.structure_pdb is not None
            assert len(structure.structure_pdb) > 0

    def test_multiple_complexes(
        self,
        sample_sequence: str,
        sample_pdb_content: str,
    ):
        """Test that multiple complexes produce separate ensembles."""
        complex_ = StructurePredictionComplex(
            chains=[{"sequence": sample_sequence, "entity_type": "protein"}]
        )
        bioemu_input = BioEmuInput(complexes=[complex_, complex_])
        bioemu_config = BioEmuConfig(num_samples=10, verbose=False)

        with patch(
            "bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample.ToolInstance",
        ) as mock_cls:
            mock_cls.dispatch.return_value = {
                "results": [
                    {
                        "pdb_frames": [sample_pdb_content] * 3,
                        "num_frames": 3,
                        "num_residues": len(sample_sequence),
                    },
                    {
                        "pdb_frames": [sample_pdb_content] * 7,
                        "num_frames": 7,
                        "num_residues": len(sample_sequence),
                    },
                ]
            }
            result = run_bioemu(bioemu_input, bioemu_config)

        assert len(result.ensembles) == 2
        assert len(result.ensembles[0].structures) == 3
        assert len(result.ensembles[1].structures) == 7
        assert result.metadata["num_complexes"] == 2
        assert result.metadata["total_structures"] == 10
