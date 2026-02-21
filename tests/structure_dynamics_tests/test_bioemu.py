"""Unit tests for the BioEmu conformational ensemble sampling tool."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from bio_programming_tools.entities.structures import Structure, StructureEnsemble
from bio_programming_tools.tools.structure_dynamics.bioemu import (
    BioEmuConfig,
    BioEmuInput,
    BioEmuOutput,
    run_bioemu,
)
from bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample import (
    _pdb_frames_to_structures,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
)


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


@pytest.fixture
def single_chain_complex(sample_sequence: str) -> StructurePredictionComplex:
    """Create a valid single-chain protein complex."""
    return StructurePredictionComplex(
        chains=[{"sequence": sample_sequence, "entity_type": "protein"}]
    )


class TestBioEmuInput:
    """Tests for BioEmuInput validation."""

    def test_new_module_path_import(self):
        """Test importing symbols from the new per-tool module path."""
        from bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample import (
            BioEmuConfig as BioEmuConfigFromModule,
        )
        from bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample import (
            BioEmuInput as BioEmuInputFromModule,
        )
        from bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample import (
            run_bioemu as run_bioemu_from_module,
        )

        assert BioEmuInputFromModule is not None
        assert BioEmuConfigFromModule is not None
        assert callable(run_bioemu_from_module)

    def test_valid_single_chain_protein(
        self, single_chain_complex: StructurePredictionComplex
    ):
        """Test that a valid single-chain protein input is accepted."""
        bioemu_input = BioEmuInput(complexes=[single_chain_complex])
        assert len(bioemu_input.complexes) == 1

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
    """Tests for BioEmuConfig defaults and validation."""

    def test_default_values(self):
        """Test default config values."""
        config = BioEmuConfig()
        assert config.num_samples == 500
        assert config.model_name == "bioemu-v1.1"
        assert config.filter_samples is True
        assert config.batch_size == 10
        assert config.output_dir is None

    def test_custom_values(self):
        """Test custom config values."""
        config = BioEmuConfig(
            num_samples=1000,
            model_name="bioemu-v1.0",
            filter_samples=False,
            batch_size=32,
        )
        assert config.num_samples == 1000
        assert config.model_name == "bioemu-v1.0"
        assert config.filter_samples is False
        assert config.batch_size == 32

    def test_invalid_values(self):
        """Test config validation for invalid values."""
        with pytest.raises(ValueError):
            BioEmuConfig(num_samples=0)
        with pytest.raises(ValueError):
            BioEmuConfig(batch_size=0)
        with pytest.raises(ValueError):
            BioEmuConfig(model_name="invalid-model")


class TestBioEmuOutput:
    """Tests for BioEmuOutput schema."""

    def test_creation(self, sample_sequence: str):
        """Test creating a BioEmuOutput."""
        mock_structure = Mock(spec=Structure)
        ensemble = StructureEnsemble(
            structures=[mock_structure] * 5,
            sequence=sample_sequence,
        )

        output = BioEmuOutput(
            ensembles=[ensemble],
            metadata={
                "num_complexes": 1,
                "total_structures": 5,
                "model_name": "bioemu-v1.1",
            },
        )

        assert len(output.ensembles) == 1
        assert output.metadata["num_complexes"] == 1
        assert output.metadata["total_structures"] == 5


class TestHelpers:
    """Tests for helper behavior."""

    def test_empty_pdb_frames(self):
        """Test conversion with empty frame list."""
        assert _pdb_frames_to_structures([], comp_idx=0) == []


class TestRunBioEmu:
    """Integration-style tests for run_bioemu."""

    @pytest.mark.include_in_env_report
    def test_local_execution_uses_tool_instance(
        self,
        single_chain_complex: StructurePredictionComplex,
        sample_pdb_content: str,
    ):
        """Test local execution through ToolInstance standalone boundary."""
        bioemu_input = BioEmuInput(complexes=[single_chain_complex])
        bioemu_config = BioEmuConfig(num_samples=10, verbose=False)

        with patch(
            "bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample.ToolInstance",
        ) as mock_cls:
            mock_cls.dispatch.return_value = {
                "results": [
                    {
                        "pdb_frames": [sample_pdb_content] * 10,
                        "num_frames": 10,
                        "num_residues": len(
                            single_chain_complex.chains[0].sequence
                        ),
                    }
                ]
            }
            result = run_bioemu(bioemu_input, bioemu_config)

        assert isinstance(result, BioEmuOutput)
        assert len(result.ensembles) == 1
        assert len(result.ensembles[0].structures) == 10
        assert result.metadata["num_complexes"] == 1
        assert result.metadata["total_structures"] == 10

        call_args = mock_cls.dispatch.call_args
        assert call_args[0][1]["sequences"] == [
            single_chain_complex.chains[0].sequence
        ]

    def test_local_execution_multiple_complexes(
        self,
        single_chain_complex: StructurePredictionComplex,
        sample_pdb_content: str,
    ):
        """Test local execution with multiple complexes."""
        bioemu_input = BioEmuInput(complexes=[single_chain_complex, single_chain_complex])
        bioemu_config = BioEmuConfig(num_samples=10, verbose=False)

        with patch(
            "bio_programming_tools.tools.structure_dynamics.bioemu.bioemu_sample.ToolInstance",
        ) as mock_cls:
            mock_cls.dispatch.return_value = {
                "results": [
                    {
                        "pdb_frames": [sample_pdb_content] * 3,
                        "num_frames": 3,
                        "num_residues": len(
                            single_chain_complex.chains[0].sequence
                        ),
                    },
                    {
                        "pdb_frames": [sample_pdb_content] * 7,
                        "num_frames": 7,
                        "num_residues": len(
                            single_chain_complex.chains[0].sequence
                        ),
                    },
                ]
            }
            result = run_bioemu(bioemu_input, bioemu_config)

        assert len(result.ensembles) == 2
        assert len(result.ensembles[0].structures) == 3
        assert len(result.ensembles[1].structures) == 7
        assert result.metadata["num_complexes"] == 2
        assert result.metadata["total_structures"] == 10
