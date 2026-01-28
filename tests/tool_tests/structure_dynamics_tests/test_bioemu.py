"""
Unit tests for BioEmu conformational ensemble sampling tool.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from bio_programming.tools.structure_dynamics.bioemu.bioemu import (
    BioEmuConfig,
    BioEmuInput,
    BioEmuOutput,
    _pdb_frames_to_structures,
    run_bioemu,
)
from bio_programming.tools.structure_dynamics.bioemu.inference import BioEmuModel
from bio_programming.tools.structure_prediction.schemas import (
    Chain,
    StructurePredictionComplex,
)
from bio_programming.tools.structures import ProteinStructure, ProteinStructureEnsemble

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_sequence():
    """A short valid protein sequence for testing."""
    return "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"


@pytest.fixture
def sample_pdb_content():
    """Minimal valid PDB content for testing."""
    return """ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00  0.00           C
END
"""


@pytest.fixture
def mock_bioemu_model_result(sample_pdb_content):
    """Mock result from BioEmuModel.__call__."""
    return {
        "pdb_frames": [sample_pdb_content] * 10,
        "num_frames": 10,
        "num_residues": 50,
    }


@pytest.fixture
def mock_complex(sample_sequence):
    """Create a mock StructurePredictionComplex."""
    mock = Mock(spec=StructurePredictionComplex)
    mock.chains = [Chain(sequence=sample_sequence, entity_type="protein")]
    mock.entity_types = ["protein"]
    mock.num_chains.return_value = 1
    mock.get_entity_type_set.return_value = {"protein"}
    mock.has_modifications.return_value = False
    return mock


# =============================================================================
# BioEmuInput Tests
# =============================================================================

class TestBioEmuInput:
    """Tests for BioEmuInput validation."""

    def test_valid_single_chain_protein(self, mock_complex):
        """Test that valid single-chain protein input is accepted."""
        bioemu_input = BioEmuInput(complexes=[mock_complex])
        assert len(bioemu_input.complexes) == 1

    def test_rejects_multi_chain_complex(self, sample_sequence):
        """Test that multi-chain complexes are rejected."""
        mock_complex = Mock()
        mock_complex.chains = [
            Chain(sequence=sample_sequence, entity_type="protein"),
            Chain(sequence=sample_sequence, entity_type="protein")
        ]
        mock_complex.entity_types = ["protein", "protein"]
        mock_complex.num_chains.return_value = 2
        mock_complex.get_entity_type_set.return_value = {"protein"}

        with pytest.raises(ValueError, match="single-chain"):
            BioEmuInput.validate_complexes([mock_complex])

    def test_rejects_non_protein_entity(self, sample_sequence):
        """Test that non-protein entities are rejected."""
        from bio_programming.tools.structure_prediction.schemas import (
            StructurePredictionComplex,
        )

        # Create a complex with DNA entity type
        dna_complex = StructurePredictionComplex(
            chains=[{"sequence": "ACGT", "entity_type": "dna"}]
        )

        with pytest.raises(ValueError, match="only supports: protein"):
            BioEmuInput(complexes=[dna_complex])

    def test_rejects_invalid_amino_acids(self):
        """Test that sequences with invalid characters are rejected."""
        mock_complex = Mock()
        mock_complex.chains = [Chain(sequence="MVLSPADKTNVKAAW123", entity_type="protein")]  # Contains numbers
        mock_complex.entity_types = ["protein"]
        mock_complex.num_chains.return_value = 1
        mock_complex.get_entity_type_set.return_value = {"protein"}

        with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu.return_invalid_protein_chars', return_value={'1', '2', '3'}):
            with pytest.raises(ValueError, match="Invalid protein characters"):
                BioEmuInput.validate_complexes([mock_complex])

    def test_warns_long_sequence(self, caplog):
        """Test that sequences >500 residues trigger a warning."""
        long_sequence = "A" * 600
        mock_complex = Mock()
        mock_complex.chains = [Chain(sequence=long_sequence, entity_type="protein")]
        mock_complex.entity_types = ["protein"]
        mock_complex.num_chains.return_value = 1
        mock_complex.get_entity_type_set.return_value = {"protein"}

        with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu.return_invalid_protein_chars', return_value=set()):
            import logging
            with caplog.at_level(logging.WARNING):
                BioEmuInput.validate_complexes([mock_complex])
            assert "500 residues" in caplog.text or len(caplog.records) > 0


# =============================================================================
# BioEmuConfig Tests
# =============================================================================

class TestBioEmuConfig:
    """Tests for BioEmuConfig defaults and validation."""

    def test_default_values(self):
        """Test that default config values are set correctly."""
        config = BioEmuConfig()
        assert config.num_samples == 500
        assert config.model_name == "bioemu-v1.1"
        assert config.filter_samples is True
        assert config.batch_size == 10
        assert config.output_dir is None

    def test_custom_values(self):
        """Test that custom config values are accepted."""
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

    def test_num_samples_minimum(self):
        """Test that num_samples must be at least 1."""
        with pytest.raises(ValueError):
            BioEmuConfig(num_samples=0)

    def test_batch_size_minimum(self):
        """Test that batch_size must be at least 1."""
        with pytest.raises(ValueError):
            BioEmuConfig(batch_size=0)

    def test_invalid_model_name(self):
        """Test that invalid model names are rejected."""
        with pytest.raises(ValueError):
            BioEmuConfig(model_name="invalid-model")


# =============================================================================
# BioEmuOutput Tests
# =============================================================================

class TestBioEmuOutput:
    """Tests for BioEmuOutput schema."""

    def test_creation(self, sample_sequence):
        """Test creating a BioEmuOutput."""
        mock_structure = Mock(spec=ProteinStructure)
        ensemble = ProteinStructureEnsemble(
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


# =============================================================================
# BioEmuModel (inference) Tests
# =============================================================================

class TestBioEmuModel:
    """Tests for BioEmuModel inference class."""

    def test_initialization(self):
        """Test model initialization state."""
        model = BioEmuModel()
        assert model._loaded is False
        assert model._model_name is None
        assert model.device is None

    def test_call_runs_sampling(self, sample_sequence, sample_pdb_content):
        """Test that __call__ runs the sampling pipeline."""
        model = BioEmuModel()

        with patch.dict('sys.modules', {'bioemu': Mock(), 'bioemu.sample': Mock()}):
            with patch('bio_programming.tools.structure_dynamics.bioemu.inference.BioEmuModel.load'):
                with patch('bio_programming.tools.structure_dynamics.bioemu.inference.BioEmuModel._extract_pdb_frames') as mock_extract:
                    mock_extract.return_value = ([sample_pdb_content] * 5, 5, 50)
                    model._loaded = True
                    model._model_name = "bioemu-v1.1"
                    model.device = "cuda"

                    result = model(
                        sequence=sample_sequence,
                        num_samples=100,
                        model_name="bioemu-v1.1",
                    )

                    assert "pdb_frames" in result
                    assert result["num_frames"] == 5
                    assert result["num_residues"] == 50


# =============================================================================
# Integration Tests (run_bioemu)
# =============================================================================

class TestRunBioemu:
    """Integration tests for run_bioemu function."""

    def test_local_execution(self, mock_complex, mock_bioemu_model_result):
        """Test run_bioemu with local execution."""
        # Setup mocks
        mock_model = Mock()
        mock_model.return_value = mock_bioemu_model_result

        mock_input = Mock(spec=BioEmuInput)
        mock_input.complexes = [mock_complex]

        mock_config = BioEmuConfig(num_samples=10, verbose=False)

        with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu.use_cloud_gpu', return_value=False):
            with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu._get_cached_bioemu_model', return_value=mock_model):
                with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu._pdb_frames_to_structures') as mock_convert:
                    mock_convert.return_value = [Mock(spec=ProteinStructure)] * 10

                    result = run_bioemu(mock_input, mock_config)

        assert isinstance(result, BioEmuOutput)
        assert len(result.ensembles) == 1
        assert result.metadata["num_complexes"] == 1

    def test_multiple_complexes(self, mock_complex, mock_bioemu_model_result):
        """Test run_bioemu with multiple input complexes."""
        mock_input = Mock(spec=BioEmuInput)
        mock_input.complexes = [mock_complex, mock_complex, mock_complex]

        mock_config = BioEmuConfig(num_samples=10, verbose=False)

        mock_model = Mock()
        mock_model.return_value = mock_bioemu_model_result

        with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu.use_cloud_gpu', return_value=False):
            with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu._get_cached_bioemu_model', return_value=mock_model):
                with patch('bio_programming.tools.structure_dynamics.bioemu.bioemu._pdb_frames_to_structures') as mock_convert:
                    mock_convert.return_value = [Mock(spec=ProteinStructure)] * 10

                    result = run_bioemu(mock_input, mock_config)

        assert len(result.ensembles) == 3
        assert result.metadata["num_complexes"] == 3
        assert result.metadata["total_structures"] == 30  # 10 * 3

    def test_empty_pdb_frames(self):
        """Test handling of empty pdb_frames list."""
        structures = _pdb_frames_to_structures([], comp_idx=0)
        assert structures == []

    @pytest.mark.slow
    @pytest.mark.uses_gpu
    def test_full_execution(self, mock_complex, mock_bioemu_model_result):
        """Test BioEmu sampling on GPU."""
        bioemu_input = BioEmuInput(complexes=[mock_complex])
        bioemu_config = BioEmuConfig(num_samples=10, filter_samples=False, verbose=False)

        result = run_bioemu(bioemu_input, bioemu_config)

        assert isinstance(result, BioEmuOutput)
        assert len(result.ensembles) == 1
        assert len(result.ensembles[0].structures) == 10
