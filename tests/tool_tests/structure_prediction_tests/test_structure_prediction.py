"""
test_structure_prediction.py

Test various structure prediction models
"""

import glob
import os

import pytest
from Bio import SeqIO

from bio_programming.tools.structure_prediction import (
    AlphaFold3Config,
    AlphaFold3Input,
    BoltzConfig,
    BoltzInput,
    ChaiConfig,
    ChaiInput,
    ESMFoldConfig,
    ESMFoldInput,
    ProtenixConfig,
    ProtenixInput,
    StructurePredictionComplex,
    StructurePredictionOutput,
    run_af3,
    run_boltz,
    run_chai,
    run_esmfold,
    run_protenix,
)
from bio_programming.tools.structures import is_valid_structure
from bio_programming.tools.tool_cache import ToolCache
from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output

STRUCTURE_PREDICTORS = {
    "esmfold": (run_esmfold, ESMFoldInput, ESMFoldConfig),
    "af3": (run_af3, AlphaFold3Input, AlphaFold3Config),
    "chai": (run_chai, ChaiInput, ChaiConfig),
    "boltz": (run_boltz, BoltzInput, BoltzConfig),
    "protenix": (run_protenix, ProtenixInput, ProtenixConfig),
}

FAST_PREDICTORS = ["esmfold"]


# =============================================================================
# File Loading
# =============================================================================
def _parse_modifications_from_header(description):
    """
    Parse modifications from FASTA header.

    Format: >name|entity_type|position:code,position:code
    Example: >peptide|protein|5:SEP,10:TPO

    Returns list of (position, code) tuples or empty list if no modifications.
    """
    parts = description.split("|")
    if len(parts) < 3:
        return []

    mod_string = parts[2].strip()
    if not mod_string:
        return []

    modifications = []
    for mod in mod_string.split(","):
        mod = mod.strip()
        if ":" in mod:
            pos_str, code = mod.split(":")
            modifications.append((int(pos_str), code.strip()))

    return modifications


def _parse_fasta_to_complexes(fasta_file):
    """
    Parse a FASTA file into a list of StructurePredictionComplex objects.

    FASTA format supports modifications in the header:
    >name|entity_type|position:code,position:code

    Examples:
    >peptide|protein|5:SEP,10:TPO  (protein with modifications at positions 5 and 10)
    >peptide|protein  (protein without modifications)
    """
    chains_data = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).strip()
        parts = record.description.split("|")
        entity_type = parts[1].strip()
        modifications = _parse_modifications_from_header(record.description)

        chain_dict = {
            "sequence": sequence,
            "entity_type": entity_type,
        }

        if modifications:
            chain_dict["modifications"] = modifications

        chains_data.append(chain_dict)

    # By default, treat all sequences as chains in a single complex
    if "two_complex" not in fasta_file:
        complexes = [StructurePredictionComplex(chains=chains_data)]
    else:
        # For two_complex.fasta, treat each sequence as a separate complex to test multi-complex prediction
        complexes = [
            StructurePredictionComplex(chains=[chain_dict])
            for chain_dict in chains_data
        ]

    return complexes


def _load_all_test_complexes():
    """Pre-load all FASTA files and parse them into complexes."""
    fasta_files = glob.glob("tests/dummy_data/structure_prediction_test_examples/*.fasta")
    test_complexes = {}

    for fasta_file in fasta_files:
        basename = os.path.basename(fasta_file).replace(".fasta", "")
        complexes = _parse_fasta_to_complexes(fasta_file)
        test_complexes[basename] = complexes

    return test_complexes


# Pre-load all test complexes at module level
TEST_COMPLEXES = _load_all_test_complexes()

# =============================================================================
# Test Parameterization
# =============================================================================
def _supports_msa(config_class):
    """Check if a config class supports MSA."""
    return hasattr(config_class, "model_fields") and "use_msa" in config_class.model_fields


def _get_complex_entity_types(complexes):
    """Get all entity types present in a list of complexes."""
    entity_types = set()
    for comp in complexes:
        entity_types.update(comp.get_entity_type_set())
    return entity_types


def _has_modifications(complexes):
    """Check if any complex has modifications."""
    return any(comp.has_modifications() for comp in complexes)


def _is_compatible_input_for_test(complexes, input_class):
    """
    Check if complexes are compatible with an input class.

    Checks:
    1. Entity types are supported (SUPPORTED_ENTITY_TYPES)
    2. Modifications are allowed if present (ALLOWS_CHAIN_MODIFICATIONS)
    """
    # Check entity types
    complex_entity_types = _get_complex_entity_types(complexes)
    supported_types = input_class.SUPPORTED_ENTITY_TYPES
    if not complex_entity_types.issubset(supported_types):
        return False

    # Check modifications
    if _has_modifications(complexes) and not input_class.ALLOWS_CHAIN_MODIFICATIONS:
        return False

    return True


def _generate_test_params():
    """
    Generate all valid test parameter combinations.

    Filters out incompatible combinations based on:
    - Input class SUPPORTED_ENTITY_TYPES
    - Input class ALLOWS_CHAIN_MODIFICATIONS
    - MSA support
    """
    params = []

    for test_name, complexes in TEST_COMPLEXES.items():
        for predictor_name, (_, input_class, config_class) in STRUCTURE_PREDICTORS.items():
            # Skip if complexes are not compatible with this predictor
            if not _is_compatible_input_for_test(complexes, input_class):
                continue

            # Generate MSA variants if supported
            if _supports_msa(config_class):
                # Add remote MSA variant
                params.append(
                    pytest.param(
                        test_name,
                        predictor_name,
                        True,  # use_msa
                        "remote",  # msa_search_mode
                        id=f"{test_name}-{predictor_name}-with_msa-remote",
                        marks=(
                            pytest.mark.slow if predictor_name not in FAST_PREDICTORS else ()
                        ),
                    )
                )

            # Add without_msa variant (all predictors support this)
            params.append(
                pytest.param(
                    test_name,
                    predictor_name,
                    False,  # use_msa
                    None,  # msa_search_mode
                    id=f"{test_name}-{predictor_name}-without_msa",
                    marks=pytest.mark.slow if predictor_name not in FAST_PREDICTORS else (),
                )
            )

    return params


# =============================================================================
# Primary Folding Test
# =============================================================================


@pytest.mark.uses_gpu
@pytest.mark.parametrize(
    "test_name,predictor_name,use_msa,msa_search_mode",
    _generate_test_params(),
)
def test_folding(test_name, predictor_name, use_msa, msa_search_mode):

    # Get pre-loaded complexes
    complexes = TEST_COMPLEXES[test_name]

    # Get predictor function, input class, and config class
    run_func, input_class, config_class = STRUCTURE_PREDICTORS[predictor_name]

    # Create input (should always succeed since we filtered incompatible combinations)
    inputs = input_class(complexes=complexes)

    # Create config with MSA settings (if supported)
    if _supports_msa(config_class):
        config = config_class(use_msa=use_msa, verbose=True)

        # Configure MSA search mode if specified
        if use_msa and msa_search_mode is not None:
            from bio_programming.tools.sequence_alignment.colabfold_search.colabfold_search import (
                ColabfoldSearchConfig,
            )

            config.colabfold_search_config = ColabfoldSearchConfig(
                search_mode=msa_search_mode, verbose=True
            )
    else:
        # Predictor doesn't support MSA (e.g., ESMFold)
        config = config_class(verbose=True)

    # Run prediction
    output = run_func(inputs, config)

    # Validate output and export functionality
    validate_output(output)

    # Ensure the output is a StructurePredictionOutput
    assert isinstance(
        output, StructurePredictionOutput
    ), f"Output is not a StructurePredictionOutput: {output}"

    # Ensure the correct number of predicted structures are returned
    num_predicted_structures = len(output.structures)
    assert num_predicted_structures == len(
        complexes
    ), f"Input contained {len(complexes)} complexes, but output contains {num_predicted_structures} predicted structures"

    # Check that each predicted structure is valid and has expected metrics
    for i, structure in enumerate(output.structures):
        if not is_valid_structure(structure.structure_cif):
            pytest.fail(
                f"Predicted structure {i} is not valid: {structure.structure_cif}"
            )

        metrics = structure.metrics

        # pLDDT.
        assert (
            "avg_plddt" in metrics
        ), f"'avg_plddt' not found in {predictor_name} metrics"
        if predictor_name in ["af3", "alphafold3"]:
            assert (
                0 <= metrics["avg_plddt"] <= 100
            ), f"'avg_plddt' has invalid value of {metrics['avg_plddt']}"
        else:
            assert (
                0 <= metrics["avg_plddt"] <= 1.0
            ), f"'avg_plddt' has invalid value of {metrics['avg_plddt']}"

        # pTM.
        assert "ptm" in metrics, f"'ptm' not found in {predictor_name} metrics"
        assert (
            0 <= metrics["ptm"] <= 1.0
        ), f"'ptm' has invalid value of {metrics['ptm']}"

        # iPTM.
        if predictor_name not in ["esmfold"]:
            assert "iptm" in metrics, f"'iptm' not found in {predictor_name} metrics"
            assert (
                metrics["iptm"] is None or 0 <= metrics["iptm"] <= 1.0
            ), f"'iptm' has invalid value of {metrics['ptm']}"

        # PAE / GPDE.
        if predictor_name == "protenix":
            assert "gpde" in metrics, f"'gpde' not found in {predictor_name} metrics"
            assert (
                0 <= metrics["gpde"]
            ), f"'gpde' has invalid value of {metrics['gpde']}"
        else:
            assert "avg_pae" in metrics, f"'avg_pae' not found in {predictor_name} metrics"
            assert (
                0 <= metrics["avg_pae"] <= 31.75
            ), f"'avg_pae' has invalid value of {metrics['avg_pae']}"


@pytest.mark.uses_gpu
def test_folding_cache():
    """
    Tests the caching functionality of structure prediction tools using small
    toy examples with ESMFold
    """
    from bio_programming.tools.tool_cache import _program_tool_cache, get_cache_info

    # Create short test complexes
    complexes_first_pass = [["MAR"], ["GAR"], ["YTW"]]
    complexes_second_pass = [["MAR"], ["GAR"], ["YTW"], ["MAT"]]

    # Set up the cache
    cache = ToolCache()
    _program_tool_cache.set(cache)

    try:
        # Run the first pass
        complexes = [
            StructurePredictionComplex(chains=chain) for chain in complexes_first_pass
        ]

        inputs = ESMFoldInput(complexes=complexes)

        output_first_pass = run_esmfold(inputs=inputs, config=ESMFoldConfig())

        # Validate output and export functionality
        validate_output(output_first_pass)

        assert len(output_first_pass.structures) == 3

        # Cache should have three entries (one per complex)
        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 3

        # Run the second pass with overlapping complexes
        complexes = [
            StructurePredictionComplex(chains=chain) for chain in complexes_second_pass
        ]
        inputs = ESMFoldInput(complexes=complexes)
        output_second_pass = run_esmfold(inputs, ESMFoldConfig())

        # Validate output and export functionality
        validate_output(output_second_pass)

        assert len(output_second_pass.structures) == 4

        # Cache should have four entries (first three were cached, one new)
        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 4

        # Verify structures are consistent - first three should match exactly
        # Compare structure CIF strings to ensure caching returns identical results
        assert (
            output_second_pass.structures[0].structure_cif
            == output_first_pass.structures[0].structure_cif
        )
        assert (
            output_second_pass.structures[1].structure_cif
            == output_first_pass.structures[1].structure_cif
        )
        assert (
            output_second_pass.structures[2].structure_cif
            == output_first_pass.structures[2].structure_cif
        )

    finally:
        # Clean up cache
        _program_tool_cache.set(None)


@pytest.mark.uses_gpu
class TestBatchedESMFoldInference:
    # Test sequences of varying lengths for batched inference
    TEST_SEQUENCES = [
        "MARFLGL",  # Short sequence
        "GARYTWMK",  # Medium sequence
        "YTWHKLAR",  # Another medium sequence
    ]

    def test_batched_vs_individual_inference_consistency(self):
        # Create complexes from test sequences
        complexes = [
            StructurePredictionComplex(chains=[seq]) for seq in self.TEST_SEQUENCES
        ]

        # Run batched inference (all sequences in one call)
        batched_input = ESMFoldInput(complexes=complexes)
        batched_config = ESMFoldConfig(
            max_batch_residues=10000,  # High limit to force all into one batch
            keep_on_gpu=True,  # Keep on GPU for subsequent individual runs
        )
        batched_output = run_esmfold(batched_input, batched_config)

        # Validate output and export functionality
        validate_output(batched_output)

        assert len(batched_output.structures) == len(self.TEST_SEQUENCES)

        # Run individual inference (one sequence at a time)
        individual_outputs = []
        for i, seq in enumerate(self.TEST_SEQUENCES):
            single_complex = [StructurePredictionComplex(chains=[seq])]
            single_input = ESMFoldInput(complexes=single_complex)
            # Keep on GPU except for the last one
            keep_on_gpu = i < len(self.TEST_SEQUENCES) - 1
            single_config = ESMFoldConfig(keep_on_gpu=keep_on_gpu)
            single_output = run_esmfold(single_input, single_config)

            assert single_output.success, f"Individual ESMFold failed for sequence {i}"
            individual_outputs.append(single_output.structures[0])

        # Compare batched vs individual results
        # For properly masked metrics (pLDDT, PTM), we expect reasonably close values.
        # Use tolerance that catches major bugs (wrong dimension slicing would cause
        # completely wrong values) but allows normal numerical variation from
        # padding/half-precision/GPU non-determinism.
        for i, (batched_struct, individual_struct) in enumerate(
            zip(batched_output.structures, individual_outputs)
        ):
            batched_metrics = batched_struct.metrics
            individual_metrics = individual_struct.metrics

            # avg_plddt, ptm, and avg_pae should be close (within 10% relative or 0.05 absolute)
            assert batched_metrics["avg_plddt"] == pytest.approx(
                individual_metrics["avg_plddt"], rel=0.10, abs=0.05
            ), f"Sequence {i}: avg_plddt mismatch (batched={batched_metrics['avg_plddt']}, individual={individual_metrics['avg_plddt']})"

            assert batched_metrics["ptm"] == pytest.approx(
                individual_metrics["ptm"], rel=0.10, abs=0.02
            ), f"Sequence {i}: ptm mismatch (batched={batched_metrics['ptm']}, individual={individual_metrics['ptm']})"

            if batched_metrics["avg_pae"] is not None:
                assert batched_metrics["avg_pae"] == pytest.approx(
                    individual_metrics["avg_pae"], rel=0.10, abs=0.05
                ), f"Sequence {i}: avg_pae mismatch (batched={batched_metrics['avg_pae']}, individual={individual_metrics['avg_pae']})"

            # Both structures should be valid
            assert is_valid_structure(
                batched_struct.structure_cif
            ), f"Sequence {i}: Batched structure is invalid"
            assert is_valid_structure(
                individual_struct.structure_cif
            ), f"Sequence {i}: Individual structure is invalid"

    def test_batched_inference_varying_lengths(self):
        # Sequences with varying lengths
        varying_length_sequences = [
            "MAR",  # 3 residues
            "GARYTWMKL",  # 9 residues
            "YTWHKLARFGMVLSPADKTN",  # 20 residues
        ]

        complexes = [
            StructurePredictionComplex(chains=[seq]) for seq in varying_length_sequences
        ]

        batched_input = ESMFoldInput(complexes=complexes)
        batched_config = ESMFoldConfig(max_batch_residues=10000)
        batched_output = run_esmfold(batched_input, batched_config)

        # Validate output and export functionality
        validate_output(batched_output)

        assert len(batched_output.structures) == len(varying_length_sequences)

        # Verify each structure is valid and has expected metrics
        for i, structure in enumerate(batched_output.structures):
            assert is_valid_structure(
                structure.structure_cif
            ), f"Structure {i} is invalid"

            metrics = structure.metrics
            assert (
                0 <= metrics["avg_plddt"] <= 1.0
            ), f"Invalid avg_plddt for structure {i}"
            assert 0 <= metrics["ptm"] <= 1.0, f"Invalid ptm for structure {i}"

    def test_batched_multichain_complexes(self):
        # Multi-chain complexes
        multichain_complexes = [
            StructurePredictionComplex(chains=["MARFGL", "GARYTWM"]),  # 2 chains
            StructurePredictionComplex(chains=["YTWHK"]),  # 1 chain
            StructurePredictionComplex(chains=["LAR", "FGM", "VLS"]),  # 3 chains
        ]

        batched_input = ESMFoldInput(complexes=multichain_complexes)
        batched_config = ESMFoldConfig(max_batch_residues=10000)
        batched_output = run_esmfold(batched_input, batched_config)

        # Validate output and export functionality
        validate_output(batched_output)

        assert len(batched_output.structures) == len(multichain_complexes)

        # Verify chain counts in output structures
        expected_chain_counts = [2, 1, 3]
        for i, (structure, expected_chains) in enumerate(
            zip(batched_output.structures, expected_chain_counts)
        ):
            assert is_valid_structure(
                structure.structure_cif
            ), f"Structure {i} is invalid"
            assert (
                structure.num_chains == expected_chains
            ), f"Structure {i}: expected {expected_chains} chains, got {structure.num_chains}"
