"""tests/structure_prediction_tests/test_structure_prediction.py.

Tests for structure prediction tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from Bio import SeqIO
from pydantic import ValidationError

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    AlphaFold2Config,
    AlphaFold2Input,
    AlphaFold3Config,
    AlphaFold3Input,
    Boltz2Config,
    Boltz2Input,
    Chai1Config,
    Chai1Input,
    ESMFoldConfig,
    ESMFoldInput,
    ProtenixConfig,
    ProtenixInput,
    StructurePredictionComplex,
    StructurePredictionOutput,
    run_alphafold2,
    run_alphafold3,
    run_boltz2,
    run_chai1,
    run_esmfold,
    run_protenix,
)
from proto_tools.utils.tool_cache import (
    ToolCache,
    _program_tool_cache,
    get_cache_info,
)
from proto_tools.utils.tool_instance import ToolInstance
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Constants ─────────────────────────────────────────────────────────────────

_FASTA_DIR = Path(__file__).parent.parent / "dummy_data" / "structure_prediction_test_examples"

_STRUCTURE_PREDICTORS = {
    "esmfold": (run_esmfold, ESMFoldInput, ESMFoldConfig),
    "alphafold2": (run_alphafold2, AlphaFold2Input, AlphaFold2Config),
    "alphafold3": (run_alphafold3, AlphaFold3Input, AlphaFold3Config),
    "chai1": (run_chai1, Chai1Input, Chai1Config),
    "boltz2": (run_boltz2, Boltz2Input, Boltz2Config),
    "protenix": (run_protenix, ProtenixInput, ProtenixConfig),
}

_FAST_PREDICTORS = ["esmfold"]

# Predictors that require Chimera cluster (hardcoded paths to /large_storage/hielab/brk/)
_CHIMERA_ONLY_PREDICTORS = ["alphafold3"]

# Short sequences used by batched-inference tests, kept at module level so they
# are visible to all flat functions that share this fixture data.
_BATCHED_TEST_SEQUENCES = [
    "MARFLGL",  # Short sequence
    "GARYTWMK",  # Medium sequence
    "YTWHKLAR",  # Another medium sequence
]


# ── File loading ───────────────────────────────────────────────────────────────


def _parse_modifications_from_header(description: str) -> list[tuple]:
    """Parse modifications from FASTA header.

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
    for raw_mod in mod_string.split(","):
        mod = raw_mod.strip()
        if ":" in mod:
            pos_str, code = mod.split(":")
            modifications.append((int(pos_str), code.strip()))

    return modifications


def _parse_fasta_to_complexes(fasta_file: Path) -> list[StructurePredictionComplex]:
    """Parse a FASTA file into a list of StructurePredictionComplex objects.

    FASTA format supports modifications in the header:
    >name|entity_type|position:code,position:code

    Examples:
    >peptide|protein|5:SEP,10:TPO  (protein with modifications at positions 5 and 10)
    >peptide|protein  (protein without modifications)
    """
    chains_data = []

    for record in SeqIO.parse(str(fasta_file), "fasta"):
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
    if "two_complex" not in fasta_file.name:
        complexes = [StructurePredictionComplex(chains=chains_data)]
    else:
        # For two_complex.fasta, treat each sequence as a separate complex to test
        # multi-complex prediction.
        complexes = [StructurePredictionComplex(chains=[chain_dict]) for chain_dict in chains_data]

    return complexes


def _load_all_test_complexes() -> dict:
    """Pre-load all FASTA files and parse them into complexes."""
    test_complexes = {}
    for fasta_file in _FASTA_DIR.glob("*.fasta"):
        complexes = _parse_fasta_to_complexes(fasta_file)
        test_complexes[fasta_file.stem] = complexes
    return test_complexes


# Pre-load all test complexes at module level.
_TEST_COMPLEXES = _load_all_test_complexes()


# ── Test parameterization helpers ──────────────────────────────────────────────


def _supports_msa(config_class) -> bool:
    """Check if a config class supports MSA."""
    return hasattr(config_class, "model_fields") and "use_msa" in config_class.model_fields


def _get_complex_entity_types(complexes: list[StructurePredictionComplex]) -> set:
    """Get all entity types present in a list of complexes."""
    entity_types = set()
    for comp in complexes:
        entity_types.update(comp.get_entity_type_set())
    return entity_types


def _has_modifications(complexes: list[StructurePredictionComplex]) -> bool:
    """Check if any complex has modifications."""
    return any(comp.has_modifications() for comp in complexes)


def _is_compatible_input_for_test(complexes, input_class) -> bool:
    """Check if complexes are compatible with an input class.

    Checks:
    1. Entity types are supported (SUPPORTED_ENTITY_TYPES)
    2. Modifications are allowed if present (ALLOWS_CHAIN_MODIFICATIONS)
    """
    complex_entity_types = _get_complex_entity_types(complexes)
    supported_types = input_class.SUPPORTED_ENTITY_TYPES
    if not complex_entity_types.issubset(supported_types):
        return False

    return not (_has_modifications(complexes) and not input_class.ALLOWS_CHAIN_MODIFICATIONS)


def _generate_test_params() -> list:
    """Generate all valid test parameter combinations.

    Outer loop is predictor_name so all tests for one predictor run together,
    enabling per-predictor worker release to free GPU memory between models.

    Filters out incompatible combinations based on:
    - Input class SUPPORTED_ENTITY_TYPES
    - Input class ALLOWS_CHAIN_MODIFICATIONS
    - MSA support
    """
    params = []

    for predictor_name, (_, input_class, config_class) in _STRUCTURE_PREDICTORS.items():
        for test_name, complexes in _TEST_COMPLEXES.items():
            if not _is_compatible_input_for_test(complexes, input_class):
                continue

            # Generate MSA variants if supported
            if _supports_msa(config_class):
                marks = []
                if predictor_name not in _FAST_PREDICTORS:
                    marks.append(pytest.mark.slow)
                if predictor_name in _CHIMERA_ONLY_PREDICTORS:
                    marks.append(pytest.mark.only_chimera)

                params.append(
                    pytest.param(
                        test_name,
                        predictor_name,
                        True,  # use_msa
                        "remote",  # msa_search_mode
                        id=f"{test_name}-{predictor_name}-with_msa-remote",
                        marks=tuple(marks) if marks else (),
                    )
                )

            # Collect marks for the without_msa variant (all predictors support this)
            marks = []
            if predictor_name not in _FAST_PREDICTORS:
                marks.append(pytest.mark.slow)
            if predictor_name in _CHIMERA_ONLY_PREDICTORS:
                marks.append(pytest.mark.only_chimera)

            params.append(
                pytest.param(
                    test_name,
                    predictor_name,
                    False,  # use_msa
                    None,  # msa_search_mode
                    id=f"{test_name}-{predictor_name}-without_msa",
                    marks=tuple(marks) if marks else (),
                )
            )

    return params


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def _persistent_tool_instances(request):
    """Keep tool workers alive across tests within this module.

    Structure prediction tests reuse GPU models across test cases.
    ToolInstance.persist() keeps workers cached across dispatch() calls for the
    duration of this module and cleans them up on exit.
    """
    if request.config.getoption("--cpu"):
        yield
        return
    with ToolInstance.persist():
        yield


@pytest.fixture
def _release_between_predictors(request):
    """Release tool workers when switching to a different predictor.

    Works inside the module-scoped ToolInstance.persist() context from the
    _persistent_tool_instances fixture above. When predictor_name changes between
    consecutive tests, shuts down the previous predictor's worker so GPU memory
    is freed before the next model loads.

    Not autouse; only parametrized tests have callspec.params. Applied
    via @pytest.mark.usefixtures on test_folding only.
    """
    prev = getattr(request.module, "_active_predictor", None)
    predictor_name = request.node.callspec.params["predictor_name"]
    if prev is not None and prev != predictor_name:
        ToolInstance.shutdown_instance(prev)

    request.module._active_predictor = predictor_name
    yield


# ── Validation tests (no GPU required) ────────────────────────────────────────


def test_esmfold_input_rejects_invalid_amino_acid():
    with pytest.raises(ValidationError, match="unsupported entity types"):
        ESMFoldInput(complexes=["MKTL!"])


def test_esmfold_input_rejects_sequence_too_long():
    with pytest.raises(ValidationError, match=r"too long|2400"):
        ESMFoldInput(complexes=["M" * 2401])


def test_esmfold_input_rejects_non_protein_entity():
    with pytest.raises(ValidationError, match=r"unsupported entity types|only supports"):
        ESMFoldInput(complexes=[StructurePredictionComplex(chains=[{"sequence": "ATCG", "entity_type": "dna"}])])


def test_structure_prediction_complex_rejects_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        StructurePredictionComplex(chains=[""])


def test_structure_prediction_complex_rejects_entity_types_param():
    with pytest.raises(ValidationError, match="entity_types"):
        StructurePredictionComplex(chains=["MKTL"], entity_types=["protein"])


# ── Primary folding test (GPU) ─────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.usefixtures("_release_between_predictors")
@pytest.mark.parametrize(
    "test_name,predictor_name,use_msa,msa_search_mode",
    _generate_test_params(),
)
def test_folding(test_name, predictor_name, use_msa, msa_search_mode):
    complexes = _TEST_COMPLEXES[test_name]

    run_func, input_class, config_class = _STRUCTURE_PREDICTORS[predictor_name]

    # Create input (should always succeed since we filtered incompatible combinations)
    inputs = input_class(complexes=complexes)

    # Create config with MSA settings if supported
    if _supports_msa(config_class):
        config = config_class(use_msa=use_msa, verbose=True)

        if use_msa and msa_search_mode is not None:
            from proto_tools.tools.sequence_alignment.colabfold_search.colabfold_search import (
                ColabfoldSearchConfig,
            )

            config.colabfold_search_config = ColabfoldSearchConfig(search_mode=msa_search_mode, verbose=True)
    else:
        config = config_class(verbose=True)

    output = run_func(inputs, config)

    validate_output(output)

    assert isinstance(output, StructurePredictionOutput), f"Output is not a StructurePredictionOutput: {output}"

    num_predicted_structures = len(output.structures)
    assert num_predicted_structures == len(complexes), (
        f"Input contained {len(complexes)} complexes, but output contains "
        f"{num_predicted_structures} predicted structures"
    )

    for i, structure in enumerate(output.structures):
        if not is_valid_structure(structure.structure_cif):
            pytest.fail(f"Predicted structure {i} is not valid: {structure.structure_cif}")

        metrics = structure.metrics

        # pLDDT
        assert "avg_plddt" in metrics, f"'avg_plddt' not found in {predictor_name} metrics"
        if predictor_name == "alphafold3":
            assert 0 <= metrics["avg_plddt"] <= 100, f"'avg_plddt' has invalid value of {metrics['avg_plddt']}"
        else:
            assert 0 <= metrics["avg_plddt"] <= 1.0, f"'avg_plddt' has invalid value of {metrics['avg_plddt']}"

        # pTM
        assert "ptm" in metrics, f"'ptm' not found in {predictor_name} metrics"
        assert 0 <= metrics["ptm"] <= 1.0, f"'ptm' has invalid value of {metrics['ptm']}"

        # iPTM
        if predictor_name != "esmfold":
            assert "iptm" in metrics, f"'iptm' not found in {predictor_name} metrics"
            assert metrics["iptm"] is None or 0 <= metrics["iptm"] <= 1.0, (
                f"'iptm' has invalid value of {metrics['iptm']}"
            )

        # PAE / GPDE
        if predictor_name == "protenix":
            assert "gpde" in metrics, f"'gpde' not found in {predictor_name} metrics"
            assert metrics["gpde"] >= 0, f"'gpde' has invalid value of {metrics['gpde']}"
        else:
            assert "avg_pae" in metrics, f"'avg_pae' not found in {predictor_name} metrics"
            assert 0 <= metrics["avg_pae"] <= 31.75, f"'avg_pae' has invalid value of {metrics['avg_pae']}"


# ── Cache test (GPU) ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_folding_cache():
    """Test caching functionality of structure prediction tools using ESMFold.

    Directly manipulates the _program_tool_cache ContextVar to activate the
    cache for this test only. This is the intended test-time API for the cache
    infrastructure; there is no higher-level public equivalent.
    """
    # Create short test complexes
    complexes_first_pass = [["MAR"], ["GAR"], ["YTW"]]
    complexes_second_pass = [["MAR"], ["GAR"], ["YTW"], ["MAT"]]

    cache = ToolCache()
    _program_tool_cache.set(cache)

    try:
        complexes = [StructurePredictionComplex(chains=chain) for chain in complexes_first_pass]
        inputs = ESMFoldInput(complexes=complexes)
        output_first_pass = run_esmfold(inputs=inputs, config=ESMFoldConfig())

        validate_output(output_first_pass)

        assert len(output_first_pass.structures) == 3

        # Cache should have three entries (one per complex)
        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 3

        # Run the second pass with overlapping complexes
        complexes = [StructurePredictionComplex(chains=chain) for chain in complexes_second_pass]
        inputs = ESMFoldInput(complexes=complexes)
        output_second_pass = run_esmfold(inputs, ESMFoldConfig())

        validate_output(output_second_pass)

        assert len(output_second_pass.structures) == 4

        # Cache should have four entries (first three were cached, one new)
        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 4

        # Verify structures are consistent; first three should match exactly
        assert output_second_pass.structures[0].structure_cif == output_first_pass.structures[0].structure_cif
        assert output_second_pass.structures[1].structure_cif == output_first_pass.structures[1].structure_cif
        assert output_second_pass.structures[2].structure_cif == output_first_pass.structures[2].structure_cif

    finally:
        _program_tool_cache.set(None)


# ---------------------------------------------------------------------------
# Batched ESMFold inference tests (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
def test_batched_vs_individual_inference_consistency():
    """Batched and per-sequence ESMFold results should agree within tolerance.

    For properly masked metrics (pLDDT, PTM), values should be close.
    Tolerance is loose enough to allow normal numerical variation from
    padding/half-precision/GPU non-determinism, but tight enough to catch
    bugs such as wrong dimension slicing (which would produce completely wrong
    values).
    """
    complexes = [StructurePredictionComplex(chains=[seq]) for seq in _BATCHED_TEST_SEQUENCES]

    # Run batched inference (all sequences in one call)
    batched_input = ESMFoldInput(complexes=complexes)
    batched_config = ESMFoldConfig(
        max_batch_residues=10000,  # High limit to force all into one batch
    )
    batched_output = run_esmfold(batched_input, batched_config)

    validate_output(batched_output)

    assert len(batched_output.structures) == len(_BATCHED_TEST_SEQUENCES)

    # Run individual inference (one sequence at a time)
    individual_outputs = []
    for i, seq in enumerate(_BATCHED_TEST_SEQUENCES):
        single_complex = [StructurePredictionComplex(chains=[seq])]
        single_input = ESMFoldInput(complexes=single_complex)
        single_output = run_esmfold(single_input, ESMFoldConfig())

        assert single_output.success, f"Individual ESMFold failed for sequence {i}"
        individual_outputs.append(single_output.structures[0])

    for i, (batched_struct, individual_struct) in enumerate(
        zip(batched_output.structures, individual_outputs, strict=False)
    ):
        batched_metrics = batched_struct.metrics
        individual_metrics = individual_struct.metrics

        assert batched_metrics["avg_plddt"] == pytest.approx(individual_metrics["avg_plddt"], rel=0.10, abs=0.05), (
            f"Sequence {i}: avg_plddt mismatch "
            f"(batched={batched_metrics['avg_plddt']}, "
            f"individual={individual_metrics['avg_plddt']})"
        )

        assert batched_metrics["ptm"] == pytest.approx(individual_metrics["ptm"], rel=0.10, abs=0.02), (
            f"Sequence {i}: ptm mismatch (batched={batched_metrics['ptm']}, individual={individual_metrics['ptm']})"
        )

        if batched_metrics["avg_pae"] is not None:
            assert batched_metrics["avg_pae"] == pytest.approx(individual_metrics["avg_pae"], rel=0.10, abs=0.05), (
                f"Sequence {i}: avg_pae mismatch "
                f"(batched={batched_metrics['avg_pae']}, "
                f"individual={individual_metrics['avg_pae']})"
            )

        assert is_valid_structure(batched_struct.structure_cif), f"Sequence {i}: Batched structure is invalid"
        assert is_valid_structure(individual_struct.structure_cif), f"Sequence {i}: Individual structure is invalid"


@pytest.mark.uses_gpu
def test_batched_inference_varying_lengths():
    """Batched ESMFold handles sequences of varying lengths correctly."""
    varying_length_sequences = [
        "MAR",  # 3 residues
        "GARYTWMKL",  # 9 residues
        "YTWHKLARFGMVLSPADKTN",  # 20 residues
    ]

    complexes = [StructurePredictionComplex(chains=[seq]) for seq in varying_length_sequences]

    batched_input = ESMFoldInput(complexes=complexes)
    batched_config = ESMFoldConfig(max_batch_residues=10000)
    batched_output = run_esmfold(batched_input, batched_config)

    validate_output(batched_output)

    assert len(batched_output.structures) == len(varying_length_sequences)

    for i, structure in enumerate(batched_output.structures):
        assert is_valid_structure(structure.structure_cif), f"Structure {i} is invalid"

        metrics = structure.metrics
        assert 0 <= metrics["avg_plddt"] <= 1.0, f"Invalid avg_plddt for structure {i}"
        assert 0 <= metrics["ptm"] <= 1.0, f"Invalid ptm for structure {i}"


@pytest.mark.uses_gpu
def test_batched_multichain_complexes():
    """Batched ESMFold correctly handles multi-chain complexes."""
    multichain_complexes = [
        StructurePredictionComplex(chains=["MARFGL", "GARYTWM"]),  # 2 chains
        StructurePredictionComplex(chains=["YTWHK"]),  # 1 chain
        StructurePredictionComplex(chains=["LAR", "FGM", "VLS"]),  # 3 chains
    ]

    batched_input = ESMFoldInput(complexes=multichain_complexes)
    batched_config = ESMFoldConfig(max_batch_residues=10000)
    batched_output = run_esmfold(batched_input, batched_config)

    validate_output(batched_output)

    assert len(batched_output.structures) == len(multichain_complexes)

    expected_chain_counts = [2, 1, 3]
    for i, (structure, expected_chains) in enumerate(
        zip(batched_output.structures, expected_chain_counts, strict=False)
    ):
        assert is_valid_structure(structure.structure_cif), f"Structure {i} is invalid"
        assert structure.num_chains == expected_chains, (
            f"Structure {i}: expected {expected_chains} chains, got {structure.num_chains}"
        )
