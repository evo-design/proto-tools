"""tests/structure_prediction_tests/test_structure_prediction.py.

Cross-tool integration tests for structure prediction tools.

Per-tool benchmarks (``benchmark_twice``-based cold/warm timings) live in the
respective ``test_{toolkit}.py`` files alongside any tool-specific tests.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    AlphaFold2Config,
    AlphaFold2Input,
    AlphaFold2Output,
    AlphaFold3Config,
    AlphaFold3Input,
    AlphaFold3Output,
    Boltz2Config,
    Boltz2Input,
    Boltz2Output,
    Chai1Config,
    Chai1Input,
    Chai1Output,
    Complex,
    ESMFoldConfig,
    ESMFoldInput,
    ESMFoldOutput,
    ProtenixConfig,
    ProtenixInput,
    ProtenixOutput,
    StructurePredictionOutput,
    run_alphafold2,
    run_alphafold3,
    run_boltz2,
    run_chai1,
    run_esmfold,
    run_protenix,
)
from proto_tools.utils.standalone_helpers_source.standalone_helpers import resolve_weights_dir
from proto_tools.utils.tool_cache import (
    ToolCache,
    _program_tool_cache,
    get_cache_info,
)
from proto_tools.utils.tool_instance import ToolInstance
from tests.structure_prediction_tests._fasta_helpers import load_all_test_complexes
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Constants ─────────────────────────────────────────────────────────────────

_STRUCTURE_PREDICTORS = {
    "esmfold": (run_esmfold, ESMFoldInput, ESMFoldConfig, ESMFoldOutput),
    "alphafold2": (run_alphafold2, AlphaFold2Input, AlphaFold2Config, AlphaFold2Output),
    "alphafold3": (run_alphafold3, AlphaFold3Input, AlphaFold3Config, AlphaFold3Output),
    "chai1": (run_chai1, Chai1Input, Chai1Config, Chai1Output),
    "boltz2": (run_boltz2, Boltz2Input, Boltz2Config, Boltz2Output),
    "protenix": (run_protenix, ProtenixInput, ProtenixConfig, ProtenixOutput),
}

_FAST_PREDICTORS = ["esmfold"]


def _missing_weights_skip_reason(predictor_name: str) -> str | None:
    """Return a precise pytest skip reason if the predictor's weights are missing.

    None means no skip needed. Currently only AlphaFold3 has gated weights
    (DeepMind ToU, see proto_tools/tools/structure_prediction/alphafold3/README.md).
    All other predictors pull public weights during setup.sh.
    """
    if predictor_name != "alphafold3":
        return None
    weights_dir = resolve_weights_dir("alphafold3")
    if weights_dir is None:
        return (
            "AlphaFold3 weights dir could not be resolved "
            "(set PROTO_ALPHAFOLD3_WEIGHTS_DIR or PROTO_MODEL_CACHE/PROTO_HOME)"
        )
    if not any(Path(weights_dir).glob("*.bin*")):
        return (
            f"AlphaFold3 weights (*.bin / *.bin.zst) not found in {weights_dir}. "
            "Request access from DeepMind and set PROTO_ALPHAFOLD3_WEIGHTS_DIR."
        )
    return None


# Short sequences used by batched-inference tests, kept at module level so they
# are visible to all flat functions that share this fixture data.
_BATCHED_TEST_SEQUENCES = [
    "MARFLGL",  # Short sequence
    "GARYTWMK",  # Medium sequence
    "YTWHKLAR",  # Another medium sequence
]


# Pre-load all test complexes at module level (FASTA parsing).
_TEST_COMPLEXES = load_all_test_complexes()


# ── Test parameterization helpers ──────────────────────────────────────────────


def _supports_msa(config_class) -> bool:
    """Check if a config class supports MSA."""
    return hasattr(config_class, "model_fields") and "use_msa" in config_class.model_fields


def _get_complex_entity_types(complexes: list[Complex]) -> set:
    """Get all entity types present in a list of complexes."""
    entity_types = set()
    for comp in complexes:
        entity_types.update(comp.get_entity_type_set())
    return entity_types


def _has_modifications(complexes: list[Complex]) -> bool:
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

    for predictor_name, (_, input_class, config_class, _output_class) in _STRUCTURE_PREDICTORS.items():
        for test_name, complexes in _TEST_COMPLEXES.items():
            if not _is_compatible_input_for_test(complexes, input_class):
                continue

            skip_reason = _missing_weights_skip_reason(predictor_name)
            supports_msa = _supports_msa(config_class)

            # Generate MSA variants if supported
            if supports_msa:
                marks = []
                if predictor_name not in _FAST_PREDICTORS:
                    marks.append(pytest.mark.slow)
                if skip_reason:
                    marks.append(pytest.mark.skip(reason=skip_reason))

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
            if skip_reason:
                marks.append(pytest.mark.skip(reason=skip_reason))

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
    if request.config.getoption("--cpu-only"):
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


def test_esmfold_prepare_complexes_enforces_cap_against_linked_length():
    """A multi-chain complex at the bare-sum cap is over the cap once linkers are inserted."""
    # sum(chains) == 2400 clears the field validator, but the 25-residue linker
    # pushes the linked length to 2425, so the model would fold over its hard cap.
    chains = [{"sequence": "M" * 1200, "entity_type": "protein"} for _ in range(2)]
    inputs = ESMFoldInput(complexes=[Complex(chains=chains)])
    with pytest.raises(ValueError, match=r"2425.*max 2400"):
        inputs.prepare_complexes(chain_linker="G" * 25)


def test_esmfold_input_rejects_non_protein_entity():
    with pytest.raises(ValidationError, match=r"unsupported entity types|only supports"):
        ESMFoldInput(complexes=[Complex(chains=[{"sequence": "ATCG", "entity_type": "dna"}])])


def test_structure_prediction_complex_rejects_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        Complex(chains=[""])


def test_structure_prediction_complex_rejects_entity_types_param():
    with pytest.raises(ValidationError, match="entity_types"):
        Complex(chains=["MKTL"], entity_types=["protein"])


# ── Dict deserialization tests (no GPU required) ────────────────────────────────


def test_normalize_complexes_roundtrip():
    """model_dump() -> model_validate() round-trip exercises the dict branch."""
    original = ESMFoldInput(complexes=["MKTL"])
    dumped = original.model_dump()
    restored = ESMFoldInput.model_validate(dumped)
    assert restored.complexes[0].chains[0].sequence == original.complexes[0].chains[0].sequence


def test_normalize_complexes_rejects_invalid_dict():
    """Invalid dict structure raises ValueError."""
    with pytest.raises(ValidationError):
        ESMFoldInput.model_validate({"complexes": [{"not_chains": "bad"}]})


def test_normalize_complexes_roundtrip_with_modifications():
    """Round-trip with modifications exercises both dict branches."""
    original = ProtenixInput(complexes=[Complex(chains=[{"sequence": "MKTL", "modifications": [(1, "MSE")]}])])
    dumped = original.model_dump()
    restored = ProtenixInput.model_validate(dumped)
    assert restored.complexes[0].chains[0].modifications[0].modification_code == "MSE"


# ── Primary folding test (GPU) ─────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.usefixtures("_release_between_predictors")
@pytest.mark.parametrize(
    "test_name,predictor_name,use_msa,msa_search_mode",
    _generate_test_params(),
)
def test_folding(test_name, predictor_name, use_msa, msa_search_mode):
    complexes = _TEST_COMPLEXES[test_name]

    run_func, input_class, config_class, _output_class = _STRUCTURE_PREDICTORS[predictor_name]

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

    assert_metrics_in_spec(output)


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
        complexes = [Complex(chains=chain) for chain in complexes_first_pass]
        inputs = ESMFoldInput(complexes=complexes)
        output_first_pass = run_esmfold(inputs=inputs, config=ESMFoldConfig())

        validate_output(output_first_pass)

        assert len(output_first_pass.structures) == 3

        # Cache should have three entries (one per complex)
        cache_info = get_cache_info()
        assert cache_info["total_entries"] == 3

        # Run the second pass with overlapping complexes
        complexes = [Complex(chains=chain) for chain in complexes_second_pass]
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
    complexes = [Complex(chains=[seq]) for seq in _BATCHED_TEST_SEQUENCES]

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
        single_complex = [Complex(chains=[seq])]
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

        if batched_metrics.get("avg_pae") is not None:
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

    complexes = [Complex(chains=[seq]) for seq in varying_length_sequences]

    batched_input = ESMFoldInput(complexes=complexes)
    batched_config = ESMFoldConfig(max_batch_residues=10000)
    batched_output = run_esmfold(batched_input, batched_config)

    validate_output(batched_output)
    assert_metrics_in_spec(batched_output)

    assert len(batched_output.structures) == len(varying_length_sequences)

    for i, structure in enumerate(batched_output.structures):
        assert is_valid_structure(structure.structure_cif), f"Structure {i} is invalid"


@pytest.mark.uses_gpu
def test_batched_multichain_complexes():
    """Batched ESMFold correctly handles multi-chain complexes."""
    multichain_complexes = [
        Complex(chains=["MARFGL", "GARYTWM"]),  # 2 chains
        Complex(chains=["YTWHK"]),  # 1 chain
        Complex(chains=["LAR", "FGM", "VLS"]),  # 3 chains
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
