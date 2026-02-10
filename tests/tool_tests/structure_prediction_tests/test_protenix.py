"""
test_protenix.py

Focused tests for Protenix model variants to ensure all available models work correctly.
Tests each of the 8 Protenix model variants with a simple protein input.
"""

import pytest

from bio_programming.bio_tools.tools.structure_prediction import (
    Chain,
    ProtenixConfig,
    ProtenixInput,
    StructurePredictionComplex,
    run_protenix,
)

# All 8 Protenix model variants
PROTENIX_MODEL_VARIANTS = [
    "protenix_base_default_v1.0.0",  # Default base model (highest accuracy)
    "protenix_base_20250630_v1.0.0",  # Base with newer training data
    "protenix_base_default_v0.5.0",  # Earlier base version
    "protenix_base_constraint_v0.5.0",  # Supports constraints
    "protenix_mini_default_v0.5.0",  # Compact model
    "protenix_mini_esm_v0.5.0",  # Mini with ESM2 embeddings
    "protenix_mini_ism_v0.5.0",  # Mini with ISM embeddings
    "protenix_tiny_default_v0.5.0",  # Smallest/fastest variant
]


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_name", PROTENIX_MODEL_VARIANTS)
def test_protenix_model_variants(model_name):
    """
    Test that each Protenix model variant successfully folds a simple protein.

    Uses Cro repressor (a simple 66-residue protein) as test input to verify
    that each model variant can:
    - Load successfully
    - Complete inference
    - Return valid structure output
    - Generate expected metrics (ptm, iptm, avg_plddt)
    """
    # Simple test protein: Cro repressor from bacteriophage lambda
    chains = [
        Chain(
            sequence="MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA",
            entity_type="protein"
        )
    ]

    complexes = [StructurePredictionComplex(chains=chains)]
    inputs = ProtenixInput(complexes=complexes)

    # Use minimal inference settings for faster testing
    config = ProtenixConfig(
        name=f"test_{model_name}",
        model_name=model_name,
        use_msa=False,  # Skip MSA for faster testing
        num_diffusion_samples=1,  # Minimal samples
        num_diffusion_steps=50,  # Reduced from default 200
        seeds=[42],  # Single deterministic seed
        verbose=False
    )

    # Run prediction
    output = run_protenix(inputs, config)

    # Validate output structure
    assert output is not None, f"Model {model_name} returned None"
    assert len(output.structures) == 1, f"Expected 1 structure, got {len(output.structures)}"

    structure = output.structures[0]

    # Validate basic structure properties
    assert structure.structure_pdb is not None, f"No structure PDB for {model_name}"
    assert len(structure.structure_pdb) > 0, f"Empty structure PDB for {model_name}"

    # Validate metrics exist
    assert structure.metrics is not None, f"No metrics for {model_name}"

    # Check for expected Protenix metrics
    expected_metrics = ["ptm", "iptm", "avg_plddt"]
    for metric in expected_metrics:
        assert metric in structure.metrics, f"Missing {metric} for {model_name}"
        assert isinstance(structure.metrics[metric], (int, float)), \
            f"{metric} should be numeric for {model_name}"
        assert 0 <= structure.metrics[metric] <= 1, \
            f"{metric} out of range [0,1] for {model_name}"

    # Additional Protenix-specific metric
    if "gpde" in structure.metrics:
        assert isinstance(structure.metrics["gpde"], (int, float)), \
            f"gpde should be numeric for {model_name}"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_protenix_mini_models_with_msa():
    """
    Test that mini models work with MSA enabled.

    The mini_esm and mini_ism variants use different protein language model
    embeddings, so this verifies they can integrate MSA data correctly.
    """
    chains = [
        Chain(
            sequence="MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA",
            entity_type="protein"
        )
    ]

    complexes = [StructurePredictionComplex(chains=chains)]

    mini_models = [
        "protenix_mini_default_v0.5.0",
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_ism_v0.5.0",
    ]

    for model_name in mini_models:
        inputs = ProtenixInput(complexes=complexes)
        config = ProtenixConfig(
            name=f"test_{model_name}_msa",
            model_name=model_name,
            use_msa=True,
            num_diffusion_samples=1,
            num_diffusion_steps=50,
            seeds=[42],
            verbose=False
        )

        output = run_protenix(inputs, config)

        assert output is not None, f"Model {model_name} with MSA returned None"
        assert len(output.structures) == 1
        assert output.structures[0].metrics["avg_plddt"] > 0
