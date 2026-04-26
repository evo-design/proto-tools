"""tests/structure_prediction_tests/test_protenix.py.

Tests for Protenix.
"""

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_prediction import (
    Chain,
    ProtenixConfig,
    ProtenixInput,
    StructurePredictionComplex,
    run_protenix,
)
from tests.conftest import make_persistent_fixture

_persistent_tool = make_persistent_fixture("protenix")

# Cro repressor from bacteriophage lambda. Short, well-folded test protein.
_CRO_SEQUENCE = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA"

_PROTENIX_MODEL_VARIANTS = [
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix_base_default_v0.5.0",
    "protenix_base_constraint_v0.5.0",
    "protenix_mini_default_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
    "protenix_tiny_default_v0.5.0",
]

_MINI_MODEL_VARIANTS = [
    "protenix_mini_default_v0.5.0",
    "protenix_mini_esm_v0.5.0",
    "protenix_mini_ism_v0.5.0",
]


# ── Validation tests (no GPU required) ──────────────────────────────────────


def test_protenix_config_rejects_invalid_model_name():
    with pytest.raises(ValidationError, match="model_name"):
        ProtenixConfig(model_name="protenix_nonexistent_v99.0.0")


def test_protenix_config_rejects_zero_diffusion_samples():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ProtenixConfig(num_diffusion_samples=0)


def test_protenix_config_rejects_zero_diffusion_steps():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ProtenixConfig(num_diffusion_steps=0)


def test_protenix_config_rejects_negative_pairformer_cycles():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ProtenixConfig(num_pairformer_cycles=-1)


def test_protenix_config_rejects_zero_timeout():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ProtenixConfig(timeout=0)


def test_protenix_config_defaults():
    config = ProtenixConfig()
    assert config.model_name == "protenix_base_default_v1.0.0"
    assert config.seeds == [0]
    assert config.use_msa is True
    assert config.num_diffusion_samples == 5
    assert config.num_diffusion_steps == 200
    assert config.num_pairformer_cycles == 10


def test_protenix_config_colabfold_lazy_init():
    """colabfold_search_config is None by default, initialized lazily in preprocess."""
    config = ProtenixConfig(verbose=True)
    # Not eagerly initialized; stays None until preprocess() is called
    assert config.colabfold_search_config is None


def test_protenix_input_accepts_string_shorthand():
    """Single-chain string input is normalised to a StructurePredictionComplex."""
    inputs = ProtenixInput(complexes=["MKTL"])
    assert len(inputs.complexes) == 1
    assert inputs.complexes[0].chains[0].sequence == "MKTL"


def test_protenix_input_accepts_chain_objects():
    chain = Chain(sequence=_CRO_SEQUENCE, entity_type="protein")
    complex_ = StructurePredictionComplex(chains=[chain])
    inputs = ProtenixInput(complexes=[complex_])
    assert len(inputs.complexes) == 1
    assert inputs.complexes[0].chains[0].entity_type == "protein"


# ── Ligand JSON shape: CCD-prefer dispatch (#502) ───────────────────────────


def _protenix_ligand_entries(chains):
    """Build a Protenix JSON payload from ``chains`` and return its ligand entries."""
    inputs = ProtenixInput(complexes=[StructurePredictionComplex(chains=chains)])
    payload = inputs.to_json(complex_idx=0, name="test")
    return [entry["ligand"] for entry in payload[0]["sequences"] if "ligand" in entry]


def test_protenix_ligand_uses_ccd_code_when_available():
    """Fragment with a resolved ccd_code serializes to ``CCD_<code>``, not raw SMILES."""
    from proto_tools.entities.ligands import Fragment

    atp = Fragment(ccd_code="ATP")
    assert atp.ccd_code == "ATP"  # invariant guard

    [ligand_entry] = _protenix_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), atp])
    assert ligand_entry == {"ligand": "CCD_ATP", "count": 1}


def test_protenix_ligand_falls_back_to_smiles_when_no_ccd_match():
    """Novel ligand (SMILES with no wwPDB CCD entry) serializes as raw SMILES."""
    from proto_tools.entities.ligands import Fragment

    # Synthetic perfluorinated terphenyl chain — not in the wwPDB CCD database.
    novel_smiles = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    novel = Fragment(smiles=novel_smiles)
    assert novel.ccd_code is None  # invariant guard

    [ligand_entry] = _protenix_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), novel])
    assert ligand_entry == {"ligand": novel.smiles, "count": 1}


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_name", _PROTENIX_MODEL_VARIANTS)
def test_protenix_model_variants(model_name):
    """Each Protenix model variant folds a simple protein and returns valid metrics."""
    complexes = [StructurePredictionComplex(chains=[Chain(sequence=_CRO_SEQUENCE, entity_type="protein")])]
    inputs = ProtenixInput(complexes=complexes)
    config = ProtenixConfig(
        model_name=model_name,
        use_msa=False,
        num_diffusion_samples=1,
        num_diffusion_steps=50,
        seeds=[42],
        verbose=True,
    )

    output = run_protenix(inputs, config)

    assert output.success
    assert len(output.structures) == 1

    structure = output.structures[0]
    assert structure.metrics is not None

    for metric in ("ptm", "iptm", "avg_plddt"):
        assert metric in structure.metrics, f"Missing {metric} for {model_name}"
        value = structure.metrics[metric]
        assert isinstance(value, (int, float)), f"{metric} should be numeric for {model_name}"
        assert 0.0 <= value <= 1.0, f"{metric}={value} out of range [0, 1] for {model_name}"

    if "gpde" in structure.metrics:
        assert isinstance(structure.metrics["gpde"], (int, float))


@pytest.mark.uses_gpu
@pytest.mark.slow
@pytest.mark.parametrize("model_name", _MINI_MODEL_VARIANTS)
def test_protenix_mini_models_with_msa(model_name):
    """Mini model variants (default, esm, ism) succeed with MSA enabled."""
    complexes = [StructurePredictionComplex(chains=[Chain(sequence=_CRO_SEQUENCE, entity_type="protein")])]
    inputs = ProtenixInput(complexes=complexes)
    config = ProtenixConfig(
        model_name=model_name,
        use_msa=True,
        num_diffusion_samples=1,
        num_diffusion_steps=50,
        seeds=[42],
        verbose=True,
    )

    output = run_protenix(inputs, config)

    assert output.success
    assert len(output.structures) == 1
    assert output.structures[0].metrics["avg_plddt"] > 0.0
