"""tests/structure_prediction_tests/test_protenix.py.

Tests for Protenix.
"""

import pytest
from pydantic import ValidationError

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Chain,
    ProtenixConfig,
    ProtenixInput,
    StructurePredictionComplex,
    run_protenix,
)
from proto_tools.utils import ToolInstance
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

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


# ── Ligand JSON shape: CCD-prefer dispatch ─────────────────────────────────


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


@pytest.mark.extensive
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_protenix_ccd_vs_smiles_input_equivalent_predictions():
    """Predictions from CCD and SMILES inputs of the same ligand should agree.

    Empirical check for Protenix's lenient SMILES path. If this test fails,
    consider tightening the implementation to strict CCD-only (the AF3
    implementation does this because AF3 with raw SMILES produces broken
    structures).
    """
    from proto_tools.entities.ligands import Fragment

    # L-tyrosine — known CCD entry "TYR", small enough for fast inference
    tyr_smiles = "c1cc(ccc1C[C@@H](C(=O)O)N)O"
    protein = Chain(sequence=_CRO_SEQUENCE, entity_type="protein")

    # CCD path: validator auto-resolves ccd_code="TYR"; implementation sends "CCD_TYR"
    ccd_frag = Fragment(smiles=tyr_smiles)
    assert ccd_frag.ccd_code == "TYR"  # invariant guard
    ccd_complex = StructurePredictionComplex(chains=[protein, ccd_frag])

    # SMILES path: force ccd_code=None so the implementation falls back to raw SMILES
    smiles_frag = Fragment(smiles=tyr_smiles)
    smiles_frag.ccd_code = None
    smiles_complex = StructurePredictionComplex(chains=[protein, smiles_frag])

    config = ProtenixConfig(
        model_name="protenix_tiny_default_v0.5.0",
        use_msa=False,
        num_diffusion_samples=1,
        num_diffusion_steps=50,
        seeds=[42],
    )

    ccd_output = run_protenix(ProtenixInput(complexes=[ccd_complex]), config)
    smiles_output = run_protenix(ProtenixInput(complexes=[smiles_complex]), config)

    assert ccd_output.success and smiles_output.success
    ccd_structure = ccd_output.structures[0]
    smiles_structure = smiles_output.structures[0]

    # Heavy-atom count must match exactly — the same molecule is being predicted
    # either way, so any difference indicates a parsing error.
    ccd_atoms = int((ccd_structure._get_atom_array().element != "H").sum())
    smiles_atoms = int((smiles_structure._get_atom_array().element != "H").sum())
    assert ccd_atoms == smiles_atoms, (
        f"Heavy-atom counts diverge: CCD={ccd_atoms}, SMILES={smiles_atoms}. "
        "SMILES input may be parsing the ligand differently than CCD."
    )

    # Confidence shouldn't collapse on the SMILES path. AF3's broken SMILES
    # predictions had plddt scattered to ~30; a sensible prediction is >50.
    ccd_plddt = ccd_structure.metrics["avg_plddt"]
    smiles_plddt = smiles_structure.metrics["avg_plddt"]
    assert smiles_plddt > 0.5, f"SMILES-input plddt={smiles_plddt:.2f} suggests broken structure"
    assert abs(ccd_plddt - smiles_plddt) < 0.2, (
        f"plddt diverges: CCD={ccd_plddt:.2f}, SMILES={smiles_plddt:.2f}. "
        "Large gap suggests one path is producing a low-quality structure."
    )


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


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_protenix_holds_runner_across_calls():
    """Two calls inside persist_tool reuse the held InferenceRunner (no rebuild)."""
    complexes = [StructurePredictionComplex(chains=[Chain(sequence=_CRO_SEQUENCE, entity_type="protein")])]
    inputs = ProtenixInput(complexes=complexes)
    config = ProtenixConfig(
        model_name="protenix_tiny_default_v0.5.0",
        use_msa=False,
        num_diffusion_samples=1,
        num_diffusion_steps=20,
        seeds=[0],
        verbose=False,
    )

    introspect_payload = {
        "operation": "introspect_loaded",
        "model_name": config.model_name,
    }

    with ToolInstance.persist_tool("protenix"):
        run_protenix(inputs, config)
        info1 = ToolInstance.dispatch("protenix", introspect_payload, config=config)

        run_protenix(inputs, config)
        info2 = ToolInstance.dispatch("protenix", introspect_payload, config=config)

    assert info1["loaded"] and info2["loaded"]
    assert info1["runner_id"] is not None
    # Same Python object across calls — proves the runner wasn't rebuilt.
    assert info1["runner_id"] == info2["runner_id"]
    assert info1["cache_key"] == info2["cache_key"]


# ── PAE matrix surfacing ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_protenix_pae_surface():
    """avg_pae always emitted; pae square + self-consistent only when flag set."""
    complexes = [StructurePredictionComplex(chains=[Chain(sequence=_CRO_SEQUENCE, entity_type="protein")])]
    inputs = ProtenixInput(complexes=complexes)
    base = {
        "model_name": "protenix_tiny_default_v0.5.0",
        "use_msa": False,
        "num_diffusion_samples": 1,
        "num_diffusion_steps": 20,
        "seeds": [0],
    }

    off = run_protenix(inputs, ProtenixConfig(**base))
    on = run_protenix(inputs, ProtenixConfig(**base, include_pae_matrix=True))

    off_m, on_m = off.structures[0].metrics, on.structures[0].metrics
    for m in (off_m, on_m):
        assert m["avg_pae"] is not None and m["avg_pae"] >= 0.0
    assert "pae" not in off_m.model_dump(exclude_none=True)

    pae = on_m["pae"]
    n = len(_CRO_SEQUENCE)
    assert pae is not None and len(pae) == n and all(len(row) == n for row in pae)
    # 2-decimal upstream rounding loosens the consistency tolerance vs Boltz2.
    expected = sum(sum(row) for row in pae) / (n * n)
    assert on_m["avg_pae"] == pytest.approx(expected, abs=1e-2)
    assert_metrics_in_spec(on)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("protenix-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_protenix_benchmark(request):
    """Benchmark protenix-prediction on the MfnG protein + L-tyrosine ligand (cold + warm).

    Single ~390-residue protein-ligand complex without MSA, run with the default
    base model. Cold pass measures weight load + first inference; warm pass
    measures inference only.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = ProtenixInput(complexes=[complex_])
    config = ProtenixConfig(use_msa=False, verbose=True)

    result = benchmark_twice(request, "protenix", lambda: run_protenix(inputs=inputs, config=config))

    assert result.success, "Protenix benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
