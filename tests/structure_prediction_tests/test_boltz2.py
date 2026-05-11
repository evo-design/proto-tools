"""tests/structure_prediction_tests/test_boltz2.py.

Tests for Boltz2.
"""

import pytest

from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Boltz2Config,
    Boltz2Input,
    Chain,
    StructurePredictionComplex,
    run_boltz2,
)
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

# Cro repressor from bacteriophage lambda. Short, well-folded test protein.
_CRO_SEQUENCE = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA"


# ── Ligand YAML shape: CCD-prefer dispatch ─────────────────────────────────


def _boltz2_ligand_entries(chains):
    """Build a Boltz2 YAML payload from ``chains`` and return its ligand entries (parsed)."""
    import yaml

    from proto_tools.tools.structure_prediction.boltz2.helpers import complex_to_yaml

    parsed = yaml.safe_load(complex_to_yaml(chains))
    return [entry["ligand"] for entry in parsed["sequences"] if "ligand" in entry]


def test_boltz2_ligand_uses_ccd_code_when_available():
    """Fragment with a resolved ccd_code serializes to ``ccd: <code>``, not raw SMILES."""
    from proto_tools.entities.ligands import Fragment

    atp = Fragment(ccd_code="ATP")
    assert atp.ccd_code == "ATP"  # invariant guard

    [ligand_entry] = _boltz2_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), atp])
    assert ligand_entry == {"id": "B", "ccd": "ATP"}


def test_boltz2_ligand_falls_back_to_smiles_when_no_ccd_match():
    """Novel ligand (SMILES with no wwPDB CCD entry) serializes as raw SMILES."""
    from proto_tools.entities.ligands import Fragment

    # Synthetic perfluorinated terphenyl chain — not in the wwPDB CCD database.
    novel_smiles = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    novel = Fragment(smiles=novel_smiles)
    assert novel.ccd_code is None  # invariant guard

    [ligand_entry] = _boltz2_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), novel])
    assert ligand_entry == {"id": "B", "smiles": novel.smiles}


# ── GPU tests ───────────────────────────────────────────────────────────────


@pytest.mark.extensive
@pytest.mark.uses_gpu
@pytest.mark.slow
def test_boltz2_ccd_vs_smiles_input_equivalent_predictions():
    """Predictions from CCD and SMILES inputs of the same ligand should agree.

    Empirical check for Boltz2's lenient SMILES path. If this test fails,
    consider tightening the implementation to strict CCD-only (the AF3
    implementation does this because AF3 with raw SMILES produces broken
    structures).
    """
    from proto_tools.entities.ligands import Fragment

    # L-tyrosine — known CCD entry "TYR", small enough for fast inference
    tyr_smiles = "c1cc(ccc1C[C@@H](C(=O)O)N)O"
    protein = Chain(sequence=_CRO_SEQUENCE, entity_type="protein")

    # CCD path: validator auto-resolves ccd_code="TYR"; implementation sends ccd: TYR
    ccd_frag = Fragment(smiles=tyr_smiles)
    assert ccd_frag.ccd_code == "TYR"  # invariant guard
    ccd_complex = StructurePredictionComplex(chains=[protein, ccd_frag])

    # SMILES path: force ccd_code=None so the implementation falls back to raw SMILES
    smiles_frag = Fragment(smiles=tyr_smiles)
    smiles_frag.ccd_code = None
    smiles_complex = StructurePredictionComplex(chains=[protein, smiles_frag])

    config = Boltz2Config(
        use_msa=False,
        sampling_steps=50,
        diffusion_samples=1,
        seed=42,
    )

    ccd_output = run_boltz2(Boltz2Input(complexes=[ccd_complex]), config)
    smiles_output = run_boltz2(Boltz2Input(complexes=[smiles_complex]), config)

    assert ccd_output.success and smiles_output.success
    ccd_structure = ccd_output.structures[0]
    smiles_structure = smiles_output.structures[0]

    # Heavy-atom count must match exactly — same molecule, two input formats.
    ccd_atoms = int((ccd_structure._get_atom_array().element != "H").sum())
    smiles_atoms = int((smiles_structure._get_atom_array().element != "H").sum())
    assert ccd_atoms == smiles_atoms, (
        f"Heavy-atom counts diverge: CCD={ccd_atoms}, SMILES={smiles_atoms}. "
        "SMILES input may be parsing the ligand differently than CCD."
    )

    # Confidence shouldn't collapse on the SMILES path.
    ccd_plddt = ccd_structure.metrics["complex_plddt"]
    smiles_plddt = smiles_structure.metrics["complex_plddt"]
    assert smiles_plddt > 0.5, f"SMILES-input plddt={smiles_plddt:.2f} suggests broken structure"
    assert abs(ccd_plddt - smiles_plddt) < 0.2, (
        f"plddt diverges: CCD={ccd_plddt:.2f}, SMILES={smiles_plddt:.2f}. "
        "Large gap suggests one path is producing a low-quality structure."
    )


# ── PAE matrix surfacing ────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_boltz2_pae_surface():
    """avg_pae always emitted; pae square + self-consistent only when flag set."""
    inputs = Boltz2Input(complexes=[_CRO_SEQUENCE])
    base = {"use_msa": False, "sampling_steps": 50, "diffusion_samples": 1, "seed": 42}

    off = run_boltz2(inputs, Boltz2Config(**base))
    on = run_boltz2(inputs, Boltz2Config(**base, include_pae_matrix=True))

    off_m, on_m = off.structures[0].metrics, on.structures[0].metrics
    # avg_pae is "always" availability — must be present in both runs and inside [0, 32).
    for m in (off_m, on_m):
        assert m["avg_pae"] is not None and 0.0 <= m["avg_pae"] < 32.0
    # pae is gated; default-off must not appear in the dump.
    assert "pae" not in off_m.model_dump(exclude_none=True)

    pae = on_m["pae"]
    n = len(_CRO_SEQUENCE)
    assert pae is not None and len(pae) == n and all(len(row) == n for row in pae)
    # avg_pae must equal the matrix mean when both come from the same call.
    expected = sum(sum(row) for row in pae) / (n * n)
    assert on_m["avg_pae"] == pytest.approx(expected, abs=1e-3)
    assert_metrics_in_spec(on)


# ── Benchmark ───────────────────────────────────────────────────────────────


@pytest.mark.benchmark("boltz2-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_boltz2_benchmark(request):
    """Benchmark boltz2-prediction on the MfnG protein + L-tyrosine ligand (cold + warm).

    Single ~390-residue protein-ligand complex without MSA — a representative
    Boltz2 workload. Cold pass measures weight load + first inference; warm
    pass measures inference only.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = Boltz2Input(complexes=[complex_])
    config = Boltz2Config(use_msa=False, verbose=True)

    result = benchmark_twice(request, "boltz2", lambda: run_boltz2(inputs=inputs, config=config))

    assert result.success, "Boltz2 benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
