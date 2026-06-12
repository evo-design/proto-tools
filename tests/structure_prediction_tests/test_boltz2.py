"""tests/structure_prediction_tests/test_boltz2.py.

Tests for Boltz2 (both ``boltz2-prediction`` and ``boltz2-affinity``).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import SingleChainSelection, is_valid_structure
from proto_tools.tools.structure_prediction import (
    Boltz2AffinityConfig,
    Boltz2AffinityInput,
    Boltz2Config,
    Boltz2Input,
    Chain,
    Complex,
    ComplexMSAs,
    run_boltz2,
    run_boltz2_affinity,
)
from proto_tools.tools.structure_prediction.boltz2.boltz2 import run_boltz2_on_complex
from proto_tools.tools.structure_prediction.boltz2.helpers import build_chain_msa_paths, complex_to_yaml
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

# Cro repressor from bacteriophage lambda. Short, well-folded test protein.
_CRO_SEQUENCE = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA"
# L-tyrosine SMILES; resolves to CCD "TYR".
_TYR_SMILES = "c1cc(ccc1C[C@@H](C(=O)O)N)O"


# ── Ligand YAML shape: CCD-prefer dispatch ─────────────────────────────────


def _boltz2_ligand_entries(chains):
    """Build a Boltz2 YAML payload from ``chains`` and return its ligand entries (parsed)."""
    parsed = yaml.safe_load(complex_to_yaml(chains))
    return [entry["ligand"] for entry in parsed["sequences"] if "ligand" in entry]


def test_boltz2_ligand_uses_ccd_code_when_available():
    """Fragment with a resolved ccd_code serializes to ``ccd: <code>``, not raw SMILES."""
    atp = Fragment(ccd_code="ATP")
    assert atp.ccd_code == "ATP"  # invariant guard

    [ligand_entry] = _boltz2_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), atp])
    assert ligand_entry == {"id": "B", "ccd": "ATP"}


def test_boltz2_ligand_falls_back_to_smiles_when_no_ccd_match():
    """Novel ligand (SMILES with no wwPDB CCD entry) serializes as raw SMILES."""
    # Synthetic perfluorinated terphenyl chain — not in the wwPDB CCD database.
    novel_smiles = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    novel = Fragment(smiles=novel_smiles)
    assert novel.ccd_code is None  # invariant guard

    [ligand_entry] = _boltz2_ligand_entries([Chain(sequence="MKTLPGCDA", entity_type="protein"), novel])
    assert ligand_entry == {"id": "B", "smiles": novel.smiles}


# ── MSA file assignment: identical chains share one MSA ──────────────────────


class _StopAfterDispatch(Exception):
    """Short-circuit run_boltz2_on_complex once the input YAML + MSAs are written."""


@pytest.mark.parametrize(
    ("seqs", "n_files"),
    [
        pytest.param((_CRO_SEQUENCE, _CRO_SEQUENCE), 1, id="homodimer_shares_one_msa"),
        pytest.param((_CRO_SEQUENCE, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"), 2, id="heterodimer_keeps_distinct"),
    ],
)
def test_boltz2_writes_one_msa_per_unique_sequence(seqs, n_files):
    """Identical chains share one MSA file (boltz rejects per-chain MSAs); distinct sequences stay separate."""
    import yaml

    complex_ = Complex(chains=[Chain(sequence=s, entity_type="protein") for s in seqs])
    complex_msas = ComplexMSAs(per_chain={ch_idx: MSA(aligned_sequences=[s, s]) for ch_idx, s in enumerate(seqs)})
    captured: dict = {}

    def fake_dispatch(_name, input_data, **_kwargs):
        yaml_path = Path(input_data["input_yaml_path"])
        captured["yaml"] = yaml.safe_load(yaml_path.read_text())
        captured["csv_files"] = sorted(p.name for p in (yaml_path.parent / "msas").iterdir())
        raise _StopAfterDispatch

    with (
        patch(
            "proto_tools.tools.structure_prediction.boltz2.boltz2.ToolInstance.dispatch",
            side_effect=fake_dispatch,
        ),
        pytest.raises(_StopAfterDispatch),
    ):
        run_boltz2_on_complex(Boltz2Config(use_msa=True), complex_, complex_msas=complex_msas)

    msa_paths = [e["protein"]["msa"] for e in captured["yaml"]["sequences"] if "protein" in e]
    assert len(captured["csv_files"]) == n_files
    assert len(set(msa_paths)) == n_files
    assert "empty" not in msa_paths


def test_build_chain_msa_paths_appends_deep_unpaired_with_key_minus_one(tmp_path):
    """Paired rows keep row-index keys; deep unpaired rows append with key=-1 (depth, no pairing).

    Boltz excludes ``key == -1`` rows from taxonomy pairing but keeps them as MSA depth.
    """
    import csv as _csv

    query_a, query_b = "MKTAYIAKQR", "GSHMEELLSK"
    cx = Complex(
        chains=[
            Chain(id="A", sequence=query_a, entity_type="protein"),
            Chain(id="B", sequence=query_b, entity_type="protein"),
        ]
    )
    paired = {
        0: MSA(aligned_sequences=[query_a, "MKTAYIAKQA", "MKTAYIAKQE"]),  # query + 2 paired
        1: MSA(aligned_sequences=[query_b, "GSHMEELLSA", "GSHMEELLSE"]),
    }
    unpaired = {
        0: MSA(aligned_sequences=[query_a, "MKTAYIAKQA", "MKTAYIAKQW", "MKTAYIAKQY"]),  # 1 overlap, 2 new
        1: MSA(aligned_sequences=[query_b, "GSHMEELLSW"]),  # 1 new
    }
    complex_msas = ComplexMSAs(per_chain=paired, paired=True, unpaired_per_chain=unpaired)

    paths = build_chain_msa_paths(cx, complex_msas, str(tmp_path))

    rows_a = list(_csv.DictReader(Path(paths["A"]).read_text().splitlines()))
    assert [r["key"] for r in rows_a] == ["0", "1", "2", "-1", "-1"]  # 2 paired + 2 new unpaired
    seqs_a = [r["sequence"] for r in rows_a]
    assert "MKTAYIAKQW" in seqs_a and "MKTAYIAKQY" in seqs_a  # new depth present, overlap deduped

    rows_b = list(_csv.DictReader(Path(paths["B"]).read_text().splitlines()))
    assert [r["key"] for r in rows_b] == ["0", "1", "2", "-1"]


# ── Affinity: YAML emission & validator ─────────────────────────────────────


def test_complex_to_yaml_emits_affinity_block():
    """A binder chain ID triggers the ``properties: [{affinity: {binder: ...}}]`` block."""
    chains = [Chain(sequence="MKTLPGCDA", entity_type="protein"), Fragment(ccd_code="ATP")]
    parsed = yaml.safe_load(complex_to_yaml(chains, affinity_binder_chain_id="B"))
    assert parsed["properties"] == [{"affinity": {"binder": "B"}}]


def test_complex_to_yaml_omits_affinity_block_by_default():
    """Without a binder chain ID, the YAML has no ``properties`` key."""
    chains = [Chain(sequence="MKTLPGCDA", entity_type="protein"), Fragment(ccd_code="ATP")]
    parsed = yaml.safe_load(complex_to_yaml(chains))
    assert "properties" not in parsed


def test_affinity_auto_detects_single_ligand_binder():
    """A complex with exactly one ligand resolves the binder automatically."""
    inputs = Boltz2AffinityInput(complexes=[[_CRO_SEQUENCE, _TYR_SMILES]])
    assert inputs.resolved_binder_chain_ids == ["B"]


def test_affinity_explicit_binder_chain_selects_named_ligand():
    """A single explicit ``binder_chain`` naming a ligand broadcasts to every complex."""
    inputs = Boltz2AffinityInput(complexes=[[_CRO_SEQUENCE, _TYR_SMILES]], binder_chain=["B"])
    assert inputs.resolved_binder_chain_ids == ["B"]


def test_affinity_binder_chain_broadcasts_to_all_complexes():
    """A length-1 ``binder_chain`` is broadcast across multiple complexes (one → N)."""
    inputs = Boltz2AffinityInput(
        complexes=[[_CRO_SEQUENCE, _TYR_SMILES], [_CRO_SEQUENCE, _TYR_SMILES]], binder_chain=["B"]
    )
    assert inputs.binder_chain == [SingleChainSelection(chain="B"), SingleChainSelection(chain="B")]
    assert inputs.resolved_binder_chain_ids == ["B", "B"]


def test_affinity_binder_chain_per_complex_one_to_one():
    """An N-length ``binder_chain`` is matched 1:1 with the complexes (no broadcast)."""
    inputs = Boltz2AffinityInput(
        complexes=[[_CRO_SEQUENCE, _TYR_SMILES], [_CRO_SEQUENCE, _TYR_SMILES]], binder_chain=["B", None]
    )
    assert inputs.resolved_binder_chain_ids == ["B", "B"]


def test_affinity_binder_chain_length_mismatch_raises():
    """A ``binder_chain`` length that is neither 1 nor N is rejected at construction."""
    with pytest.raises(ValidationError, match="expected 1"):
        Boltz2AffinityInput(
            complexes=[[_CRO_SEQUENCE, _TYR_SMILES], [_CRO_SEQUENCE, _TYR_SMILES]],
            binder_chain=["B", "B", "B"],
        )


def test_affinity_binder_ids_track_model_copy_partition():
    """resolved_binder_chain_ids re-resolves under model_copy so ToolPool partitions stay aligned."""
    inputs = Boltz2AffinityInput(complexes=[[_CRO_SEQUENCE, _TYR_SMILES], [_CRO_SEQUENCE, _TYR_SMILES]])
    partition = inputs.model_copy(update={"complexes": [inputs.complexes[0]]})
    assert partition.resolved_binder_chain_ids == ["B"]


def _two_ligand_complex():
    return Complex(
        chains=[
            Chain(sequence=_CRO_SEQUENCE, entity_type="protein"),
            Fragment(ccd_code="ATP"),
            Fragment(ccd_code="MG"),
        ]
    )


def _ligand_only_complex():
    return Complex(chains=[Fragment(ccd_code="ATP")])


def _oversized_ligand_complex():
    return Complex(chains=[Chain(sequence=_CRO_SEQUENCE, entity_type="protein"), Fragment(smiles="C" + "C" * 150)])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"complexes": [[_CRO_SEQUENCE, _TYR_SMILES]], "binder_chain": ["A"]}, r"not a ligand chain"),
        ({"complexes": [[_CRO_SEQUENCE, _TYR_SMILES]], "binder_chain": ["Z"]}, r"not a chain in complex"),
        ({"complexes": [[_CRO_SEQUENCE]]}, r"exactly one"),
        ({"complexes": [_two_ligand_complex()]}, r"exactly one"),
        ({"complexes": [_ligand_only_complex()]}, r"at least one protein chain"),
        ({"complexes": [_oversized_ligand_complex()]}, r"heavy atoms"),
    ],
    ids=[
        "non-ligand binder",
        "unknown chain ID",
        "zero ligands",
        "multiple ligands without explicit binder",
        "no protein target",
        "ligand exceeds 128 heavy atoms",
    ],
)
def test_affinity_validator_rejects_invalid_input(kwargs, match):
    """The affinity validator rejects each upstream-enforced constraint with a clear error."""
    with pytest.raises(ValueError, match=match):
        Boltz2AffinityInput(**kwargs)


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
    # L-tyrosine — known CCD entry "TYR", small enough for fast inference
    tyr_smiles = "c1cc(ccc1C[C@@H](C(=O)O)N)O"
    protein = Chain(sequence=_CRO_SEQUENCE, entity_type="protein")

    # CCD path: validator auto-resolves ccd_code="TYR"; implementation sends ccd: TYR
    ccd_frag = Fragment(smiles=tyr_smiles)
    assert ccd_frag.ccd_code == "TYR"  # invariant guard
    ccd_complex = Complex(chains=[protein, ccd_frag])

    # SMILES path: force ccd_code=None so the implementation falls back to raw SMILES
    smiles_frag = Fragment(smiles=tyr_smiles)
    smiles_frag.ccd_code = None
    smiles_complex = Complex(chains=[protein, smiles_frag])

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


# ── Affinity: GPU smoke ──────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_boltz2_affinity_end_to_end():
    """End-to-end smoke: affinity scores are well-formed and the CIF is valid."""
    inputs = Boltz2AffinityInput(complexes=[[_CRO_SEQUENCE, _TYR_SMILES]])
    config = Boltz2AffinityConfig(
        use_msa=False,
        sampling_steps=50,
        diffusion_samples=1,
        sampling_steps_affinity=50,
        diffusion_samples_affinity=2,
        seed=42,
    )

    result = run_boltz2_affinity(inputs, config)

    assert result.success
    structure = result.structures[0]
    assert is_valid_structure(structure.structure_cif)
    assert isinstance(structure.metrics["affinity_pred_value"], float)
    assert 0.0 <= structure.metrics["affinity_probability_binary"] <= 1.0
    assert_metrics_in_spec(result)


# ── Affinity: Benchmark ──────────────────────────────────────────────────────


@pytest.mark.benchmark("boltz2-affinity")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_boltz2_affinity_benchmark(request):
    """Benchmark boltz2-affinity: virtual-screen 8 diverse small molecules against Cro repressor, MSA-free (cold + warm)."""
    ligands = [
        _TYR_SMILES,  # L-tyrosine
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
        "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "O=C(O)c1ccccc1O",  # salicylic acid
        "O=C(O)c1cccnc1",  # nicotinic acid
        "N[C@@H](Cc1ccccc1)C(=O)O",  # L-phenylalanine
    ]
    inputs = Boltz2AffinityInput(complexes=[[_CRO_SEQUENCE, smiles] for smiles in ligands])
    config = Boltz2AffinityConfig(
        use_msa=False,
        sampling_steps=50,
        diffusion_samples=1,
        sampling_steps_affinity=50,
        diffusion_samples_affinity=2,
        seed=42,
        verbose=True,
    )

    result = benchmark_twice(request, "boltz2", lambda: run_boltz2_affinity(inputs, config))

    assert result.success, "Boltz2 affinity benchmark run failed"
    assert result.tool_id == "boltz2-affinity"
    assert len(result.structures) == len(ligands)
    for structure in result.structures:
        assert is_valid_structure(structure.structure_cif)
        assert isinstance(structure.metrics["affinity_pred_value"], float)
        assert 0.0 <= structure.metrics["affinity_probability_binary"] <= 1.0
    assert_metrics_in_spec(result)
