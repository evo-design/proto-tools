"""tests/structure_prediction_tests/test_af3.py.

Tests for AlphaFold3.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from proto_tools.entities.ligands import Fragment, Ligands
from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    AlphaFold3Config,
    AlphaFold3Input,
    Chain,
    Complex,
    run_alphafold3,
)
from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs
from proto_tools.utils.standalone_helpers_source.standalone_helpers import resolve_weights_dir
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec


def _alphafold3_weights_skip_reason() -> str | None:
    """Return a skip reason if AlphaFold3 weights are missing, else None.

    AlphaFold3 weights are gated under DeepMind's ToU and must be obtained
    separately. See ``proto_tools/tools/structure_prediction/alphafold3/README.md``.
    """
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


# ── Module-level constants ────────────────────────────────────────────────────

_EPINEPHRINE_SMILES = "CNC[C@@H](c1ccc(c(c1)O)O)O"  # L-epinephrine → ALE
_ATP_SMILES = "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N"  # ATP → ATP


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_af3_inference(tmp_path):
    """Patch ToolInstance.dispatch to capture the input JSON written to disk.

    The mock runs synchronously inside run_alphafold3's tempfile.TemporaryDirectory
    context, so input_json_path is still live when mock_dispatch reads it.
    """
    dummy_pdb_file = tmp_path / "dummy.pdb"
    dummy_pdb_file.write_text(
        "HEADER    DUMMY PDB\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.95           C\nEND\n"
    )
    dummy_pdb_path = str(dummy_pdb_file)
    mock_metrics = {"avg_plddt": 0.95, "ptm": 0.8}

    captured_data = {}

    def mock_dispatch(toolkit, input_data, **kwargs):
        with open(input_data["input_json_path"]) as f:
            captured_data["input_json"] = json.load(f)
        captured_data["input_data"] = input_data
        return {
            "structure_pdb": dummy_pdb_path,
            "metrics": mock_metrics,
        }

    with patch("proto_tools.tools.structure_prediction.alphafold3.alphafold3.ToolInstance") as mock_ti:
        mock_ti.dispatch = mock_dispatch
        yield captured_data


# ── JSON structure tests ──────────────────────────────────────────────────────


def test_af3_ligand_and_nucleic_acids(mock_af3_inference):
    """DNA, RNA, and ligands are correctly formatted in the AF3 JSON dialect."""
    chains = [
        Chain(sequence="MVLSPADKTN", entity_type="protein"),
        Chain(sequence="ACGT", entity_type="dna"),
        Chain(sequence="CCO", entity_type="ligand"),
    ]
    complexes = [Complex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_entities", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 3

    assert sequences[0]["protein"]["id"] == "A"
    assert sequences[0]["protein"]["sequence"] == "MVLSPADKTN"

    assert sequences[1]["dna"]["id"] == "B"
    assert sequences[1]["dna"]["sequence"] == "ACGT"

    assert sequences[2]["ligand"]["id"] == "C"
    assert sequences[2]["ligand"]["ccdCodes"] == ["EOH"]


def test_af3_common_seed_overrides_model_seeds(mock_af3_inference):
    """The common BaseConfig.seed drives AlphaFold3 modelSeeds when supplied."""
    complexes = [Complex(chains=[Chain(sequence="MVLSPADKTN", entity_type="protein")])]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_seed", use_msa=False, seed=123, seeds=[0, 1])

    result = run_alphafold3(inputs, config)
    assert result.success
    assert mock_af3_inference["input_json"]["modelSeeds"] == [123]


def test_af3_rejects_msa_keyed_to_wrong_chain(mock_af3_inference):
    """Precomputed MSAs must have the matching chain sequence in their first row."""
    expected = "MVLSPADKTN"
    wrong = "MKTAYIAKQR"
    complexes = [Complex(chains=[Chain(sequence=expected, entity_type="protein")])]
    inputs = AlphaFold3Input(
        complexes=complexes,
        msas=[ComplexMSAs(per_chain={0: MSA(aligned_sequences=[wrong, wrong])})],
    )
    config = AlphaFold3Config(name="test_msa_guard", use_msa=False)

    with pytest.raises(ValueError, match="has query row"):
        run_alphafold3(inputs, config)


def test_af3_unpaired_msa_path_gets_deep_per_chain_unpaired(tmp_path):
    """The unpaired slot receives each chain's deep unpaired MSA; the paired slot the shallow paired set."""
    from proto_tools.entities.complex import Chain, Complex
    from proto_tools.tools.structure_prediction.alphafold3.alphafold3 import _assign_msas_to_input_json

    seq_a, seq_b = "MKTAYIAKQR", "GSHMEELLSK"
    cx = Complex(chains=[Chain(sequence=seq_a, entity_type="protein"), Chain(sequence=seq_b, entity_type="protein")])
    paired = {i: MSA(aligned_sequences=[s, s]) for i, s in enumerate([seq_a, seq_b])}  # 2 rows
    unpaired = {i: MSA(aligned_sequences=[s, s, s, s, s]) for i, s in enumerate([seq_a, seq_b])}  # 5 rows
    input_json = {"sequences": [{"protein": {"id": "A"}}, {"protein": {"id": "B"}}]}

    out = _assign_msas_to_input_json(input_json, paired, unpaired, True, cx, str(tmp_path), verbose=0)

    for entry in out["sequences"]:
        prot = entry["protein"]
        unpaired_rows = (tmp_path / prot["unpairedMsaPath"]).read_text().count(">")
        paired_rows = (tmp_path / prot["pairedMsaPath"]).read_text().count(">")
        assert unpaired_rows == 5, f"unpairedMsaPath should carry the deep unpaired MSA, got {unpaired_rows} rows"
        assert paired_rows == 2, f"pairedMsaPath should carry the paired set, got {paired_rows} rows"


def test_af3_unpaired_falls_back_to_paired_when_no_separate_unpaired(tmp_path):
    """With no separate unpaired MSA, unpairedMsaPath falls back to the primary MSA (prior behavior)."""
    from proto_tools.entities.complex import Chain, Complex
    from proto_tools.tools.structure_prediction.alphafold3.alphafold3 import _assign_msas_to_input_json

    seq_a, seq_b = "MKTAYIAKQR", "GSHMEELLSK"
    cx = Complex(chains=[Chain(sequence=seq_a, entity_type="protein"), Chain(sequence=seq_b, entity_type="protein")])
    paired = {i: MSA(aligned_sequences=[s, s, s]) for i, s in enumerate([seq_a, seq_b])}  # 3 rows
    input_json = {"sequences": [{"protein": {"id": "A"}}, {"protein": {"id": "B"}}]}

    out = _assign_msas_to_input_json(input_json, paired, None, True, cx, str(tmp_path), verbose=0)

    for entry in out["sequences"]:
        prot = entry["protein"]
        unpaired_rows = (tmp_path / prot["unpairedMsaPath"]).read_text().count(">")
        paired_rows = (tmp_path / prot["pairedMsaPath"]).read_text().count(">")
        assert unpaired_rows == 3 and paired_rows == 3  # both slots from the same set


def test_af3_rna_entity(mock_af3_inference):
    """RNA chains are correctly formatted in the AF3 JSON dialect."""
    chains = [
        Chain(sequence="AUGC", entity_type="rna"),
    ]
    complexes = [Complex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_rna", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 1
    assert sequences[0]["rna"]["id"] == "A"
    assert sequences[0]["rna"]["sequence"] == "AUGC"


def test_af3_protein_modifications_in_json(mock_af3_inference):
    """Protein PTMs are serialised as ptmType/ptmPosition in the AF3 JSON."""
    chains = [
        Chain(
            sequence="MVLSPADKTN",
            entity_type="protein",
            modifications=[(4, "SEP")],  # position 4 is 'S' (serine)
        ),
    ]
    complexes = [Complex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_ptm", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    protein_entry = mock_af3_inference["input_json"]["sequences"][0]["protein"]
    assert "modifications" in protein_entry
    assert len(protein_entry["modifications"]) == 1
    mod = protein_entry["modifications"][0]
    assert mod["ptmType"] == "SEP"
    assert mod["ptmPosition"] == 4


def test_af3_nucleic_acid_modifications_in_json(mock_af3_inference):
    """RNA modifications are serialised as modificationType/basePosition in the AF3 JSON."""
    chains = [
        Chain(
            sequence="AUGCAUGC",
            entity_type="rna",
            modifications=[(3, "2MG")],  # position 3 is 'G' (guanosine)
        ),
    ]
    complexes = [Complex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_rna_mod", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    rna_entry = mock_af3_inference["input_json"]["sequences"][0]["rna"]
    assert "modifications" in rna_entry
    assert len(rna_entry["modifications"]) == 1
    mod = rna_entry["modifications"][0]
    assert mod["modificationType"] == "2MG"
    assert mod["basePosition"] == 3


# ── SMILES-to-CCD conversion tests ───────────────────────────────────────────


def test_af3_ligand_smile_to_ccd_conversion(mock_af3_inference):
    """Known SMILES strings are automatically mapped to their CCD codes."""
    chains = [
        Chain(sequence=_EPINEPHRINE_SMILES, entity_type="ligand"),
        Chain(sequence=_ATP_SMILES, entity_type="ligand"),
    ]
    complexes = [Complex(chains=chains)]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_entities", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 2

    assert sequences[0]["ligand"]["id"] == "A"
    assert sequences[0]["ligand"]["ccdCodes"] == ["ALE"]

    assert sequences[1]["ligand"]["id"] == "B"
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]


def test_af3_accepts_fragment_directly(mock_af3_inference):
    """Fragment objects in chains list flow through AF3 with their CCD code."""
    complexes = [
        Complex(
            chains=["MVLSPADKTN", Fragment(ccd_code="ATP")],
        )
    ]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_fragment", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 2
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]


def test_af3_accepts_ligands_collection_expanded(mock_af3_inference):
    """A Ligands collection in chains list expands to one AF3 ligand entry per fragment."""
    complexes = [
        Complex(
            chains=["MVLSPADKTN", Ligands(ccd_codes=["ATP", "MG"])],
        )
    ]
    inputs = AlphaFold3Input(complexes=complexes)
    config = AlphaFold3Config(name="test_ligands_collection", use_msa=False)

    result = run_alphafold3(inputs, config)
    assert result.success

    sequences = mock_af3_inference["input_json"]["sequences"]
    assert len(sequences) == 3
    assert sequences[1]["ligand"]["ccdCodes"] == ["ATP"]
    assert sequences[2]["ligand"]["ccdCodes"] == ["MG"]


# We enforce CCD-only ligand input even though AF3 accepts SMILES: AF3+SMILES empirically scatters
# heavy atoms across the box. Trade-off: novel ligands without a CCD entry are unsupported here.


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("alphafold3-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
@pytest.mark.skipif(_alphafold3_weights_skip_reason() is not None, reason=_alphafold3_weights_skip_reason() or "")
def test_alphafold3_benchmark(request):
    """Benchmark alphafold3-prediction on the MfnG protein + L-tyrosine ligand (cold + warm).

    Single ~390-residue protein-ligand complex without MSA — a representative
    AF3 workload. Cold pass measures weight load + first inference; warm pass
    measures inference only.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = AlphaFold3Input(complexes=[complex_])
    config = AlphaFold3Config(use_msa=False, verbose=True)

    result = benchmark_twice(request, "alphafold3", lambda: run_alphafold3(inputs=inputs, config=config))

    assert result.success, "AlphaFold3 benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
