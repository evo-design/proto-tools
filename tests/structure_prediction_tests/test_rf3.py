"""tests/structure_prediction_tests/test_rf3.py.

Tests for RoseTTAFold3 (``rf3-prediction``).
"""

import json
from pathlib import Path

import pytest

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools.structure_prediction import (
    Chain,
    Complex,
    RF3Config,
    RF3Input,
    run_rf3_prediction,
)
from proto_tools.tools.structure_prediction.rf3.helpers import complex_to_rf3_json
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

# Cro repressor from bacteriophage lambda; short, well-folded.
_CRO_SEQUENCE = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA"
# L-tyrosine; resolves to CCD "TYR".
_TYR_SMILES = "c1cc(ccc1C[C@@H](C(=O)O)N)O"


# ── Helpers serialization ─────────────────────────────────────────────────────


def test_complex_to_rf3_json_protein_emits_seq_and_chain_id():
    """A bare protein chain renders as ``{"seq": ..., "chain_id": ...}``."""
    payload = json.loads(complex_to_rf3_json([Chain(sequence="MKTL")]))
    [example] = payload
    assert example["name"] == "complex"
    assert example["components"] == [{"seq": "MKTL", "chain_id": "A"}]


def test_complex_to_rf3_json_ligand_prefers_ccd_code():
    """Fragment with a resolved ``ccd_code`` serializes as ``{"ccd_code": ...}``, not SMILES."""
    components = json.loads(complex_to_rf3_json([Chain(sequence="MKTL"), Fragment(ccd_code="ATP")]))[0]["components"]
    assert components[1] == {"ccd_code": "ATP"}


def test_complex_to_rf3_json_ligand_falls_back_to_smiles():
    """A SMILES-only Fragment (no CCD match) serializes as ``{"smiles": ...}``."""
    # Synthetic perfluorinated chain — not in wwPDB CCD.
    novel = Fragment(smiles="FC(F)(F)C(F)(F)C(F)(F)c1ccccc1")
    assert novel.ccd_code is None  # invariant guard
    components = json.loads(complex_to_rf3_json([Chain(sequence="MKTL"), novel]))[0]["components"]
    assert components[1] == {"smiles": novel.smiles}


def test_complex_to_rf3_json_includes_msa_paths_when_provided():
    """A chain with an MSA path picks up ``msa_path`` in its component dict."""
    components = json.loads(
        complex_to_rf3_json(
            [Chain(id="A", sequence="MKTL")],
            chain_msa_paths={"A": "/path/to/A.a3m"},
        )
    )[0]["components"]
    assert components[0]["msa_path"] == "/path/to/A.a3m"


def test_complex_to_rf3_json_omits_cyclic_chains_field():
    """``cyclic_chains`` is intentionally not in the JSON wrapper — upstream ignores it there.

    ``rf3.data.InferenceInput.from_json_dict`` does not read ``cyclic_chains`` from
    the JSON; cyclization is applied via the Hydra CLI override
    ``cyclic_chains=[A,B]``. This test guards against accidentally re-adding the
    silently-ignored JSON field.
    """
    example = json.loads(complex_to_rf3_json([Chain(id="A", sequence="MKTL")]))[0]
    assert "cyclic_chains" not in example
    # Only ``name`` and ``components`` are emitted at the example level.
    assert set(example.keys()) == {"name", "components"}


def test_build_chain_a3m_paths_feeds_deep_unpaired_msa(tmp_path):
    """Each chain's a3m carries the deep per-chain unpaired MSA, not the shallow paired set.

    RF3 re-pairs by tax_id parsed from the a3m headers, so the deep unpaired MSA
    (full per-chain depth, UniRef ``TaxID=`` headers preserved) is the correct feed.
    """
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.rf3.helpers import build_chain_a3m_paths
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    seq_a, seq_b = "MKTAYIAKQR", "GSHMEELLSK"
    cx = Complex(
        chains=[
            Chain(id="A", sequence=seq_a, entity_type="protein"),
            Chain(id="B", sequence=seq_b, entity_type="protein"),
        ]
    )
    paired = {i: MSA(aligned_sequences=[s, s]) for i, s in enumerate([seq_a, seq_b])}  # 2 rows
    unpaired = {i: MSA(aligned_sequences=[s, s, s, s, s]) for i, s in enumerate([seq_a, seq_b])}  # 5 rows
    complex_msas = ComplexMSAs(per_chain=paired, paired=True, unpaired_per_chain=unpaired)

    paths = build_chain_a3m_paths(cx, complex_msas, str(tmp_path))

    for chain_id in ("A", "B"):
        rows = Path(paths[chain_id]).read_text().count(">")
        assert rows == 5, f"chain {chain_id} a3m should carry the deep unpaired MSA, got {rows} rows"


def test_build_chain_a3m_paths_falls_back_to_per_chain_when_no_unpaired(tmp_path):
    """With no separate unpaired MSA, the per-chain (primary) MSA is written as before."""
    from proto_tools.entities.msa import MSA
    from proto_tools.tools.structure_prediction.rf3.helpers import build_chain_a3m_paths
    from proto_tools.tools.structure_prediction.shared_data_models import ComplexMSAs

    seq = "MKTAYIAKQR"
    cx = Complex(chains=[Chain(id="A", sequence=seq, entity_type="protein")])
    complex_msas = ComplexMSAs(per_chain={0: MSA(aligned_sequences=[seq, seq, seq])}, paired=False)

    paths = build_chain_a3m_paths(cx, complex_msas, str(tmp_path))

    assert Path(paths["A"]).read_text().count(">") == 3


# ── Config: RF3 emits no per-token PAE matrix ─────────────────────────────────


def test_rf3_config_rejects_include_pae_matrix():
    """RF3 emits only chain-pair PAE aggregates; the inherited toggle must be rejected."""
    with pytest.raises(ValueError, match="include_pae_matrix"):
        RF3Config(include_pae_matrix=True)


# ── Live GPU dispatch ─────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_rf3_basic_execution():
    """Single-chain prediction populates the headline metrics and a valid mmCIF."""
    result = run_rf3_prediction(
        RF3Input(complexes=[_CRO_SEQUENCE]),
        RF3Config(use_msa=False, diffusion_batch_size=1, verbose=False),
    )
    assert result.success
    assert result.tool_id == "rf3-prediction"
    assert len(result.structures) == 1
    s = result.structures[0]
    assert is_valid_structure(s.structure_cif)
    m = s.metrics
    assert 0.0 <= m.avg_plddt <= 1.0
    assert 0.0 <= m.ptm <= 1.0
    assert m.avg_pae >= 0.0
    assert m.pde >= 0.0
    assert len(m.chain_ptm) == 1
    # Single-chain inputs get empty chain-pair matrices (no pair to compute).
    assert m.chain_pair_pae == []
    assert m.chain_pair_pde == []
    assert isinstance(m.has_clash, bool)
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_rf3_multi_chain_populates_chain_pair_matrices():
    """A 2-chain input fills the (symmetric) 2-by-2 chain-pair PAE/PDE matrices."""
    comp = Complex(
        chains=[
            Chain(id="A", sequence="MKTLILALSLVLAFSSATAA"),
            Chain(id="B", sequence="GGGGSGGGGSGGGGS"),
        ]
    )
    result = run_rf3_prediction(
        RF3Input(complexes=[comp]),
        RF3Config(use_msa=False, diffusion_batch_size=1, verbose=False),
    )
    assert result.success
    m = result.structures[0].metrics
    assert len(m.chain_ptm) == 2
    assert m.iptm is not None
    assert len(m.chain_pair_pae) == 2 and len(m.chain_pair_pae[0]) == 2
    assert m.chain_pair_pae[0][0] == 0.0 and m.chain_pair_pae[1][1] == 0.0
    assert m.chain_pair_pae[0][1] == m.chain_pair_pae[1][0] >= 0.0
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_rf3_with_ligand_smiles():
    """A protein + SMILES ligand complex folds and reports has_clash as a bool."""
    comp = Complex(chains=[Chain(sequence=_CRO_SEQUENCE), Fragment(smiles=_TYR_SMILES)])
    result = run_rf3_prediction(
        RF3Input(complexes=[comp]),
        RF3Config(use_msa=False, diffusion_batch_size=1, verbose=False),
    )
    assert result.success
    s = result.structures[0]
    assert isinstance(s.metrics.has_clash, bool)
    assert is_valid_structure(s.structure_cif)


@pytest.mark.uses_gpu
def test_rf3_seed_advances_per_complex():
    """Duplicate inputs in one call diversify because the wrapper advances ``seed + i``."""
    result = run_rf3_prediction(
        RF3Input(complexes=[_CRO_SEQUENCE, _CRO_SEQUENCE]),
        RF3Config(use_msa=False, diffusion_batch_size=1, seed=42, verbose=False),
    )
    assert result.success
    assert len(result.structures) == 2
    # The two structures must differ — confirms per-complex seed advancement.
    assert result.structures[0].structure != result.structures[1].structure
