"""tests/structure_prediction_tests/test_opendde.py.

Tests for OpenDDE all-atom structure prediction (``opendde-prediction``).
"""

import importlib.util
import logging
import re
import sys
import types
from pathlib import Path

import pytest

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools import (
    Chain,
    OpenDDEConfig,
    OpenDDEInput,
    run_opendde,
)
from proto_tools.tools.structure_prediction import Complex, ComplexMSAs
from proto_tools.tools.structure_prediction.opendde.helpers import build_chain_msa_paths, complex_to_opendde_json
from tests._structure_fixtures import synthetic_cif
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_export_output

# Cro repressor from bacteriophage lambda — short, well-folded test protein.
_CRO_SEQUENCE = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA"
# A short, foldable peptide for GPU smoke tests.
_TINY_PEPTIDE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

# OpenDDE pairs paired rows by a species id from the header (_UNIPROT_REGEX); our paired A3M must match it.
_OPENDDE_UNIPROT_REGEX = re.compile(
    r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)


# ── OpenDDE JSON shape: entity mapping ───────────────────────────────────────


def test_opendde_json_top_level_and_protein_entry():
    """The job dict carries name/modelSeeds/sequences and a protein → proteinChain entry."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein")]
    job = complex_to_opendde_json(chains, "myjob", [7])

    assert set(job) == {"name", "modelSeeds", "sequences"}
    assert job["name"] == "myjob"
    assert job["modelSeeds"] == [7]

    [entry] = job["sequences"]
    assert set(entry) == {"proteinChain"}
    protein = entry["proteinChain"]
    assert protein["sequence"] == "MVLSPADKTN"
    assert protein["count"] == 1
    assert protein["id"] == ["A"]
    # No MSA supplied → single-sequence, no unpairedMsaPath key.
    assert "unpairedMsaPath" not in protein


def test_opendde_ligand_uses_ccd_code_when_available():
    """A Fragment with a resolved ccd_code serializes to CCD_-prefixed ``ligand`` (CCD preferred).

    OpenDDE parses a ligand string as SMILES unless it carries the ``CCD_`` prefix,
    so bare CCD codes must be prefixed.
    """
    atp = Fragment(ccd_code="ATP")
    assert atp.ccd_code == "ATP"  # invariant guard

    job = complex_to_opendde_json([Chain(sequence="MKTLPGCDA", entity_type="protein"), atp], "j", [0])
    ligand_entries = [e["ligand"] for e in job["sequences"] if "ligand" in e]
    assert ligand_entries == [{"ligand": "CCD_ATP", "count": 1, "id": ["B"]}]


def test_opendde_ligand_falls_back_to_smiles_when_no_ccd_match():
    """A ligand with no CCD match serializes with its raw SMILES in the ``ligand`` field."""
    # Synthetic perfluorinated terphenyl — not in the wwPDB CCD database.
    novel_smiles = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    novel = Fragment(smiles=novel_smiles)
    assert novel.ccd_code is None  # invariant guard

    job = complex_to_opendde_json([Chain(sequence="MKTLPGCDA", entity_type="protein"), novel], "j", [0])
    [ligand] = [e["ligand"] for e in job["sequences"] if "ligand" in e]
    assert ligand == {"ligand": novel_smiles, "count": 1, "id": ["B"]}


def test_opendde_dna_maps_to_dna_sequence():
    """A DNA chain maps to a ``dnaSequence`` entry carrying the raw nucleotide sequence."""
    chains = [Chain(sequence="ACGTACGT", entity_type="dna")]
    job = complex_to_opendde_json(chains, "j", [0])

    [entry] = job["sequences"]
    assert set(entry) == {"dnaSequence"}
    assert entry["dnaSequence"] == {"sequence": "ACGTACGT", "count": 1, "id": ["A"]}


def test_opendde_chain_msa_paths_set_unpaired_msa_path_on_protein():
    """A supplied chain_msa_paths entry populates ``unpairedMsaPath`` on the protein entry."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein")]
    job = complex_to_opendde_json(chains, "j", [0], chain_msa_paths={"A": "/scratch/chain_A.a3m"})

    assert job["sequences"][0]["proteinChain"]["unpairedMsaPath"] == "/scratch/chain_A.a3m"


def test_opendde_json_emits_paired_msa_path():
    """A supplied chain_paired_msa_paths entry populates ``pairedMsaPath`` alongside the unpaired one."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein")]
    job = complex_to_opendde_json(
        chains,
        "j",
        [0],
        chain_msa_paths={"A": "/scratch/chain_A.a3m"},
        chain_paired_msa_paths={"A": "/scratch/chain_A.paired.a3m"},
    )
    entry = job["sequences"][0]["proteinChain"]
    assert entry["unpairedMsaPath"] == "/scratch/chain_A.a3m"
    assert entry["pairedMsaPath"] == "/scratch/chain_A.paired.a3m"


def test_opendde_build_chain_msa_paths_emits_species_paired_headers(tmp_path):
    """For paired MSAs, a pairedMsaPath is written whose headers OpenDDE's species regex pairs by row."""
    query_a, query_b = "MKTAYIAKQR", "GSHMEELLSK"
    cx = Complex(
        chains=[
            Chain(id="A", sequence=query_a, entity_type="protein"),
            Chain(id="B", sequence=query_b, entity_type="protein"),
        ]
    )
    # Paired set: row i of A and row i of B are the same species (taxonomy-aligned).
    paired = {
        0: MSA(aligned_sequences=[query_a, "MKTAYIAKQA", "MKTAYIAKQE"]),
        1: MSA(aligned_sequences=[query_b, "GSHMEELLSA", "GSHMEELLSE"]),
    }
    complex_msas = ComplexMSAs(per_chain=paired, paired=True)

    unpaired_paths, paired_paths = build_chain_msa_paths(cx, complex_msas, str(tmp_path))

    # Both chains get an unpaired AND a paired A3M.
    assert set(unpaired_paths) == {"A", "B"}
    assert set(paired_paths) == {"A", "B"}

    def _species_by_row(a3m_path):
        headers = [ln[1:] for ln in Path(a3m_path).read_text().splitlines() if ln.startswith(">")]
        # Row 0 is the inert query; non-query rows must match OpenDDE's UniProt regex.
        ids = []
        for h in headers[1:]:
            m = _OPENDDE_UNIPROT_REGEX.match(h)
            assert m is not None, f"header {h!r} does not match OpenDDE's species regex"
            ids.append(m.group("SpeciesId"))
        return ids

    species_a = _species_by_row(paired_paths["A"])
    species_b = _species_by_row(paired_paths["B"])
    # Same species token at each row index across chains -> OpenDDE pairs those rows.
    assert species_a == species_b
    assert len(set(species_a)) == len(species_a)  # distinct per row (row-index encoded)


def test_opendde_protein_modifications_prefixed_ptm():
    """Protein PTMs serialize as CCD_-prefixed ptmType / ptmPosition."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein", modifications=[(4, "SEP")])]
    job = complex_to_opendde_json(chains, "j", [0])

    mods = job["sequences"][0]["proteinChain"]["modifications"]
    assert mods == [{"ptmType": "CCD_SEP", "ptmPosition": 4}]


def test_opendde_dna_modifications_prefixed_base():
    """Nucleic-acid modifications serialize as CCD_-prefixed modificationType / basePosition."""
    chains = [Chain(sequence="ACGT", entity_type="dna", modifications=[(2, "5CM")])]
    job = complex_to_opendde_json(chains, "j", [0])

    mods = job["sequences"][0]["dnaSequence"]["modifications"]
    assert mods == [{"modificationType": "CCD_5CM", "basePosition": 2}]


# ── Checkpoint selection (standalone argv) ───────────────────────────────────


def _load_standalone_inference():
    """Import the standalone ``inference.py`` with a stubbed ``standalone_helpers``."""
    _sh = sys.modules.setdefault("standalone_helpers", types.SimpleNamespace())
    if not hasattr(_sh, "get_logger"):
        _sh.get_logger = logging.getLogger
    standalone_dir = (
        Path(__file__).resolve().parents[2]
        / "proto_tools"
        / "tools"
        / "structure_prediction"
        / "opendde"
        / "standalone"
    )
    spec = importlib.util.spec_from_file_location("_opendde_si_for_tests", standalone_dir / "inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The console script only exists inside the tool venv; argv assembly is what is under test.
    module._resolve_opendde_bin = lambda: "/venv/bin/opendde"
    return module


def _build_cmd(model, **overrides):
    """Call ``OpenDDEModel._build_cmd`` with defaults for the args under test."""
    kwargs = {
        "input_json_path": "input.json",
        "output_dir": "output",
        "job_name": "job",
        "model_checkpoint": "opendde_v1",
        "num_samples": 1,
        "num_steps": 200,
        "num_cycles": 10,
        "use_msa": False,
        "use_template": False,
        "use_rna_msa": False,
        "seed": 0,
        "device": "cuda",
        "include_pae_matrix": False,
    }
    kwargs.update(overrides)
    return model._build_cmd(**kwargs)


def test_opendde_need_atom_confidence_tracks_include_pae_matrix(tmp_path):
    """--need_atom_confidence is 'true' only when the PAE matrix is requested."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True

    off = _build_cmd(model, include_pae_matrix=False)
    on = _build_cmd(model, include_pae_matrix=True)
    assert off[off.index("--need_atom_confidence") + 1] == "false"
    assert on[on.index("--need_atom_confidence") + 1] == "true"


def test_opendde_bundled_abag_routes_via_checkpoint_path(tmp_path):
    """The ``opendde_abag`` name routes to its bundled checkpoint via --load_checkpoint_path (-n stays opendde_v1)."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "opendde_abag.pt").write_bytes(b"")

    cmd = _build_cmd(model, model_checkpoint="opendde_abag")

    assert cmd[cmd.index("-n") + 1] == "opendde_v1"
    assert cmd[cmd.index("--load_checkpoint_path") + 1] == str(ckpt_dir / "opendde_abag.pt")


def test_opendde_default_model_sends_no_checkpoint_override(tmp_path):
    """``opendde_v1`` lets OpenDDE load its own default weights (no --load_checkpoint_path)."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True

    assert "--load_checkpoint_path" not in _build_cmd(model, model_checkpoint="opendde_v1")


def test_opendde_missing_bundled_checkpoint_fails_before_dispatch(tmp_path):
    """A bundled checkpoint that hasn't been downloaded fails here, not inside the CLI."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True
    (tmp_path / "checkpoint").mkdir()

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        _build_cmd(model, model_checkpoint="opendde_abag")


def test_opendde_explicit_checkpoint_path_is_passed_through(tmp_path):
    """A model_checkpoint that isn't a bundled name is used as an explicit path."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True
    custom = tmp_path / "custom.pt"
    custom.write_bytes(b"")

    cmd = _build_cmd(model, model_checkpoint=str(custom))

    assert cmd[cmd.index("--load_checkpoint_path") + 1] == str(custom)


def test_opendde_unknown_checkpoint_value_raises(tmp_path):
    """A value that is neither a bundled name nor an existing path fails clearly (typo guard)."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        _build_cmd(model, model_checkpoint="opendde_v2")


def test_opendde_bundled_checkpoints_are_downloaded_by_setup():
    """Every bundled checkpoint the resolver can select must be fetched by setup.sh."""
    mod = _load_standalone_inference()
    standalone_dir = Path(mod.__file__).resolve().parent
    setup = (standalone_dir / "setup.sh").read_text()
    assert "opendde.pt" in setup  # opendde_v1 default weights
    for rel in mod._BUNDLED_CHECKPOINTS.values():
        if rel:
            assert Path(rel).name in setup, f"setup.sh does not download bundled checkpoint {rel}"


# ── Config: cloud support gate ───────────────────────────────────────────────


def test_opendde_cloud_unsupported_reason():
    """A bundled model name is cloud-OK; a custom checkpoint path is not."""
    assert OpenDDEConfig(use_msa=False).cloud_unsupported_reason() is None
    assert OpenDDEConfig(use_msa=False, model_checkpoint="opendde_abag").cloud_unsupported_reason() is None
    assert OpenDDEConfig(use_msa=False, model_checkpoint="/local/ckpt.pt").cloud_unsupported_reason() is not None


# ── Dispatch / metric assembly (mocked worker) ───────────────────────────────


def _fake_dispatch_factory(captured, *, metrics):
    """Build a ToolInstance.dispatch stand-in that records each call into ``captured``."""

    def fake_dispatch(toolkit, input_data, **kwargs):
        captured.setdefault("toolkits", []).append(toolkit)
        captured.setdefault("input_data", []).append(input_data)
        return {"structure_cif_output": synthetic_cif(["A"]), "metrics": dict(metrics)}

    return fake_dispatch


def test_opendde_run_builds_structure_with_metrics(monkeypatch):
    """run_opendde threads config through dispatch and assembles OpenDDEMetrics (monomer iptm=0.0)."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 91.2,
        "ptm": 0.85,
        "iptm": 0.0,
        "gpde": 1.2,
        "ranking_score": 0.77,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(
        OpenDDEInput(complexes=[_CRO_SEQUENCE]),
        OpenDDEConfig(
            use_msa=False,
            num_samples=2,
            num_steps=33,
            num_cycles=4,
            model_checkpoint="opendde_abag",
        ),
    )

    assert result.success
    assert result.tool_id == "opendde-prediction"
    assert len(result.structures) == 1

    structure = result.structures[0]
    assert structure.source == "opendde-prediction"
    assert is_valid_structure(structure.structure_cif)

    m = structure.metrics
    assert m["avg_plddt"] == pytest.approx(91.2)
    assert m["ptm"] == pytest.approx(0.85)
    assert m["gpde"] == pytest.approx(1.2)
    assert m["ranking_score"] == pytest.approx(0.77)
    assert m["has_clash"] is False
    # OpenDDE always reports iptm (0.0 for a single-chain input).
    assert m["iptm"] == pytest.approx(0.0)
    assert_metrics_in_spec(result)

    # Config sampling params must reach the worker payload, not just the config object.
    assert captured["toolkits"] == ["opendde"]
    [input_data] = captured["input_data"]
    assert input_data["operation"] == "predict"
    assert input_data["num_samples"] == 2
    assert input_data["num_steps"] == 33
    assert input_data["num_cycles"] == 4
    assert input_data["model_checkpoint"] == "opendde_abag"


def test_opendde_iptm_surfaced_when_present(monkeypatch):
    """A non-None iptm from the worker is carried onto OpenDDEMetrics."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 80.0,
        "ptm": 0.7,
        "iptm": 0.62,
        "gpde": 2.0,
        "ranking_score": 0.5,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(OpenDDEInput(complexes=[_CRO_SEQUENCE]), OpenDDEConfig(use_msa=False))

    assert result.structures[0].metrics["iptm"] == pytest.approx(0.62)


def test_opendde_pae_attached_when_requested(monkeypatch):
    """include_pae_matrix=True threads the flag to the worker and attaches pae/avg_pae."""
    captured: dict = {}
    pae = [[0.0, 1.5], [1.5, 0.0]]
    metrics = {
        "avg_plddt": 90.0,
        "ptm": 0.8,
        "iptm": 0.0,
        "gpde": 1.0,
        "ranking_score": 0.7,
        "has_clash": False,
        "avg_pae": 0.75,
        "pae": pae,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(
        OpenDDEInput(complexes=[_CRO_SEQUENCE]),
        OpenDDEConfig(use_msa=False, include_pae_matrix=True),
    )

    assert captured["input_data"][0]["include_pae_matrix"] is True
    m = result.structures[0].metrics
    assert m["avg_pae"] == pytest.approx(0.75)
    assert m["pae"] == pae
    assert_metrics_in_spec(result)


def test_opendde_pae_absent_by_default(monkeypatch):
    """Without include_pae_matrix the worker gets False and no pae/avg_pae is attached."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 90.0,
        "ptm": 0.8,
        "iptm": 0.0,
        "gpde": 1.0,
        "ranking_score": 0.7,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(OpenDDEInput(complexes=[_CRO_SEQUENCE]), OpenDDEConfig(use_msa=False))

    assert captured["input_data"][0]["include_pae_matrix"] is False
    m = result.structures[0].metrics
    assert "pae" not in m
    assert "avg_pae" not in m


def test_opendde_supplied_msas_force_use_msa_true(monkeypatch, tmp_path):
    """Supplied MSAs must set OpenDDE --use_msa=true; otherwise its featurizer ignores them.

    OpenDDE's infer_dataloader skips MSA featurization entirely when use_msa is false,
    so passing false with supplied paths silently folds single-sequence.
    """
    captured: dict = {}
    metrics = {
        "avg_plddt": 90.0,
        "ptm": 0.8,
        "iptm": 0.5,
        "gpde": 1.0,
        "ranking_score": 0.7,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )
    seq_a, seq_b = "MKTAYIAKQR", "GSHMEELLSK"
    cx = Complex(
        chains=[
            Chain(id="A", sequence=seq_a, entity_type="protein"),
            Chain(id="B", sequence=seq_b, entity_type="protein"),
        ]
    )
    cx_msas = ComplexMSAs(
        per_chain={0: MSA(aligned_sequences=[seq_a, seq_a]), 1: MSA(aligned_sequences=[seq_b, seq_b])}
    )

    # use_msa=False on the config, but MSAs are supplied → OpenDDE must still load them.
    run_opendde(OpenDDEInput(complexes=[cx], msas=[cx_msas]), OpenDDEConfig(use_msa=False))
    assert captured["input_data"][0]["use_msa"] is True


def test_opendde_no_msa_single_sequence_sets_use_msa_false(monkeypatch):
    """No supplied MSAs and use_msa=False → OpenDDE folds single-sequence (--use_msa false)."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 90.0,
        "ptm": 0.8,
        "iptm": 0.0,
        "gpde": 1.0,
        "ranking_score": 0.7,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )
    run_opendde(OpenDDEInput(complexes=[_CRO_SEQUENCE]), OpenDDEConfig(use_msa=False))
    assert captured["input_data"][0]["use_msa"] is False


def test_opendde_one_structure_per_complex(monkeypatch):
    """Each input complex yields exactly one structure (1:1 cardinality)."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 88.0,
        "ptm": 0.8,
        "iptm": 0.0,
        "gpde": 1.5,
        "ranking_score": 0.6,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(
        OpenDDEInput(complexes=[_CRO_SEQUENCE, _TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False),
    )

    assert result.success
    assert len(result.structures) == 2
    assert captured["toolkits"] == ["opendde", "opendde"]


# ── GPU tests ────────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_opendde_basic_execution():
    """Fold a tiny peptide end-to-end (MSA-free); headline pLDDT is populated.

    Skips on hosts without OpenDDE weights / a GPU (the tool env fails to build).
    """
    result = run_opendde(
        OpenDDEInput(complexes=[_TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False, num_samples=1, num_steps=50, num_cycles=3, seed=42),
    )

    assert result.success
    assert result.tool_id == "opendde-prediction"
    assert len(result.structures) == 1
    structure = result.structures[0]
    assert is_valid_structure(structure.structure_cif)
    assert structure.metrics["avg_plddt"] is not None
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_opendde_pae_matrix_end_to_end():
    """include_pae_matrix=True yields a square per-token PAE matrix + avg_pae in Å."""
    result = run_opendde(
        OpenDDEInput(complexes=[_TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False, num_samples=1, num_steps=50, num_cycles=3, seed=42, include_pae_matrix=True),
    )
    assert result.success
    m = result.structures[0].metrics
    pae = m["pae"]
    assert isinstance(pae, list) and len(pae) > 0
    assert all(len(row) == len(pae) for row in pae), "PAE matrix must be square (n_token x n_token)"
    assert 0.0 <= m["avg_pae"] <= 32.0
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_opendde_export(tmp_path):
    """A folded OpenDDE output exports a non-empty structure file to disk."""
    result = run_opendde(
        OpenDDEInput(complexes=[_TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False, num_samples=1, num_steps=50, num_cycles=3, seed=42),
    )
    assert result.success

    result.export(name="opendde_pred", export_path=str(tmp_path), file_format="cif")
    assert validate_export_output(tmp_path / "opendde_pred")


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("opendde-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_opendde_benchmark(request):
    """Benchmark opendde-prediction on the MfnG protein + L-tyrosine ligand (cold + warm).

    Single ~390-residue protein-ligand complex without MSA — a representative
    co-folding workload shared with the other AF3-family predictors. Cold pass
    measures weight load + first inference; warm pass measures inference only.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = OpenDDEInput(complexes=[complex_])
    config = OpenDDEConfig(use_msa=False, verbose=True)

    result = benchmark_twice(request, "opendde", lambda: run_opendde(inputs=inputs, config=config))

    assert result.success, "OpenDDE benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
