"""Tests for Mol-1 all-atom structure prediction."""

import importlib.util
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import Structure, is_valid_structure
from proto_tools.tools import Chain, Mol1Config, Mol1Input, run_mol1_prediction
from proto_tools.tools.structure_prediction import Complex, ComplexMSAs
from proto_tools.tools.structure_prediction.mol1.helpers import complex_to_mol1_json
from tests._structure_fixtures import synthetic_cif
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec

_PEPTIDE = "NLYIQWLKDGGPSSGRPPPS"
_METRICS = {
    "avg_plddt": 91.2,
    "ptm": 0.85,
    "iptm": 0.0,
    "gpde": 1.2,
    "ranking_score": 0.17,
    "has_clash": False,
    "chain_plddt": [0.912],
    "chain_pair_iptm": [[0.0]],
}


@pytest.fixture
def captured_dispatch(monkeypatch):
    """Capture worker payloads and read temporary alignments before cleanup."""
    captured = []

    def dispatch(toolkit, input_data, **kwargs):
        alignments = {}
        chains = []
        for entry in input_data["job"]["sequences"]:
            if "ligand" in entry:
                continue
            chain = next(iter(entry.values()))
            chain_id = chain["id"][0]
            chains.append(chain_id)
            alignments[chain_id] = {
                field: Path(chain[field]).read_text()
                for field in ("unpairedMsaPath", "pairedMsaPath")
                if field in chain
            }
        captured.append({"toolkit": toolkit, "payload": input_data, "alignments": alignments})
        metrics = dict(_METRICS)
        if input_data["include_pae_matrix"]:
            metrics.update(pae=[[0.0, 1.5], [1.5, 0.0]], avg_pae=0.75)
        return {"structure_cif_output": synthetic_cif(chains), "metrics": metrics}

    monkeypatch.setattr("proto_tools.tools.structure_prediction.mol1.mol1_prediction.ToolInstance.dispatch", dispatch)
    return captured


def test_mol1_json_preserves_ligand_representations_and_ids():
    """CCD ligands use the native prefix while novel ligands retain raw SMILES."""
    novel = Fragment(
        id="Novel",
        smiles="FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
    )
    assert novel.ccd_code is None
    job = complex_to_mol1_json(
        [
            Chain(id="Receptor", entity_type="protein", sequence=_PEPTIDE),
            Fragment(id="ATP", ccd_code="atp"),
            novel,
        ],
        "prediction",
    )
    assert job == {
        "name": "prediction",
        "sequences": [
            {"proteinChain": {"id": ["Receptor"], "count": 1, "sequence": _PEPTIDE}},
            {"ligand": {"id": ["ATP"], "count": 1, "ligand": "CCD_ATP"}},
            {"ligand": {"id": ["Novel"], "count": 1, "ligand": novel.smiles}},
        ],
    }


@pytest.mark.parametrize(
    ("kind", "sequence", "code", "entry_key", "expected_mod"),
    [
        ("protein", "MSTK", "sep", "proteinChain", {"ptmType": "CCD_SEP", "ptmPosition": 2}),
        ("dna", "ACGT", "5cm", "dnaSequence", {"modificationType": "CCD_5CM", "basePosition": 2}),
        ("rna", "ACGU", "omc", "rnaSequence", {"modificationType": "CCD_OMC", "basePosition": 2}),
    ],
)
def test_mol1_json_preserves_residue_modifications(kind, sequence, code, entry_key, expected_mod):
    """Modified residues keep their one-based positions and native CCD encoding."""
    chain = Chain(id="Modified", entity_type=kind, sequence=sequence, modifications=[(2, code)])
    job = complex_to_mol1_json([chain], "prediction")
    assert job["sequences"] == [
        {entry_key: {"id": ["Modified"], "count": 1, "sequence": sequence, "modifications": [expected_mod]}}
    ]


@pytest.mark.parametrize("include_pae", [False, True])
def test_mol1_dispatch_assembles_metrics_and_forwards_sampling(captured_dispatch, include_pae):
    """Sampling settings reach the worker and confidence retains its documented scales."""
    result = run_mol1_prediction(
        Mol1Input(complexes=[Complex(chains=[Chain(id="Receptor", sequence=_PEPTIDE)])]),
        Mol1Config(
            device="cpu",
            use_msa=False,
            num_samples=2,
            num_steps=33,
            num_cycles=4,
            dtype="fp32",
            chunk_size=128,
            sample_diffusion_chunk_size=2,
            include_pae_matrix=include_pae,
            seed=101,
        ),
    )
    assert result.success
    assert result.tool_id == "mol1-prediction"
    assert len(result.structures) == 1
    structure = result.structures[0]
    assert structure.source == "mol1-prediction"
    assert is_valid_structure(structure.structure_cif)
    assert structure.get_chain_ids() == ["Receptor"]
    for name, expected in _METRICS.items():
        assert structure.metrics[name] == expected
    if include_pae:
        assert structure.metrics["pae"] == [[0.0, 1.5], [1.5, 0.0]]
        assert structure.metrics["avg_pae"] == pytest.approx(0.75)
    else:
        assert "pae" not in structure.metrics
        assert "avg_pae" not in structure.metrics
    assert_metrics_in_spec(result)
    [capture] = captured_dispatch
    assert capture["toolkit"] == "mol1"
    payload = capture["payload"]
    expected = {
        "operation": "predict",
        "num_samples": 2,
        "num_steps": 33,
        "num_cycles": 4,
        "dtype": "fp32",
        "chunk_size": 128,
        "sample_diffusion_chunk_size": 2,
        "include_pae_matrix": include_pae,
        "seed": 101,
        "use_msa": False,
    }
    assert {key: payload[key] for key in expected} == expected


def test_mol1_supplied_msas_keep_paired_rows_and_unpaired_depth(captured_dispatch):
    """Separate paired and independent alignments survive serialization and activate MSA features."""
    seq_a, seq_b = "MKTAYIAKQR", "GSHMEELLSK"
    paired = {
        0: MSA(aligned_sequences=[seq_a, "MKTAYIAKQA", "MKTAYIAKQE"]),
        1: MSA(aligned_sequences=[seq_b, "GSHMEELLSA", "GSHMEELLSE"]),
    }
    unpaired = {
        0: MSA(aligned_sequences=[seq_a, "MKTAYIAKQD", "MKTAYIAKQN", "MKTAYIAKQS"]),
        1: MSA(aligned_sequences=[seq_b, "GSHMEELLSD"]),
    }
    inputs = Mol1Input(
        complexes=[Complex(chains=[Chain(id="Alpha", sequence=seq_a), Chain(id="Beta", sequence=seq_b)])],
        msas=[ComplexMSAs(per_chain=paired, paired=True, unpaired_per_chain=unpaired)],
    )
    config = Mol1Config(device="cpu", use_msa=False)
    result = run_mol1_prediction(inputs, config)
    assert result.success
    assert config.use_msa is False
    [capture] = captured_dispatch
    assert capture["payload"]["use_msa"] is True
    for index, chain_id in enumerate(("Alpha", "Beta")):
        alignments = capture["alignments"][chain_id]
        unpaired_rows = [line for line in alignments["unpairedMsaPath"].splitlines() if not line.startswith(">")]
        assert unpaired_rows == unpaired[index].aligned_sequences
        paired_lines = alignments["pairedMsaPath"].splitlines()
        assert paired_lines[1::2] == paired[index].aligned_sequences
        assert paired_lines[0] == ">query"
        assert all(header.startswith(">tr|") for header in paired_lines[2::2])


def test_mol1_unpaired_msa_has_no_paired_path(captured_dispatch):
    """An independent alignment is supplied only to the unpaired input slot."""
    msa = MSA(aligned_sequences=[_PEPTIDE, "NLYIQWLKDGGPSSGRPPPA"])
    result = run_mol1_prediction(
        Mol1Input(complexes=[_PEPTIDE], msas=[{0: msa}]), Mol1Config(device="cpu", use_msa=False)
    )
    assert result.success
    [capture] = captured_dispatch
    assert set(capture["alignments"]["A"]) == {"unpairedMsaPath"}
    rows = capture["alignments"]["A"]["unpairedMsaPath"].splitlines()[1::2]
    assert rows == msa.aligned_sequences


def test_mol1_seeded_batch_preserves_duplicate_items_and_item_indices(captured_dispatch):
    """Duplicate complexes each reach the worker with a distinct RNG offset."""
    result = run_mol1_prediction(
        Mol1Input(complexes=[_PEPTIDE, _PEPTIDE]), Mol1Config(device="cpu", use_msa=False, seed=7)
    )
    assert result.success
    assert len(result.structures) == 2
    assert [capture["payload"]["item_index"] for capture in captured_dispatch] == [0, 1]
    assert [capture["payload"]["seed"] for capture in captured_dispatch] == [7, 7]


@pytest.mark.parametrize("device", ["proto", "modal"])
def test_mol1_local_checkpoint_is_rejected_on_remote_workers(device):
    """Remote workers cannot read an arbitrary checkpoint from the caller's filesystem."""
    assert Mol1Config().remote_unsupported_reason(device) is None
    assert Mol1Config(model_checkpoint="/local/custom.pt").remote_unsupported_reason(device)


@pytest.fixture
def standalone_module(monkeypatch):
    """Load the isolated runner with the same shared helpers used by real workers."""
    from proto_tools.utils.standalone_helpers_source import standalone_helpers

    monkeypatch.setitem(sys.modules, "standalone_helpers", standalone_helpers)
    path = Path(__file__).resolve().parents[2] / "proto_tools/tools/structure_prediction/mol1/standalone/inference.py"
    spec = importlib.util.spec_from_file_location("_mol1_standalone_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_prediction_sample(root, seed, rank, score, chain_id, pae=None):
    """Write the native upstream confidence/CIF filenames for one diffusion sample."""
    directory = root / "job" / f"seed_{seed}" / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in _METRICS.items() if key != "avg_plddt"}
    summary.update(plddt=_METRICS["avg_plddt"], ranking_score=score)
    (directory / f"job_summary_confidence_sample_{rank}.json").write_text(json.dumps(summary))
    cif = synthetic_cif([chain_id])
    (directory / f"job_sample_{rank}.cif").write_text(cif)
    if pae is not None:
        (directory / f"job_full_data_sample_{rank}.json").write_text(json.dumps({"token_pair_pae": pae}))
    return cif


@pytest.mark.parametrize("include_pae", [False, True])
def test_mol1_standalone_selects_best_sample_without_rescaling_confidence(standalone_module, tmp_path, include_pae):
    """Selection spans seeds and reads PAE only from the winning sample when requested."""
    _write_prediction_sample(tmp_path, seed=101, rank=0, score=-99.0, chain_id="Loser")
    pae = [[0.0, 1.5], [1.5, 0.0]]
    winner_cif = _write_prediction_sample(
        tmp_path, seed=102, rank=1, score=0.8, chain_id="Winner", pae=pae if include_pae else None
    )
    output = standalone_module._extract_structure_and_scores(str(tmp_path), include_pae_matrix=include_pae)
    assert output["structure_cif_output"] == winner_cif
    metrics = output["metrics"]
    assert metrics["ranking_score"] == pytest.approx(0.8)
    assert metrics["avg_plddt"] == pytest.approx(91.2)
    assert metrics["chain_plddt"] == [0.912]
    assert metrics["iptm"] == 0.0
    assert metrics["chain_pair_iptm"] == [[0.0]]
    if include_pae:
        assert metrics["pae"] == pae
        assert metrics["avg_pae"] == pytest.approx(0.75)
    else:
        assert "pae" not in metrics
        assert "avg_pae" not in metrics


def test_mol1_standalone_rejects_nonfinite_ranking(standalone_module, tmp_path):
    """A corrupt score cannot silently choose an arbitrary conformation."""
    _write_prediction_sample(tmp_path, seed=101, rank=0, score=float("nan"), chain_id="A")
    with pytest.raises(ValueError, match="ranking score"):
        standalone_module._extract_structure_and_scores(str(tmp_path))


@pytest.mark.parametrize("pae", [[], [[1.0, 2.0], [3.0]], [[float("nan")]]])
def test_mol1_standalone_rejects_malformed_pae(standalone_module, tmp_path, pae):
    """Requested token PAE must be a nonempty square matrix with finite entries."""
    _write_prediction_sample(tmp_path, seed=101, rank=0, score=0.8, chain_id="A", pae=pae)
    with pytest.raises(ValueError, match="token_pair_pae"):
        standalone_module._extract_structure_and_scores(str(tmp_path), include_pae_matrix=True)


@pytest.mark.parametrize(("seed", "item_index", "expected"), [(7, 0, 7), (7, 1, 8), (2**32 - 1, 1, 0), (None, 1, 99)])
def test_mol1_standalone_dispatch_derives_per_item_sampling_seed(
    standalone_module, monkeypatch, seed, item_index, expected
):
    """Seeded duplicates diversify reproducibly; unseeded calls retain random sampling."""
    seen, seeded = [], []
    standalone_module._model = types.SimpleNamespace(predict=lambda payload: seen.append(payload) or {})
    monkeypatch.setattr(standalone_module, "set_torch_seed", seeded.append)
    monkeypatch.setattr(standalone_module, "get_random_int", lambda: 99)
    standalone_module.dispatch({"operation": "predict", "seed": seed, "item_index": item_index})
    assert seeded == [expected if seed is not None else None]
    assert seen[0]["sampling_seed"] == expected


def test_mol1_warm_runner_applies_changed_sampling_and_recycling(standalone_module, monkeypatch):
    """A reused model updates cached recycling state, diffusion settings and confidence dumping."""
    snapshots, dumpers, deterministic_modes = [], [], []
    configs = types.SimpleNamespace(
        model=types.SimpleNamespace(N_cycle=20),
        sample_diffusion=types.SimpleNamespace(),
        infer_setting=types.SimpleNamespace(),
        use_template=False,
        use_rna_msa=False,
        use_seeds_in_json=False,
    )

    def update_model_configs(active_configs):
        runner.model.configs = active_configs

    runner = types.SimpleNamespace(
        configs=configs,
        model=types.SimpleNamespace(N_cycle=20, configs=configs),
        init_dumper=lambda **kwargs: dumpers.append(kwargs),
        update_model_configs=update_model_configs,
    )

    def infer_predict(active_runner, active_configs):
        assert active_runner is runner
        active_runner.update_model_configs(active_configs)
        assert json.loads(Path(active_configs.input_json_path).read_text()) == [{"name": "job", "sequences": []}]
        snapshots.append((deepcopy(active_configs), runner.model.N_cycle))

    upstream = types.ModuleType("mol.runner.inference")
    upstream.infer_predict = infer_predict
    upstream.update_gpu_compatible_configs = lambda _: None
    monkeypatch.setitem(sys.modules, "mol.runner.inference", upstream)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(use_deterministic_algorithms=deterministic_modes.append)
    )
    monkeypatch.setattr(standalone_module, "_extract_structure_and_scores", lambda *args, **kwargs: {})
    model = standalone_module.Mol1Model()
    model.runner, model.device, model.model_checkpoint, model._loaded = runner, "cuda:0", "mol-1-base", True
    payload = {
        "job": {"name": "job", "sequences": []},
        "device": "cuda:0",
        "model_checkpoint": "mol-1-base",
        "num_cycles": 4,
        "num_samples": 2,
        "num_steps": 33,
        "dtype": "fp32",
        "chunk_size": 128,
        "sample_diffusion_chunk_size": 2,
        "use_msa": False,
        "include_pae_matrix": False,
        "sampling_seed": 101,
        "seed": 101,
        "verbose": False,
    }
    requests = [
        payload,
        {
            **payload,
            "num_cycles": 6,
            "num_samples": 3,
            "num_steps": 44,
            "dtype": "bf16",
            "chunk_size": None,
            "sample_diffusion_chunk_size": None,
            "use_msa": True,
            "include_pae_matrix": True,
            "sampling_seed": 102,
            "seed": None,
        },
    ]
    for request in requests:
        model.predict(request)
    for request, (snapshot, cached_cycles) in zip(requests, snapshots, strict=True):
        assert snapshot.model.N_cycle == cached_cycles == request["num_cycles"]
        assert snapshot.sample_diffusion.N_sample == request["num_samples"]
        assert snapshot.sample_diffusion.N_step == request["num_steps"]
        assert snapshot.dtype == request["dtype"]
        assert snapshot.infer_setting.chunk_size == request["chunk_size"]
        assert snapshot.infer_setting.sample_diffusion_chunk_size == request["sample_diffusion_chunk_size"]
        assert snapshot.use_msa == request["use_msa"]
        assert snapshot.need_atom_confidence == request["include_pae_matrix"]
        assert snapshot.seeds == [request["sampling_seed"]]
        assert snapshot.deterministic is (request["seed"] is not None)
        assert snapshot.use_template is False
        assert snapshot.use_rna_msa is False
        assert snapshot.use_seeds_in_json is False
    assert dumpers == [
        {"need_atom_confidence": False, "sorted_by_ranking_score": True},
        {"need_atom_confidence": True, "sorted_by_ranking_score": True},
    ]
    assert deterministic_modes == [True, False]


@pytest.mark.uses_gpu
def test_mol1_prediction_default_sampling():
    """Fold a real peptide with the upstream default sampling configuration."""
    result = run_mol1_prediction(Mol1Input(complexes=[_PEPTIDE]), Mol1Config(use_msa=False, seed=101))
    assert result.success
    assert result.tool_id == "mol1-prediction"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert "".join(aa for aa, _ in result.structures[0].get_residue_position_map()["A"]) == _PEPTIDE
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_mol1_prediction_nondefault_sampling_with_pae(tmp_path):
    """Nondefault sampling returns token PAE in angstroms and exportable coordinates."""
    result = run_mol1_prediction(
        Mol1Input(complexes=[_PEPTIDE]),
        Mol1Config(
            use_msa=False,
            num_samples=2,
            num_steps=25,
            num_cycles=2,
            seed=42,
            dtype="fp32",
            chunk_size=128,
            sample_diffusion_chunk_size=2,
            include_pae_matrix=True,
        ),
    )
    assert result.success
    assert result.tool_id == "mol1-prediction"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    metrics = result.structures[0].metrics
    pae = metrics["pae"]
    assert len(pae) == len(_PEPTIDE)
    assert all(len(row) == len(_PEPTIDE) for row in pae)
    assert metrics["avg_pae"] == pytest.approx(sum(map(sum, pae)) / len(_PEPTIDE) ** 2)
    assert_metrics_in_spec(result)
    for file_format in ("pdb", "cif"):
        result.export(file_format, export_path=tmp_path, file_format=file_format)
        exported = Structure.from_file(tmp_path / file_format / f"structure_0.{file_format}")
        assert "".join(aa for aa, _ in exported.get_residue_position_map()["A"]) == _PEPTIDE


@pytest.mark.benchmark("mol1-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_mol1_prediction_benchmark(request):
    """Benchmark protein-ligand co-folding with MSA search on cold and warm workers."""
    from tests.conftest import benchmark_twice
    from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex

    inputs = Mol1Input(complexes=[load_benchmark_complex("MfnG_and_ligand")])
    config = Mol1Config(use_msa=True, verbose=True)
    result = benchmark_twice(request, "mol1", lambda: run_mol1_prediction(inputs, config))
    assert result.success
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
