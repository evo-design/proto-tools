"""Tests for the Cherimoya toolkit (CATv1 chromatin accessibility models)."""

import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_HS2_FASTA = Path(__file__).parent.parent / "dummy_data" / "hbb_lcr_hs2_hg38.fna"
_K562_DNASE = "ENCSR000EOT"
_HEPG2_DNASE = "ENCSR149XIL"
# Fold 0 log counts of ENCSR000EOT at HS2, from cherimoya 0.3.0 called directly on the CPU.
_HS2_K562_FOLD0_LOG_COUNTS = 7.7422


def _hs2() -> str:
    """The 2,114 bp hg38 window centered on the beta-globin LCR HS2 cCRE."""
    return "".join(line.strip() for line in _HS2_FASTA.read_text().splitlines()[1:])


def _shuffles(sequence: str, n: int) -> list[str]:
    rng = random.Random(0)
    return ["".join(rng.sample(sequence, len(sequence))) for _ in range(n)]


def _one_hot_logits(sequence: str, scale: float) -> list[list[float]]:
    return [[scale if base == b else 0.0 for b in "ACGT"] for base in sequence]


def _fake_applied(payload: dict) -> list[dict]:
    return [
        {"model": spec["model"], "fold": spec["fold"], "source": spec["source"], "sha256": "0" * 64}
        for spec in payload["checkpoints"]
    ]


def _fake_score_rows(payload: dict) -> list[list[dict]]:
    return [
        [{"model": model, "log_counts": 1.0, "fold_log_counts": [1.0]} for model in payload["models"]]
        for _ in payload["sequences"]
    ]


# ── Catalog ───────────────────────────────────────────────────────────────────


def test_cherimoya_catalog_bundles_every_catv1_experiment() -> None:
    """The bundled snapshot covers all 1,518 CATv1 experiments with one default per (assay, biosample)."""
    from proto_tools.tools.sequence_scoring.cherimoya.catalog import load_catalog

    catalog = load_catalog("catv1")
    defaults = Counter((entry.assay, entry.biosample) for entry in catalog.values() if entry.is_default)

    assert len(catalog) == 1518
    assert Counter(entry.assay for entry in catalog.values()) == {"DNase-seq": 1149, "ATAC-seq": 369}
    assert len(defaults) == len({(entry.assay, entry.biosample) for entry in catalog.values()}) == 492
    assert set(defaults.values()) == {1}
    assert catalog[_K562_DNASE].biosample == "K562"


def test_cherimoya_catalog_csv_names_the_pinned_revision() -> None:
    """The bundled CSV was built from the revision the worker downloads checkpoints at."""
    from proto_tools.tools.sequence_scoring.cherimoya import CHERIMOYA_COLLECTIONS, catalog

    spec = CHERIMOYA_COLLECTIONS["catv1"]
    lines = Path(catalog.__file__).with_name(spec.catalog_file).read_text().splitlines()
    header = " ".join(line for line in lines if line.startswith("#"))

    assert f"{spec.hf_repo} at revision {spec.revision}" in header


def test_cherimoya_catalog_filters_and_ranks() -> None:
    """Filters combine, perturbed experiments are hidden from searches, and matches rank by count Pearson."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CHERIMOYA_COLLECTIONS,
        CherimoyaCatalogInput,
        run_cherimoya_catalog,
    )

    k562 = run_cherimoya_catalog(CherimoyaCatalogInput(biosample="^k562$"))
    dnase = run_cherimoya_catalog(CherimoyaCatalogInput(biosample="^K562$", assay="DNase-seq"))
    defaults = run_cherimoya_catalog(CherimoyaCatalogInput(biosample="^K562$", defaults_only=True))
    hidden = run_cherimoya_catalog(CherimoyaCatalogInput(limit=5000))
    shown = run_cherimoya_catalog(CherimoyaCatalogInput(limit=5000, include_perturbed=True))

    pearsons = [entry.count_pearson_fold_0 for entry in k562.experiments]
    assert pearsons == sorted(pearsons, reverse=True)
    assert [entry.experiment_accession for entry in dnase.experiments] == [_K562_DNASE]
    assert {entry.assay for entry in defaults.experiments} == {"DNase-seq", "ATAC-seq"}
    assert shown.n_matched - hidden.n_matched == 8
    assert k562.revision == CHERIMOYA_COLLECTIONS["catv1"].revision


def test_cherimoya_catalog_returns_named_and_default_perturbed_experiments() -> None:
    """Defaults and named accessions come back even when ENCODE marks them perturbed."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaCatalogInput, run_cherimoya_catalog

    defaults = run_cherimoya_catalog(CherimoyaCatalogInput(defaults_only=True, limit=5000))
    named = run_cherimoya_catalog(CherimoyaCatalogInput(accessions=["ENCSR000EPI"]))

    assert defaults.n_matched == 492
    assert sum(entry.perturbed for entry in defaults.experiments) == 4
    assert [(entry.experiment_accession, entry.perturbed) for entry in named.experiments] == [("ENCSR000EPI", True)]
    with pytest.raises(ValueError, match="not experiments in collection"):
        run_cherimoya_catalog(CherimoyaCatalogInput(accessions=["ENCSR000XXX"]))


def test_cherimoya_catalog_rejects_invalid_regex() -> None:
    """The biosample filter must compile as a regular expression."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaCatalogInput

    with pytest.raises(ValidationError, match="not a valid regular expression"):
        CherimoyaCatalogInput(biosample="K562(")


# ── Model selection ───────────────────────────────────────────────────────────


def test_cherimoya_config_suggests_close_accessions() -> None:
    """An accession missing from the collection fails with the nearest matches."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaScoreConfig

    with pytest.raises(ValidationError, match="did you mean ENCSR000EOT"):
        CherimoyaScoreConfig(models=["ENCSR000EOX"])
    with pytest.raises(ValidationError, match=r"neither an ENCODE experiment accession nor a \.torch"):
        CherimoyaScoreConfig(models=["K562"])
    with pytest.raises(ValidationError, match="models must be unique"):
        CherimoyaScoreConfig(models=[_K562_DNASE, _K562_DNASE])
    with pytest.raises(ValidationError, match="folds must be unique"):
        CherimoyaScoreConfig(folds=[0, 0])


def test_cherimoya_config_checkpoint_paths_are_local_only(tmp_path: Path) -> None:
    """A checkpoint path resolves to an absolute file and is refused on remote devices."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaScoreConfig

    checkpoint = tmp_path / "my_model.torch"
    checkpoint.write_bytes(b"")

    config = CherimoyaScoreConfig(models=[_K562_DNASE, str(checkpoint)])

    assert config.models == [_K562_DNASE, str(checkpoint.resolve())]
    assert "local checkpoint files" in (config.remote_unsupported_reason("proto") or "")
    assert CherimoyaScoreConfig(models=[_K562_DNASE]).remote_unsupported_reason("proto") is None
    with pytest.raises(ValidationError, match="does not exist"):
        CherimoyaScoreConfig(models=[str(tmp_path / "missing.torch")])


def test_cherimoya_checkpoint_specs_expand_folds(tmp_path: Path) -> None:
    """Accessions expand to one pinned Hugging Face file per fold; a checkpoint path is one model."""
    from proto_tools.tools.sequence_scoring.cherimoya import CHERIMOYA_COLLECTIONS
    from proto_tools.tools.sequence_scoring.cherimoya.shared_data_models import checkpoint_specs

    revision = CHERIMOYA_COLLECTIONS["catv1"].revision
    specs = checkpoint_specs([_K562_DNASE, str(tmp_path / "x.torch")], [0, 2], "catv1")

    assert [(spec["model"], spec["fold"]) for spec in specs] == [
        (_K562_DNASE, 0),
        (_K562_DNASE, 2),
        (str(tmp_path / "x.torch"), None),
    ]
    assert specs[1]["source"] == f"programmable-genomics/CATv1@{revision}:models/ENCSR000EOT/cherimoya.fold_2.torch"
    assert specs[2]["source"] == str(tmp_path / "x.torch")


def test_cherimoya_gradient_loss_terms_reject_unknown_keys() -> None:
    """A misspelled loss-term key is an error, not a silently ignored field."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaGradientConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CherimoyaGradientConfig(loss_terms=[{"model": _HEPG2_DNASE, "dirction": "min"}])
    with pytest.raises(ValidationError, match="only used by loss terms with quantity='log_target_counts'"):
        CherimoyaGradientConfig(target_start=10)


# ── Wrappers (dispatch mocked) ────────────────────────────────────────────────


def test_cherimoya_score_maps_worker_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_cherimoya_score sends output-relative target spans and maps rows per model."""
    import proto_tools.tools.sequence_scoring.cherimoya.cherimoya_score as cherimoya_score
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_score,
    )

    payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        payloads.append((toolkit, payload))
        row = {"model": _K562_DNASE, "log_counts": 5.0, "fold_log_counts": [4.0, 6.0], "target_counts": 12.5}
        rows = [[row, {**row, "model": _HEPG2_DNASE, "log_counts": 3.0}]]
        return {"scores": rows, "applied": _fake_applied(payload)}

    monkeypatch.setattr(cherimoya_score.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    sequence = "A" * 3000
    result = run_cherimoya_score(
        CherimoyaScoreInput(sequences=[{"sequence": sequence, "target_range": {"start": 1000, "end": 1200}}]),
        CherimoyaScoreConfig(models=[_K562_DNASE, _HEPG2_DNASE], folds=[0, 3]),
    )

    toolkit, payload = payloads[0]
    assert toolkit == "cherimoya"
    assert payload["sequences"] == [sequence[443:2557]]
    assert payload["target_spans"] == [(0, 200)]
    assert [(spec["model"], spec["fold"]) for spec in payload["checkpoints"]] == [
        (_K562_DNASE, 0),
        (_K562_DNASE, 3),
        (_HEPG2_DNASE, 0),
        (_HEPG2_DNASE, 3),
    ]
    scores = result.results[0].scores
    assert [score.model for score in scores] == [_K562_DNASE, _HEPG2_DNASE]
    assert [score["log_counts"] for score in scores] == [5.0, 3.0]
    assert scores[0]["target_counts"] == 12.5
    assert result.models[1].biosample == "HepG2"
    assert result.models[1].folds == [0, 3]
    assert_metrics_in_spec(result)


def test_cherimoya_score_refuses_mislabeled_worker_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint loaded from another file, or rows in another model order, raise instead of mislabeling."""
    import proto_tools.tools.sequence_scoring.cherimoya.cherimoya_score as cherimoya_score
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_score,
    )

    def wrong_file(toolkit, payload, *, instance=None, config=None):
        applied = [{**entry, "source": "other/repo@abc:x.torch"} for entry in _fake_applied(payload)]
        return {"scores": _fake_score_rows(payload), "applied": applied}

    def wrong_order(toolkit, payload, *, instance=None, config=None):
        return {"scores": [row[::-1] for row in _fake_score_rows(payload)], "applied": _fake_applied(payload)}

    inputs = CherimoyaScoreInput(sequences=["A" * 2114])
    monkeypatch.setattr(cherimoya_score.ToolInstance, "dispatch", staticmethod(wrong_file))
    with pytest.raises(RuntimeError, match="cherimoya worker ran checkpoints"):
        run_cherimoya_score(inputs, CherimoyaScoreConfig())
    monkeypatch.setattr(cherimoya_score.ToolInstance, "dispatch", staticmethod(wrong_order))
    with pytest.raises(RuntimeError, match="cherimoya worker returned models"):
        run_cherimoya_score(inputs, CherimoyaScoreConfig(models=[_K562_DNASE, _HEPG2_DNASE]))


def test_cherimoya_predict_maps_worker_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_cherimoya_predict passes the value kind through and maps one profile per model."""
    import proto_tools.tools.sequence_scoring.cherimoya.cherimoya_predict as cherimoya_predict
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaPredictConfig,
        CherimoyaPredictInput,
        run_cherimoya_predict,
    )

    payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        payloads.append(payload)
        rows = [
            [
                {"model": model, "log_counts": 2.0, "fold_log_counts": [2.0], "values": [0.001] * 1000}
                for model in payload["models"]
            ]
            for _ in payload["sequences"]
        ]
        return {"profiles": rows, "applied": _fake_applied(payload)}

    monkeypatch.setattr(cherimoya_predict.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    result = run_cherimoya_predict(
        CherimoyaPredictInput(sequences=["C" * 2114, "G" * 2114]), CherimoyaPredictConfig(values="probabilities")
    )

    assert payloads[0]["values"] == "probabilities"
    assert result.values == "probabilities"
    assert [len(r.profiles) for r in result.results] == [1, 1]
    assert result.results[1].profiles[0].model == _K562_DNASE
    assert (result.results[0].output_start, result.results[0].output_end) == (557, 1557)


def test_cherimoya_gradient_loads_shared_checkpoints_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss terms on the same model share checkpoints, and the target span is output-relative."""
    import proto_tools.tools.sequence_scoring.cherimoya.cherimoya_gradient as cherimoya_gradient
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaGradientConfig,
        CherimoyaGradientInput,
        CherimoyaGradientLossTerm,
        run_cherimoya_gradient,
    )

    payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        payloads.append(payload)
        return {
            "gradient": [[[0.0] * 4] * 200],
            "loss": 0.5,
            "metrics": {"losses": [0.5], "loss_terms": [[]]},
            "vocab": ["A", "C", "G", "T"],
            "applied": _fake_applied(payload),
        }

    monkeypatch.setattr(cherimoya_gradient.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    result = run_cherimoya_gradient(
        CherimoyaGradientInput(logits=[[[0.0] * 4] * 200]),
        CherimoyaGradientConfig(
            context_left="A" * 957,
            context_right="C" * 957,
            target_start=50,
            target_end=150,
            loss_terms=[
                CherimoyaGradientLossTerm(model=_K562_DNASE, folds=[0], quantity="log_target_counts"),
                CherimoyaGradientLossTerm(model=_HEPG2_DNASE, direction="min"),
                CherimoyaGradientLossTerm(model=_K562_DNASE, folds=[0, 1]),
            ],
        ),
    )

    payload = payloads[0]
    assert [(spec["model"], spec["fold"]) for spec in payload["checkpoints"]] == [
        (_K562_DNASE, 0),
        (_K562_DNASE, 1),
        (_HEPG2_DNASE, 0),
    ]
    assert [term["checkpoints"] for term in payload["loss_terms"]] == [[0], [2], [0, 1]]
    assert payload["target_span"] == (450, 550)
    assert [info.model for info in result.models] == [_K562_DNASE, _HEPG2_DNASE]
    assert result.sample_metrics[0]["loss"] == 0.5


def test_cherimoya_gradient_validates_window_geometry() -> None:
    """Context plus logits must total 2,114 bp, and a target span must sit in the output window."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaGradientConfig,
        CherimoyaGradientInput,
        CherimoyaGradientLossTerm,
        run_cherimoya_gradient,
    )

    with pytest.raises(ValueError, match="Cherimoya needs 2114"):
        run_cherimoya_gradient(CherimoyaGradientInput(logits=[[[0.0] * 4] * 200]), CherimoyaGradientConfig())
    with pytest.raises(ValueError, match="central 1,000 bp output window"):
        run_cherimoya_gradient(
            CherimoyaGradientInput(logits=[[[0.0] * 4] * 2114]),
            CherimoyaGradientConfig(loss_terms=[CherimoyaGradientLossTerm(quantity="log_target_counts")]),
        )


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
def test_cherimoya_score_hs2_on_cpu() -> None:
    """On CPU, the K562 DNase model scores HS2 as the direct model call does, above shuffled copies."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_score,
    )

    hs2 = _hs2()
    result = run_cherimoya_score(
        CherimoyaScoreInput(sequences=[hs2, *_shuffles(hs2, 4)]),
        CherimoyaScoreConfig(models=[_K562_DNASE], device="cpu"),
    )

    validate_output(result)
    assert_metrics_in_spec(result)
    values = [r.scores[0]["log_counts"] for r in result.results]
    assert values[0] == pytest.approx(_HS2_K562_FOLD0_LOG_COUNTS, abs=1e-3)
    assert values[0] > max(values[1:])
    assert len(result.models[0].checkpoint_sha256[0]) == 64


@pytest.mark.integration
def test_cherimoya_checkpoint_path_matches_its_accession_on_cpu(tmp_path: Path) -> None:
    """A downloaded CATv1 checkpoint passed as a path scores exactly as its accession does."""
    import requests

    from proto_tools.tools.sequence_scoring.cherimoya import (
        CHERIMOYA_COLLECTIONS,
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_score,
    )

    spec = CHERIMOYA_COLLECTIONS["catv1"]
    url = f"https://huggingface.co/{spec.hf_repo}/resolve/{spec.revision}/{spec.checkpoint_filename(_K562_DNASE, 0)}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    checkpoint = tmp_path / "k562_fold0.torch"
    checkpoint.write_bytes(response.content)

    result = run_cherimoya_score(
        CherimoyaScoreInput(sequences=[_hs2()]),
        CherimoyaScoreConfig(models=[_K562_DNASE, str(checkpoint)], device="cpu"),
    )

    by_accession, by_path = result.results[0].scores
    assert by_path["log_counts"] == by_accession["log_counts"]
    assert result.models[1].folds == []
    assert result.models[1].collection is None
    assert result.models[1].checkpoint_sha256 == result.models[0].checkpoint_sha256


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_cherimoya_profiles_and_strands_are_consistent() -> None:
    """Expected counts sum to expm1(log_counts), and strand-averaged scores ignore orientation."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaPredictConfig,
        CherimoyaPredictInput,
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_predict,
        run_cherimoya_score,
    )

    hs2 = _hs2()
    reverse_complement = hs2[::-1].translate(str.maketrans("ACGT", "TGCA"))
    counts = run_cherimoya_predict(CherimoyaPredictInput(sequences=[hs2]), CherimoyaPredictConfig(folds=[0, 1]))
    probabilities = run_cherimoya_predict(
        CherimoyaPredictInput(sequences=[hs2]), CherimoyaPredictConfig(values="probabilities")
    )
    stranded = run_cherimoya_score(
        CherimoyaScoreInput(sequences=[hs2, reverse_complement]),
        CherimoyaScoreConfig(models=[_K562_DNASE, _HEPG2_DNASE], reverse_complement_average=True),
    )

    validate_output(counts)
    profile = counts.results[0].profiles[0]
    assert len(profile.values) == 1000
    assert sum(profile.values) == pytest.approx(math.expm1(profile.log_counts), rel=1e-4)
    assert profile.log_counts == pytest.approx(sum(profile.fold_log_counts) / 2, rel=1e-6)
    assert sum(probabilities.results[0].profiles[0].values) == pytest.approx(1.0, rel=1e-5)
    forward, reverse = stranded.results
    for a, b in zip(forward.scores, reverse.scores, strict=True):
        assert a["log_counts"] == pytest.approx(b["log_counts"], abs=1e-4)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_cherimoya_gradient_matches_score_and_finite_differences() -> None:
    """Near-one-hot logits reproduce the score tool, and the gradient matches central differences."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaGradientConfig,
        CherimoyaGradientInput,
        CherimoyaGradientLossTerm,
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_gradient,
        run_cherimoya_score,
    )

    hs2 = _hs2()
    left, design, right = hs2[:957], hs2[957:1157], hs2[1157:]
    term = CherimoyaGradientLossTerm(model=_K562_DNASE, direction="max", sigmoid_center=7.0)
    targeted = CherimoyaGradientLossTerm(model=_K562_DNASE, quantity="log_target_counts", sigmoid_center=6.0)
    config = CherimoyaGradientConfig(
        context_left=left, context_right=right, loss_terms=[term, targeted], deterministic=True
    )
    score = run_cherimoya_score(CherimoyaScoreInput(sequences=[hs2]), CherimoyaScoreConfig())
    span = run_cherimoya_score(
        CherimoyaScoreInput(sequences=[{"sequence": hs2, "target_range": {"start": 957, "end": 1157}}]),
        CherimoyaScoreConfig(),
    )

    sharp = run_cherimoya_gradient(CherimoyaGradientInput(logits=[_one_hot_logits(design, 30.0)]), config)
    window, target = sharp.sample_metrics[0].loss_terms
    assert window["value"] == pytest.approx(score.results[0].scores[0]["log_counts"], abs=1e-3)
    assert span.results[0].output_start == 557
    assert target["value"] == pytest.approx(math.log1p(span.results[0].scores[0]["target_counts"]), abs=1e-3)

    # Soft logits put the K562 value near 4.4; a steep sigmoid there keeps gradients far above
    # the float32 noise in a differenced loss.
    logits = _one_hot_logits(design, 2.0)
    steep = config.model_copy(
        update={"loss_terms": [term.model_copy(update={"sigmoid_center": 4.4, "sigmoid_scale": 0.25})]}
    )
    first = run_cherimoya_gradient(CherimoyaGradientInput(logits=[logits]), steep)
    second = run_cherimoya_gradient(CherimoyaGradientInput(logits=[logits]), steep)
    validate_output(first)
    assert_metrics_in_spec(first)
    assert first.gradient is not None
    assert first.gradient == second.gradient

    forward_only = steep.model_copy(update={"compute_gradient": False})
    largest = sorted(
        (
            (abs(value), position, column)
            for position, row in enumerate(first.gradient[0])
            for column, value in enumerate(row)
        ),
        reverse=True,
    )[:3]
    eps = 5e-2
    for _, position, column in largest:
        plus = [row[:] for row in logits]
        minus = [row[:] for row in logits]
        plus[position][column] += eps
        minus[position][column] -= eps
        loss_plus = run_cherimoya_gradient(CherimoyaGradientInput(logits=[plus]), forward_only).loss
        loss_minus = run_cherimoya_gradient(CherimoyaGradientInput(logits=[minus]), forward_only).loss
        numeric = (loss_plus - loss_minus) / (2 * eps)
        assert first.gradient[0][position][column] == pytest.approx(numeric, rel=0.02)


# ── Benchmarks ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("cherimoya-catalog")
@pytest.mark.slow
def test_cherimoya_catalog_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark cherimoya-catalog: a full listing plus one search per CATv1 biosample (cold + warm)."""
    from proto_tools.tools.sequence_scoring.cherimoya import CherimoyaCatalogInput, run_cherimoya_catalog
    from proto_tools.tools.sequence_scoring.cherimoya.catalog import load_catalog

    biosamples = sorted({entry.biosample for entry in load_catalog("catv1").values()})

    def runner() -> list:
        return [run_cherimoya_catalog(CherimoyaCatalogInput(limit=5000))] + [
            run_cherimoya_catalog(CherimoyaCatalogInput(biosample=f"^{re.escape(b)}$", include_perturbed=True))
            for b in biosamples
        ]

    t0 = time.perf_counter()
    runner()
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    results = runner()
    warm = time.perf_counter() - t0
    request.node.user_properties.append(("cold_seconds", cold))
    request.node.user_properties.append(("warm_seconds", warm))

    validate_output(results[0])
    assert results[0].n_matched == 1510
    assert all(result.n_matched >= 1 for result in results[1:])


@pytest.mark.benchmark("cherimoya-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_cherimoya_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark cherimoya-score on 10,000 2,114 bp sequences with 10 models (cold + warm)."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaCatalogInput,
        CherimoyaScoreConfig,
        CherimoyaScoreInput,
        run_cherimoya_catalog,
        run_cherimoya_score,
    )

    models = [e.experiment_accession for e in run_cherimoya_catalog(CherimoyaCatalogInput(limit=10)).experiments]
    inputs = CherimoyaScoreInput(sequences=random_dna_sequences(n=10000, length=2114, seed=0))  # type: ignore[arg-type]
    config = CherimoyaScoreConfig(models=models, batch_size=256)

    result = benchmark_twice(request, "cherimoya", lambda: run_cherimoya_score(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)
    assert len(result.results) == 10000
    assert all(math.isfinite(score["log_counts"]) for r in result.results for score in r.scores)


@pytest.mark.benchmark("cherimoya-predict")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_cherimoya_predict_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark cherimoya-predict on 1,000 2,114 bp sequences with 4 models and 5 folds (cold + warm)."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaPredictConfig,
        CherimoyaPredictInput,
        run_cherimoya_predict,
    )

    inputs = CherimoyaPredictInput(sequences=random_dna_sequences(n=1000, length=2114, seed=0))  # type: ignore[arg-type]
    config = CherimoyaPredictConfig(
        models=[_K562_DNASE, _HEPG2_DNASE, "ENCSR868FGK", "ENCSR291GJU"], folds=[0, 1, 2, 3, 4], batch_size=128
    )

    result = benchmark_twice(request, "cherimoya", lambda: run_cherimoya_predict(inputs, config))
    validate_output(result)
    assert len(result.results) == 1000
    assert all(len(profile.values) == 1000 for r in result.results for profile in r.profiles)


@pytest.mark.benchmark("cherimoya-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_cherimoya_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark cherimoya-gradient: 64 relaxed 200 bp designs in HS2 context, K562 up and HepG2 down."""
    from proto_tools.tools.sequence_scoring.cherimoya import (
        CherimoyaGradientConfig,
        CherimoyaGradientInput,
        CherimoyaGradientLossTerm,
        run_cherimoya_gradient,
    )

    hs2 = _hs2()
    rng = random.Random(0)
    logits = [[[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(200)] for _ in range(64)]
    inputs = CherimoyaGradientInput(logits=logits)
    config = CherimoyaGradientConfig(
        context_left=hs2[:957],
        context_right=hs2[1157:],
        loss_terms=[
            CherimoyaGradientLossTerm(model=_K562_DNASE, direction="max"),
            CherimoyaGradientLossTerm(model=_HEPG2_DNASE, direction="min"),
        ],
    )

    result = benchmark_twice(request, "cherimoya", lambda: run_cherimoya_gradient(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)
    assert result.gradient is not None
    assert len(result.gradient) == 64
    assert all(math.isfinite(value) for sample in result.gradient for row in sample for value in row)
