"""Tests for Malinois MPRA sequence scoring."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec


def _skip_if_no_malinois_gpu() -> None:
    from proto_tools.utils.device import number_of_visible_gpus

    if number_of_visible_gpus() < 1:
        pytest.skip("Malinois GPU test requires a visible CUDA GPU")


def test_malinois_input_normalizes_and_validates_dna() -> None:
    """Malinois input accepts one DNA string and rejects non-ACGT bases."""
    from proto_tools.tools.sequence_scoring.malinois import MalinoisScoreInput

    inputs = MalinoisScoreInput(sequences="acgt" * 50)

    assert inputs.sequences == ["ACGT" * 50]
    assert len(inputs) == 1
    with pytest.raises(ValueError, match="Invalid nucleotide characters"):
        MalinoisScoreInput(sequences="ACGN")


def test_malinois_config_rejects_duplicate_cell_types() -> None:
    """Malinois config requires unique requested cell types."""
    from proto_tools.tools.sequence_scoring.malinois import MalinoisScoreConfig

    with pytest.raises(ValidationError, match="cell_types must be unique"):
        MalinoisScoreConfig(cell_types=["K562", "K562"])


def test_malinois_gradient_input_validates_dna_logits() -> None:
    """Malinois gradient input requires a B x L x 4 DNA logit tensor."""
    from proto_tools.tools.sequence_scoring.malinois import MalinoisGradientInput

    inputs = MalinoisGradientInput(logits=[[[0.0] * 4] * 200])

    assert len(inputs.logits) == 1
    assert len(inputs.logits[0]) == 200
    with pytest.raises(ValidationError, match="must have 4 columns"):
        MalinoisGradientInput(logits=[[[0.0] * 3]])
    with pytest.raises(ValidationError):
        MalinoisGradientInput(logits=[[0.0] * 4] * 200)


def test_malinois_run_dispatches_and_maps_scores(monkeypatch) -> None:
    """run_malinois_score dispatches to the Malinois worker and maps per-cell scores."""
    import proto_tools.tools.sequence_scoring.malinois.malinois_score as malinois_score
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisScoreConfig,
        MalinoisScoreInput,
        run_malinois_score,
    )

    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_payloads.append((toolkit, payload))
        return {"scores": [{"K562": 4.0, "HepG2": 1.5}, {"K562": 5.0, "HepG2": 2.5}]}

    monkeypatch.setattr(malinois_score.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_malinois_score(
        MalinoisScoreInput(sequences=["A" * 200, "C" * 200]),
        MalinoisScoreConfig(cell_types=["K562", "HepG2"], batch_size=2, device="cuda"),
    )

    assert captured_payloads[0][0] == "malinois"
    assert captured_payloads[0][1]["sequences"] == ["A" * 200, "C" * 200]
    assert captured_payloads[0][1]["cell_types"] == ["K562", "HepG2"]
    assert captured_payloads[0][1]["seq_length"] == 200
    assert captured_payloads[0][1]["artifact_path"] == ""
    assert captured_payloads[0][1]["artifact_url"].startswith("https://zenodo.org/records/10698014/files/")
    assert captured_payloads[0][1]["artifact_md5"] == "375142a714e7df73c463b46113a65210"
    assert captured_payloads[0][1]["malinois_dir"] == ""
    assert result.cell_types == ["K562", "HepG2"]
    assert dict(result.results[0].scores.items()) == {"K562": 4.0, "HepG2": 1.5}
    assert dict(result.results[1].scores.items()) == {"K562": 5.0, "HepG2": 2.5}
    assert result.results[0].scores.K562 == 4.0
    assert_metrics_in_spec(result)


def test_malinois_run_rejects_wrong_sequence_length() -> None:
    """Malinois scoring requires sequences to match config.seq_length."""
    from proto_tools.tools.sequence_scoring.malinois import MalinoisScoreConfig, MalinoisScoreInput, run_malinois_score

    with pytest.raises(ValueError, match="must have length 200"):
        run_malinois_score(MalinoisScoreInput(sequences=["A" * 199]), MalinoisScoreConfig())


def test_malinois_gradient_dispatches_loss_terms(monkeypatch) -> None:
    """run_malinois_gradient dispatches relaxed DNA logits and max/min objectives."""
    import proto_tools.tools.sequence_scoring.malinois.malinois_score as malinois_score
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisGradientConfig,
        MalinoisGradientInput,
        MalinoisGradientLossTerm,
        run_malinois_gradient,
    )

    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_payloads.append((toolkit, payload))
        return {
            "gradient": [[[0.1, 0.0, 0.0, -0.1]] * 200],
            "loss": 0.25,
            "metrics": {"raw_scores": [{"K562": 5.0}], "loss_terms": [[]], "losses": [0.25]},
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(malinois_score.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_malinois_gradient(
        MalinoisGradientInput(logits=[[[0.0] * 4] * 200], temperature=0.7),
        MalinoisGradientConfig(
            loss_terms=[
                MalinoisGradientLossTerm(cell_type="K562", direction="max", weight=2.0),
                MalinoisGradientLossTerm(cell_type="HepG2", direction="min", weight=0.5),
            ],
            soft=1.0,
            hard=0.5,
            device="cuda",
        ),
    )

    assert captured_payloads[0][0] == "malinois"
    payload = captured_payloads[0][1]
    assert payload["operation"] == "compute_gradient"
    assert payload["temperature"] == 0.7
    assert payload["soft"] == 1.0
    assert payload["hard"] == 0.5
    assert payload["loss_terms"][0]["cell_type"] == "K562"
    assert payload["loss_terms"][0]["direction"] == "max"
    assert payload["loss_terms"][0]["weight"] == 2.0
    assert payload["loss_terms"][1]["direction"] == "min"
    assert result.gradient is not None
    assert len(result.gradient) == 1
    assert len(result.gradient[0]) == 200
    assert len(result.sample_metrics) == 1
    assert result.sample_metrics[0]["loss"] == 0.25
    assert result.sample_metrics[0]["K562"] == 5.0
    assert_metrics_in_spec(result)
    assert result.vocab == ["A", "C", "G", "T"]


def test_malinois_gradient_rejects_wrong_sequence_length() -> None:
    """Malinois gradient logits must match config.seq_length."""
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )

    with pytest.raises(ValueError, match="logits must have length 200"):
        run_malinois_gradient(MalinoisGradientInput(logits=[[[0.0] * 4] * 199]), MalinoisGradientConfig())


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_real_gpu_scores_sequences() -> None:
    """Real GPU smoke test for Malinois scoring through the tool worker."""
    _skip_if_no_malinois_gpu()

    from proto_tools.tools.sequence_scoring.malinois import MalinoisScoreConfig, MalinoisScoreInput, run_malinois_score

    result = run_malinois_score(
        MalinoisScoreInput(sequences=["ACGT" * 50, "TGCA" * 50]),
        MalinoisScoreConfig(cell_types=["K562", "HepG2", "SKNSH"], batch_size=2, device="cuda"),
    )

    assert len(result.results) == 2
    assert result.cell_types == ["K562", "HepG2", "SKNSH"]
    assert result.seq_length == 200
    for sequence_result in result.results:
        assert sequence_result.sequence_length == 200
        assert set(sequence_result.scores) == {"K562", "HepG2", "SKNSH"}
        assert all(math.isfinite(score) for score in sequence_result.scores.values())
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_real_gpu_batched_inference_matches_original_reference() -> None:
    """Batched wrapper inference matches the original Malinois scoring helper."""
    _skip_if_no_malinois_gpu()

    from proto_tools.tools.sequence_scoring.malinois.malinois_score import (
        DEFAULT_MALINOIS_ARTIFACT_MD5,
        DEFAULT_MALINOIS_ARTIFACT_PATH,
        DEFAULT_MALINOIS_ARTIFACT_URL,
        DEFAULT_MALINOIS_DIR,
    )

    standalone_dir = (
        Path(__file__).parents[2] / "proto_tools" / "tools" / "sequence_scoring" / "malinois" / "standalone"
    )
    sys.path.insert(0, str(standalone_dir))
    try:
        spec = importlib.util.spec_from_file_location("malinois_reference_inference", standalone_dir / "inference.py")
        assert spec is not None and spec.loader is not None
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        sequences = ["ACGT" * 50, "TGCA" * 50]
        inference._model.load(
            artifact_path=DEFAULT_MALINOIS_ARTIFACT_PATH,
            artifact_url=DEFAULT_MALINOIS_ARTIFACT_URL,
            artifact_md5=DEFAULT_MALINOIS_ARTIFACT_MD5,
            malinois_dir=DEFAULT_MALINOIS_DIR,
            seq_length=200,
            device="cuda",
            verbose=False,
        )
        wrapped_scores = inference._model.score_sequences(
            sequences=sequences,
            cell_types=["K562", "HepG2", "SKNSH"],
            batch_size=2,
            device="cuda",
        )
        reference_scores = inference.malinois_scoring.score_sequences(
            sequences,
            inference._model.model,
            inference._model.flank_builder,
            batch_size=2,
        )
    finally:
        sys.path.remove(str(standalone_dir))

    for row_idx, wrapped_row in enumerate(wrapped_scores):
        assert wrapped_row["K562"] == pytest.approx(reference_scores[row_idx, 0], abs=1e-6)
        assert wrapped_row["HepG2"] == pytest.approx(reference_scores[row_idx, 1], abs=1e-6)
        assert wrapped_row["SKNSH"] == pytest.approx(reference_scores[row_idx, 2], abs=1e-6)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_real_gpu_gradient() -> None:
    """Real GPU smoke test for batched differentiable Malinois scoring."""
    _skip_if_no_malinois_gpu()

    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisGradientConfig,
        MalinoisGradientInput,
        MalinoisGradientLossTerm,
        run_malinois_gradient,
    )

    result = run_malinois_gradient(
        MalinoisGradientInput(logits=[[[0.0] * 4] * 200, [[0.1, 0.2, 0.3, 0.4]] * 200], temperature=1.0),
        MalinoisGradientConfig(
            loss_terms=[
                MalinoisGradientLossTerm(cell_type="K562", direction="max"),
                MalinoisGradientLossTerm(cell_type="HepG2", direction="min"),
                MalinoisGradientLossTerm(cell_type="SKNSH", direction="min"),
            ],
            device="cuda",
        ),
    )

    assert result.gradient is not None
    assert len(result.gradient) == 2
    assert all(len(matrix) == 200 for matrix in result.gradient)
    assert all(len(row) == 4 for matrix in result.gradient for row in matrix)
    assert all(math.isfinite(value) for matrix in result.gradient for row in matrix for value in row)
    assert any(value != 0.0 for matrix in result.gradient for row in matrix for value in row)
    assert math.isfinite(result.loss)
    assert len(result.sample_metrics) == 2
    assert all(math.isfinite(sample["loss"]) for sample in result.sample_metrics)
    assert all(set(sample) == {"loss", "K562", "HepG2", "SKNSH"} for sample in result.sample_metrics)
    assert len(result.metrics["losses"]) == 2
    assert len(result.metrics["raw_scores"]) == 2
    assert all(set(raw_scores) == {"K562", "HepG2", "SKNSH"} for raw_scores in result.metrics["raw_scores"])
    for term_metrics in result.metrics["loss_terms"]:
        assert [(term["cell_type"], term["direction"]) for term in term_metrics] == [
            ("K562", "max"),
            ("HepG2", "min"),
            ("SKNSH", "min"),
        ]
    assert result.vocab == ["A", "C", "G", "T"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_real_gpu_batched_gradient_matches_batch_size_one_calls() -> None:
    """Batched gradient results match separate B=1 calls for the same logits."""
    _skip_if_no_malinois_gpu()

    import numpy as np

    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisGradientConfig,
        MalinoisGradientInput,
        MalinoisGradientLossTerm,
        run_malinois_gradient,
    )
    from proto_tools.utils import ToolInstance

    logits_batch = [
        [[0.0, 0.1, 0.2, 0.3]] * 200,
        [[0.3, 0.2, 0.1, 0.0]] * 200,
    ]
    config = MalinoisGradientConfig(
        loss_terms=[
            MalinoisGradientLossTerm(cell_type="K562", direction="max"),
            MalinoisGradientLossTerm(cell_type="HepG2", direction="min"),
            MalinoisGradientLossTerm(cell_type="SKNSH", direction="min"),
        ],
        device="cuda",
    )
    with ToolInstance.persist_tool("malinois") as instance:
        batched = run_malinois_gradient(MalinoisGradientInput(logits=logits_batch), config, instance=instance)
        singles = [
            run_malinois_gradient(MalinoisGradientInput(logits=[logits]), config, instance=instance)
            for logits in logits_batch
        ]

    assert batched.gradient is not None
    assert len(batched.gradient) == 2
    for idx, single in enumerate(singles):
        assert single.gradient is not None
        assert batched.sample_metrics[idx]["loss"] == pytest.approx(single.sample_metrics[0]["loss"], abs=1e-6)
        assert batched.metrics["losses"][idx] == pytest.approx(single.metrics["losses"][0], abs=1e-6)
        assert batched.metrics["raw_scores"][idx] == pytest.approx(single.metrics["raw_scores"][0], abs=1e-6)
        np.testing.assert_allclose(np.asarray(batched.gradient[idx]), np.asarray(single.gradient[0]), atol=1e-6)
