"""Tests for PARADE UTR activity and mRNA stability prediction."""

import math

import pytest
from pydantic import ValidationError

from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output


def _skip_if_no_gpu() -> None:
    from proto_tools.utils.device import number_of_visible_gpus

    if number_of_visible_gpus() < 1:
        pytest.skip("PARADE GPU test requires a visible CUDA GPU")


def test_parade_input_normalizes_and_validates() -> None:
    """PARADE input maps RNA U to T, allows N, and rejects other characters."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityInput

    inputs = ParadeActivityInput(sequences="acgu N")

    assert inputs.sequences == ["ACGTN"]
    assert len(inputs) == 1
    with pytest.raises(ValueError, match="Invalid nucleotide characters"):
        ParadeActivityInput(sequences="ACGTX")


def test_parade_activity_config_resolves_panel() -> None:
    """Empty cell_types resolves to the full panel for the construct type."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig

    assert ParadeActivityConfig(construct_type="utr5").cell_types == ["c1", "c2", "c4", "c6", "c17"]
    assert ParadeActivityConfig(construct_type="utr3").cell_types == ["c1", "c2", "c4", "c6", "c17", "c13"]


def test_parade_activity_config_rejects_offpanel_and_duplicate_cells() -> None:
    """c13 is 3'UTR-only, and requested codes must be unique."""
    from proto_tools.tools.sequence_scoring.parade import ParadeActivityConfig

    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        ParadeActivityConfig(construct_type="utr5", cell_types=["c13"])
    with pytest.raises(ValidationError, match="cell_types must be unique"):
        ParadeActivityConfig(construct_type="utr3", cell_types=["c2", "c2"])


def test_parade_activity_run_dispatches_and_maps_scores(monkeypatch) -> None:
    """run_parade_activity dispatches to the worker and pivots per-cell scores."""
    import proto_tools.tools.sequence_scoring.parade.parade_activity as parade_activity
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    captured = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append((toolkit, payload))
        return {"scores": [{"c1": 2.4, "c2": 2.5}, {"c1": 1.3, "c2": 1.4}]}

    monkeypatch.setattr(parade_activity.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_activity(
        ParadeActivityInput(sequences=["A" * 50, "C" * 50]),
        ParadeActivityConfig(construct_type="utr5", cell_types=["c1", "c2"], batch_size=2, device="cuda"),
    )

    toolkit, payload = captured[0]
    assert toolkit == "parade"
    assert payload["operation"] == "activity"
    assert payload["construct_type"] == "utr5"
    assert payload["cell_types"] == ["c1", "c2"]
    assert payload["checkpoint_url"].endswith("model-utr5-deltas-epoch%3D9-step%3D840.ckpt")
    assert payload["checkpoint_md5"] == "a48aeffc516e32f4d8780b855bbcd849"
    assert payload["checkpoint_filename"] == "parade-model-utr5-deltas-epoch9-step840.ckpt"
    assert result.construct_type == "utr5"
    assert result.cell_types == ["c1", "c2"]
    assert dict(result.results[0].scores.items()) == {"c1": 2.4, "c2": 2.5}
    assert result.results[1].scores.c1 == 1.3
    assert_metrics_in_spec(result)


def test_parade_activity_rejects_mixed_lengths() -> None:
    """Activity scoring requires all sequences in a call to share a length."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    with pytest.raises(ValueError, match="must share a length"):
        run_parade_activity(ParadeActivityInput(sequences=["A" * 50, "C" * 49]), ParadeActivityConfig())


def test_parade_stability_run_dispatches_and_maps_log_ratios(monkeypatch) -> None:
    """run_parade_stability dispatches to the worker and maps per-sequence log-ratios."""
    import proto_tools.tools.sequence_scoring.parade.parade_stability as parade_stability
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    captured = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append((toolkit, payload))
        return {"log_ratios": [-1.64, -0.86]}

    monkeypatch.setattr(parade_stability.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_stability(
        ParadeStabilityInput(sequences=["A" * 50, "G" * 50]),
        ParadeStabilityConfig(device="cuda"),
    )

    toolkit, payload = captured[0]
    assert toolkit == "parade"
    assert payload["operation"] == "stability"
    assert payload["checkpoint_url"].endswith("stability-epoch%3D24-step%3D725.ckpt")
    assert payload["checkpoint_md5"] == "511c0b4d794f948708ab1e6fa866734b"
    assert result.results[0].log_ratio == -1.64
    assert result.results[1].sequence_length == 50


def test_parade_stability_rejects_mixed_lengths() -> None:
    """Stability scoring requires all sequences in a call to share a length."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    with pytest.raises(ValueError, match="must share a length"):
        run_parade_stability(ParadeStabilityInput(sequences=["A" * 50, "C" * 51]), ParadeStabilityConfig())


def test_parade_gradient_input_validates_logits() -> None:
    """PARADE gradient input requires a B x L x 4 logit tensor."""
    from proto_tools.tools.sequence_scoring.parade import ParadeGradientInput

    inputs = ParadeGradientInput(logits=[[[0.0] * 4] * 50])
    assert len(inputs.logits) == 1
    assert len(inputs.logits[0]) == 50
    with pytest.raises(ValidationError, match="must have 4 columns"):
        ParadeGradientInput(logits=[[[0.0] * 3]])
    with pytest.raises(ValidationError):
        ParadeGradientInput(logits=[[0.0] * 4] * 50)


def test_parade_gradient_config_rejects_offpanel_cell() -> None:
    """A loss term cell code must belong to the construct-type panel."""
    from proto_tools.tools.sequence_scoring.parade import ParadeGradientConfig, ParadeGradientLossTerm

    with pytest.raises(ValidationError, match="not in the utr5 panel"):
        ParadeGradientConfig(construct_type="utr5", loss_terms=[ParadeGradientLossTerm(cell_type="c13")])


def test_parade_gradient_run_dispatches_loss_terms(monkeypatch) -> None:
    """run_parade_gradient dispatches relaxed logits, construct type, and objectives."""
    import proto_tools.tools.sequence_scoring.parade.parade_gradient as parade_gradient
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )

    captured = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.append((toolkit, payload))
        return {
            "gradient": [[[0.1, 0.0, 0.0, -0.1]] * 50],
            "loss": 0.25,
            "metrics": {"raw_scores": [{"c2": 2.5}], "loss_terms": [[]], "losses": [0.25]},
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(parade_gradient.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_parade_gradient(
        ParadeGradientInput(logits=[[[0.0] * 4] * 50], temperature=0.7),
        ParadeGradientConfig(
            construct_type="utr5",
            loss_terms=[
                ParadeGradientLossTerm(cell_type="c2", direction="max", weight=2.0),
                ParadeGradientLossTerm(cell_type="c6", direction="min", weight=0.5),
            ],
            soft=1.0,
            hard=0.5,
            device="cuda",
        ),
    )

    toolkit, payload = captured[0]
    assert toolkit == "parade"
    assert payload["operation"] == "gradient"
    assert payload["construct_type"] == "utr5"
    assert payload["temperature"] == 0.7
    assert payload["soft"] == 1.0
    assert payload["hard"] == 0.5
    assert payload["checkpoint_url"].endswith("model-utr5-deltas-epoch%3D9-step%3D840.ckpt")
    assert payload["loss_terms"][0]["cell_type"] == "c2"
    assert payload["loss_terms"][1]["direction"] == "min"
    assert result.gradient is not None
    assert len(result.gradient[0]) == 50
    assert len(result.sample_metrics) == 1
    assert result.sample_metrics[0]["loss"] == 0.25
    assert result.sample_metrics[0]["c2"] == 2.5
    assert result.vocab == ["A", "C", "G", "T"]
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_activity_real_gpu() -> None:
    """Real GPU smoke test for PARADE UTR activity through the tool worker."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    result = run_parade_activity(
        ParadeActivityInput(sequences=["ACGT" * 12 + "AC", "TGCA" * 12 + "GT"]),
        ParadeActivityConfig(construct_type="utr5", batch_size=2, device="cuda"),
    )

    assert len(result.results) == 2
    assert result.cell_types == ["c1", "c2", "c4", "c6", "c17"]
    for sequence_result in result.results:
        assert set(sequence_result.scores) == {"c1", "c2", "c4", "c6", "c17"}
        assert all(math.isfinite(score) for score in sequence_result.scores.values())
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_stability_real_gpu() -> None:
    """Real GPU smoke test for PARADE 3' UTR stability through the tool worker."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    result = run_parade_stability(
        ParadeStabilityInput(sequences=["ACGT" * 12 + "AC", "TGCA" * 12 + "GT"]),
        ParadeStabilityConfig(batch_size=2, device="cuda"),
    )

    assert len(result.results) == 2
    assert all(math.isfinite(sequence_result.log_ratio) for sequence_result in result.results)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_gradient_real_gpu() -> None:
    """Real GPU smoke test for differentiable PARADE UTR activity."""
    _skip_if_no_gpu()

    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )

    result = run_parade_gradient(
        ParadeGradientInput(logits=[[[0.0] * 4] * 50, [[0.1, 0.2, 0.3, 0.4]] * 50], temperature=1.0),
        ParadeGradientConfig(
            construct_type="utr5",
            loss_terms=[
                ParadeGradientLossTerm(cell_type="c2", direction="max"),
                ParadeGradientLossTerm(cell_type="c6", direction="min"),
            ],
            device="cuda",
        ),
    )

    assert result.gradient is not None
    assert len(result.gradient) == 2
    assert all(len(matrix) == 50 for matrix in result.gradient)
    assert any(value != 0.0 for matrix in result.gradient for row in matrix for value in row)
    assert math.isfinite(result.loss)
    assert len(result.sample_metrics) == 2


@pytest.mark.benchmark("parade-activity")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_activity_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-activity on 25000 length-50 UTRs across the 5'UTR panel (cold + warm)."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeActivityConfig,
        ParadeActivityInput,
        run_parade_activity,
    )

    inputs = ParadeActivityInput(sequences=random_dna_sequences(n=25000, length=50, seed=0))
    config = ParadeActivityConfig(construct_type="utr5", batch_size=64)

    result = benchmark_twice(request, "parade", lambda: run_parade_activity(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)
    assert result.tool_id == "parade-activity"
    assert len(result.results) == 25000


@pytest.mark.benchmark("parade-stability")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_stability_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-stability on 25000 length-50 3'UTRs (cold + warm)."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeStabilityConfig,
        ParadeStabilityInput,
        run_parade_stability,
    )

    inputs = ParadeStabilityInput(sequences=random_dna_sequences(n=25000, length=50, seed=0))
    config = ParadeStabilityConfig(batch_size=64)

    result = benchmark_twice(request, "parade", lambda: run_parade_stability(inputs, config))
    validate_output(result)
    assert result.tool_id == "parade-stability"
    assert len(result.results) == 25000


@pytest.mark.benchmark("parade-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-gradient: 5 loops of a 256x50x4 batched UTR logit backward pass (cold + warm)."""
    from proto_tools.tools.sequence_scoring.parade import (
        ParadeGradientConfig,
        ParadeGradientInput,
        ParadeGradientLossTerm,
        run_parade_gradient,
    )
    from proto_tools.utils import DNA_NUCLEOTIDES

    base_index = {nucleotide: column for column, nucleotide in enumerate(DNA_NUCLEOTIDES)}
    sequences = random_dna_sequences(n=256, length=50, seed=0)
    logits = [
        [[1.0 if column == base_index[base] else 0.0 for column in range(len(DNA_NUCLEOTIDES))] for base in sequence]
        for sequence in sequences
    ]
    inputs = ParadeGradientInput(logits=logits, temperature=1.0)
    config = ParadeGradientConfig(
        construct_type="utr5",
        loss_terms=[
            ParadeGradientLossTerm(cell_type="c2", direction="max"),
            ParadeGradientLossTerm(cell_type="c6", direction="min"),
        ],
    )

    def run_batch() -> object:
        last = None
        for _ in range(5):
            last = run_parade_gradient(inputs, config)
        return last

    result = benchmark_twice(request, "parade", run_batch)
    validate_output(result)
    assert_metrics_in_spec(result)
    assert result.tool_id == "parade-gradient"
    assert result.gradient is not None
    assert len(result.gradient) == 256
