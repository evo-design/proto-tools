"""Tests for Malinois MPRA sequence scoring."""

import math

import pytest
from pydantic import ValidationError

from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output


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

    inputs = MalinoisGradientInput(inputs=[[[0.0] * 4] * 200])

    assert len(inputs.inputs) == 1
    assert len(inputs.inputs[0]) == 200
    with pytest.raises(ValidationError, match="must have 4 columns"):
        MalinoisGradientInput(inputs=[[[0.0] * 3]])
    # A bare (L, 4) matrix is one item, not an error.
    bare = MalinoisGradientInput(inputs=[[0.0] * 4] * 200)
    assert len(bare.inputs) == 1
    assert len(bare.inputs[0]) == 200
    with pytest.raises(ValidationError, match="at least one item"):
        MalinoisGradientInput(inputs=[])


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


def test_malinois_gradient_dispatches_gradient_terms(monkeypatch) -> None:
    """run_malinois_gradient dispatches relaxed DNA logits and one term per requested cell."""
    import proto_tools.tools.sequence_scoring.malinois.malinois_score as malinois_score
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisActivityTerm,
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )

    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_payloads.append((toolkit, payload))
        return {
            "terms": [
                {
                    "cell_type": term["cell_type"],
                    "values": [5.0 + index],
                    "gradients": [[[0.1 * (index + 1), 0.0, 0.0, -0.1]] * 200],
                }
                for index, term in enumerate(payload["gradient_terms"])
            ],
            "metrics": {"batch_size": 1},
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(malinois_score.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    result = run_malinois_gradient(
        MalinoisGradientInput(inputs=[[[0.0] * 4] * 200], temperature=0.7),
        MalinoisGradientConfig(
            gradient_terms=[
                MalinoisActivityTerm(cell_type="K562"),
                MalinoisActivityTerm(cell_type="HepG2"),
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
    assert [term["cell_type"] for term in payload["gradient_terms"]] == ["K562", "HepG2"]
    assert len(result.results) == 1

    # Each cell comes back as its own term carrying the raw activity, not a transform of it
    k562, hepg2 = result.results[0].terms
    assert [t.term.cell_type for t in (k562, hepg2)] == ["K562", "HepG2"]
    assert (k562.term_value, hepg2.term_value) == (5.0, 6.0)
    assert k562.gradient is not None and len(k562.gradient) == 200
    assert k562.metadata is not None
    assert k562.metadata.positions == hepg2.metadata.positions, "one basis shared across a sample's terms"
    assert result.vocab == ["A", "C", "G", "T"]


def test_malinois_gradient_rejects_wrong_sequence_length() -> None:
    """Malinois gradient logits must match config.seq_length."""
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )

    with pytest.raises(ValueError, match="logits must have length 200"):
        run_malinois_gradient(MalinoisGradientInput(inputs=[[[0.0] * 4] * 199]), MalinoisGradientConfig())


def _softmax_rows(logits):
    """Softmax each row of a logits matrix."""
    rows = []
    for row in logits:
        peak = max(row)
        exponentiated = [math.exp(value - peak) for value in row]
        total = sum(exponentiated)
        rows.append([value / total for value in exponentiated])
    return rows


@pytest.mark.uses_gpu
def test_malinois_gradient_probability_mode_matches_logit_mode() -> None:
    """temperature=None on softmax(L) evaluates the same relaxed sequence as temperature=1.0 on L."""
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )

    logits = [[0.1, 0.2, 0.3, 0.4]] * 200
    config = MalinoisGradientConfig(device="cuda")

    logit_mode = run_malinois_gradient(MalinoisGradientInput(inputs=[logits], temperature=1.0), config)
    probability_mode = run_malinois_gradient(
        MalinoisGradientInput(inputs=[_softmax_rows(logits)], temperature=None), config
    )

    (probability_term,) = probability_mode.results[0].terms
    (logit_term,) = logit_mode.results[0].terms
    assert probability_term.term_value == pytest.approx(logit_term.term_value, rel=1e-5)
    assert probability_term.gradient is not None
    assert all(math.isfinite(v) for row in probability_term.gradient for v in row)


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
def test_malinois_real_gpu_gradient() -> None:
    """Real GPU smoke test for batched differentiable Malinois scoring."""
    _skip_if_no_malinois_gpu()

    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisActivityTerm,
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )

    result = run_malinois_gradient(
        MalinoisGradientInput(inputs=[[[0.0] * 4] * 200, [[0.1, 0.2, 0.3, 0.4]] * 200], temperature=1.0),
        MalinoisGradientConfig(
            gradient_terms=[
                MalinoisActivityTerm(cell_type="K562"),
                MalinoisActivityTerm(cell_type="HepG2"),
                MalinoisActivityTerm(cell_type="SKNSH"),
            ],
            device="cuda",
        ),
    )

    assert len(result.results) == 2
    for item in result.results:
        assert [term.term.cell_type for term in item.terms] == ["K562", "HepG2", "SKNSH"]
        for term in item.terms:
            assert term.gradient is not None
            assert len(term.gradient) == 200
            assert all(len(row) == 4 for row in term.gradient)
            assert all(math.isfinite(value) for row in term.gradient for value in row)
            assert any(value != 0.0 for row in term.gradient for value in row)
            assert math.isfinite(term.term_value)
    assert result.vocab == ["A", "C", "G", "T"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_one_hot_relaxed_input_matches_discrete_scores() -> None:
    """A one-hot relaxed input must score the same as the discrete sequence it encodes.

    The relaxed path feeds a probability matrix through the MPRA flank builder while the scoring
    path tokenizes a string, so agreement rules out a transposed nucleotide column or an
    off-by-one in the relaxed encoding. ``temperature=None`` makes the comparison exact: a
    softmax over one-hot *logits* would not be one-hot.
    """
    _skip_if_no_malinois_gpu()

    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisActivityTerm,
        MalinoisGradientConfig,
        MalinoisGradientInput,
        MalinoisScoreConfig,
        MalinoisScoreInput,
        run_malinois_gradient,
        run_malinois_score,
    )
    from proto_tools.utils.sequence import DNA_NUCLEOTIDES

    sequence = ("ACGTTGCAAGGCTTACGATCGATTACGGCATTCAGGATCCGTAACGTTAC" * 4)[:200]
    one_hot = [[1.0 if base == nucleotide else 0.0 for nucleotide in DNA_NUCLEOTIDES] for base in sequence]

    discrete = run_malinois_score(
        MalinoisScoreInput(sequences=[sequence]),
        MalinoisScoreConfig(cell_types=["K562"], device="cuda"),
    )
    relaxed = run_malinois_gradient(
        MalinoisGradientInput(inputs=[one_hot], temperature=None),
        MalinoisGradientConfig(
            gradient_terms=[MalinoisActivityTerm(cell_type="K562")],
            compute_gradient=False,
            device="cuda",
        ),
    )

    expected = discrete.results[0].scores["K562"]
    # The term reports the raw activity, which is the same quantity malinois-score returns.
    (activity,) = relaxed.results[0].terms
    assert activity.term_value == pytest.approx(expected, rel=1e-4, abs=1e-4), (
        f"relaxed one-hot activity {activity.term_value} != discrete {expected}"
    )
    assert activity.gradient is None, "compute_gradient=False reports the value without a gradient"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_malinois_real_gpu_batched_gradient_matches_batch_size_one_calls() -> None:
    """Batched gradient results match separate B=1 calls for the same logits."""
    _skip_if_no_malinois_gpu()

    import numpy as np

    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisActivityTerm,
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )
    from proto_tools.utils import ToolInstance

    logits_batch = [
        [[0.0, 0.1, 0.2, 0.3]] * 200,
        [[0.3, 0.2, 0.1, 0.0]] * 200,
    ]
    config = MalinoisGradientConfig(
        gradient_terms=[
            MalinoisActivityTerm(cell_type="K562"),
            MalinoisActivityTerm(cell_type="HepG2"),
            MalinoisActivityTerm(cell_type="SKNSH"),
        ],
        device="cuda",
    )
    with ToolInstance.persist_tool("malinois") as instance:
        batched = run_malinois_gradient(MalinoisGradientInput(inputs=logits_batch), config, instance=instance)
        singles = [
            run_malinois_gradient(MalinoisGradientInput(inputs=[logits]), config, instance=instance)
            for logits in logits_batch
        ]

    assert len(batched.results) == 2
    for idx, single in enumerate(singles):
        for batched_term, single_term in zip(batched.results[idx].terms, single.results[0].terms, strict=True):
            assert batched_term.term.cell_type == single_term.term.cell_type
            assert batched_term.term_value == pytest.approx(single_term.term_value, abs=1e-6)
            assert batched_term.gradient is not None and single_term.gradient is not None
            # Conv kernels reduce in a different order at batch size 2; a real bug would misalign samples.
            np.testing.assert_allclose(
                np.asarray(batched_term.gradient), np.asarray(single_term.gradient), rtol=1e-3, atol=1e-6
            )


@pytest.mark.benchmark("malinois-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_malinois_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark malinois-score on 25000 DNA inserts of length 200 (cold + warm)."""
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisScoreConfig,
        MalinoisScoreInput,
        run_malinois_score,
    )

    sequences = random_dna_sequences(n=25000, length=200, seed=0)
    inputs = MalinoisScoreInput(sequences=sequences)
    config = MalinoisScoreConfig(
        cell_types=["K562", "HepG2", "SKNSH"],
        seq_length=200,
        batch_size=32,
    )

    result = benchmark_twice(request, "malinois", lambda: run_malinois_score(inputs, config))
    validate_output(result)
    assert_metrics_in_spec(result)

    assert result.tool_id == "malinois-score"
    assert len(result.results) == 25000
    assert result.cell_types == ["K562", "HepG2", "SKNSH"]
    assert result.seq_length == 200
    for sequence_result in result.results:
        assert sequence_result.sequence_length == 200
        assert set(sequence_result.scores) == {"K562", "HepG2", "SKNSH"}
        assert all(math.isfinite(score) for score in sequence_result.scores.values())


@pytest.mark.benchmark("malinois-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_malinois_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark malinois-gradient: 5 loops of a 256x200x4 batched DNA logit backward pass (cold + warm)."""
    from proto_tools.tools.sequence_scoring.malinois import (
        MalinoisActivityTerm,
        MalinoisGradientConfig,
        MalinoisGradientInput,
        run_malinois_gradient,
    )
    from proto_tools.utils import DNA_NUCLEOTIDES

    base_index = {nucleotide: column for column, nucleotide in enumerate(DNA_NUCLEOTIDES)}
    sequences = random_dna_sequences(n=256, length=200, seed=0)
    logits = [
        [[1.0 if column == base_index[base] else 0.0 for column in range(len(DNA_NUCLEOTIDES))] for base in sequence]
        for sequence in sequences
    ]
    inputs = MalinoisGradientInput(inputs=logits, temperature=1.0)
    config = MalinoisGradientConfig(
        gradient_terms=[
            MalinoisActivityTerm(cell_type="K562"),
            MalinoisActivityTerm(cell_type="HepG2"),
            MalinoisActivityTerm(cell_type="SKNSH"),
        ],
    )

    def run_batch() -> object:
        last = None
        for _ in range(5):
            last = run_malinois_gradient(inputs, config)
        return last

    result = benchmark_twice(request, "malinois", run_batch)
    validate_output(result)

    assert result.tool_id == "malinois-gradient"
    assert len(result.results) == 256
    for item in result.results:
        assert [term.term.cell_type for term in item.terms] == ["K562", "HepG2", "SKNSH"]
        assert all(term.gradient is not None for term in item.terms)
        assert all(len(term.gradient) == 200 for term in item.terms)
        assert all(len(row) == 4 for term in item.terms for row in term.gradient)
    assert result.vocab == ["A", "C", "G", "T"]
