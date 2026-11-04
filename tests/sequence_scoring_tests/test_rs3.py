"""Tests for Rule Set 3 guide activity scoring."""

import pytest

from proto_tools.tools.sequence_scoring.rs3 import RS3ScoreConfig, RS3ScoreInput, run_rs3_score
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output


def test_input_normalizes_and_validates_context() -> None:
    context = "gacgaaagcgacaacgcgttcatccgggca"
    assert RS3ScoreInput(sequences=context).sequences == [context.upper()]
    with pytest.raises(ValueError, match="exactly 30"):
        RS3ScoreInput(sequences="A" * 29)
    with pytest.raises(ValueError, match="NGG PAM"):
        RS3ScoreInput(sequences="A" * 30)


def test_dispatch_and_typed_output(monkeypatch) -> None:
    import proto_tools.tools.sequence_scoring.rs3.rs3_score as module

    contexts = ["GACGAAAGCGACAACGCGTTCATCCGGGCA", "AGAAAACACTAGCATCCCCACCCGCGGACT"]
    captured = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured.update({"toolkit": toolkit, "payload": payload})
        return {"scores": [-0.9, 1.1]}

    monkeypatch.setattr(module.ToolInstance, "dispatch", staticmethod(fake_dispatch))
    output = run_rs3_score(RS3ScoreInput(sequences=contexts), RS3ScoreConfig(tracr_rna="Chen2013"))

    assert captured == {"toolkit": "rs3", "payload": {"sequences": contexts, "tracr_rna": "Chen2013", "device": "cpu"}}
    assert [result.metrics.score for result in output.results] == [-0.9, 1.1]
    assert output.results[0].protospacer == contexts[0][4:24]
    assert output.results[0].pam == contexts[0][24:27]
    assert output.model_version == "0.0.18"


def test_rejects_missing_score(monkeypatch) -> None:
    import proto_tools.tools.sequence_scoring.rs3.rs3_score as module

    monkeypatch.setattr(module.ToolInstance, "dispatch", staticmethod(lambda *args, **kwargs: {"scores": []}))
    with pytest.raises(ValueError, match="Expected 1"):
        run_rs3_score(RS3ScoreInput(sequences="GACGAAAGCGACAACGCGTTCATCCGGGCA"), RS3ScoreConfig())


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.integration
def test_rs3_scores_a_known_active_guide_above_a_poly_a_guide() -> None:
    """The real model ranks a screening-validated context above a poly-A one, per tracrRNA."""
    active = "GACGAAAGCGACAACGCGTTCATCCGGGCA"
    inert = "AAAAAAAAAAAAAAAAAAAAAAAACGGAAA"

    for tracr in ("Hsu2013", "Chen2013"):
        result = run_rs3_score(RS3ScoreInput(sequences=[active, inert]), RS3ScoreConfig(tracr_rna=tracr))

        validate_output(result)
        active_score, inert_score = (item.metrics.score for item in result.results)
        assert active_score > inert_score
        assert result.tracr_rna == tracr


@pytest.mark.benchmark("rs3-score")
@pytest.mark.slow
def test_rs3_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark Rule Set 3 on 1,000 valid, distinct 30-nt SpCas9 contexts."""
    bases = "ACGT"
    contexts = [
        "A" * 19 + "".join(bases[(index >> shift) & 3] for shift in (8, 6, 4, 2, 0)) + "TGGGCA"
        for index in range(1_000)
    ]
    inputs = RS3ScoreInput(sequences=contexts)
    result = benchmark_twice(request, "rs3", lambda: run_rs3_score(inputs, RS3ScoreConfig()))

    validate_output(result)
    assert result.tool_id == "rs3-score"
    assert len(result.results) == len(contexts)
