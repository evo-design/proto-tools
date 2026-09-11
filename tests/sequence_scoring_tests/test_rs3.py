"""Tests for Rule Set 3 guide activity scoring."""

import pytest

from proto_tools.tools.sequence_scoring.rs3 import RS3ScoreConfig, RS3ScoreInput, run_rs3_score


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
