"""Cross-tool contract tests for causal model shared base classes.

Verifies that all 4 causal model tools (evo1, evo2, progen2, progen3) honor
the shared Input/Config/Output contracts after migration to base classes.
"""

import pytest
from pydantic import ValidationError

from proto_tools.tools.causal_models.evo1 import (
    Evo1SampleConfig,
    Evo1SampleInput,
    Evo1SampleOutput,
    Evo1ScoringConfig,
    Evo1ScoringInput,
    Evo1ScoringOutput,
)
from proto_tools.tools.causal_models.evo2 import (
    Evo2SampleConfig,
    Evo2SampleInput,
    Evo2SampleOutput,
    Evo2ScoringConfig,
    Evo2ScoringInput,
    Evo2ScoringOutput,
)
from proto_tools.tools.causal_models.progen2 import (
    ProGen2SampleConfig,
    ProGen2SampleInput,
    ProGen2SampleOutput,
    ProGen2ScoringConfig,
    ProGen2ScoringInput,
    ProGen2ScoringOutput,
)
from proto_tools.tools.causal_models.progen3 import (
    ProGen3SampleConfig,
    ProGen3SampleInput,
    ProGen3SampleOutput,
    ProGen3ScoringConfig,
    ProGen3ScoringInput,
    ProGen3ScoringOutput,
)
from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleConfig,
    CausalModelSampleInput,
    CausalModelSampleOutput,
    CausalModelScoringOutput,
    SequenceScores,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TOOLS = [
    pytest.param(Evo1SampleInput, Evo1SampleConfig, Evo1SampleOutput, id="evo1"),
    pytest.param(Evo2SampleInput, Evo2SampleConfig, Evo2SampleOutput, id="evo2"),
    pytest.param(ProGen2SampleInput, ProGen2SampleConfig, ProGen2SampleOutput, id="progen2"),
    pytest.param(ProGen3SampleInput, ProGen3SampleConfig, ProGen3SampleOutput, id="progen3"),
]

SCORING_TOOLS = [
    pytest.param(Evo1ScoringInput, Evo1ScoringConfig, Evo1ScoringOutput, id="evo1"),
    pytest.param(Evo2ScoringInput, Evo2ScoringConfig, Evo2ScoringOutput, id="evo2"),
    pytest.param(ProGen2ScoringInput, ProGen2ScoringConfig, ProGen2ScoringOutput, id="progen2"),
    pytest.param(ProGen3ScoringInput, ProGen3ScoringConfig, ProGen3ScoringOutput, id="progen3"),
]


# ── Sample Contract ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_input_contract(input_cls, config_cls, output_cls):
    """Input inherits base, normalizes single string, rejects empty."""
    assert issubclass(input_cls, CausalModelSampleInput)
    assert input_cls(prompts="ATCG").prompts == ["ATCG"]
    with pytest.raises(ValidationError, match="prompts must not be empty"):
        input_cls(prompts=[])


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_config_contract(input_cls, config_cls, output_cls):
    """Config inherits base, has shared fields, rejects invalid values."""
    assert issubclass(config_cls, CausalModelSampleConfig)
    config = config_cls()
    for field in ("temperature", "top_p", "batch_size", "device", "prepend_prompt"):
        assert hasattr(config, field), f"Missing field: {field}"
    with pytest.raises(ValidationError):
        config_cls(temperature=0.0)
    with pytest.raises(ValidationError):
        config_cls(top_p=0.0)
    with pytest.raises(ValidationError):
        config_cls(top_p=1.5)


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_output_contract(input_cls, config_cls, output_cls):
    """Output inherits base, has sequences field, supports fasta/txt export."""
    assert issubclass(output_cls, CausalModelSampleOutput)
    output = output_cls(sequences=["ATCG", "GCTA"])
    assert output.sequences == ["ATCG", "GCTA"]
    assert {"fasta", "txt"} <= set(output.output_format_options)


# ── Scoring Contract ────────────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_input_contract(input_cls, config_cls, output_cls):
    """Input has sequences field, normalizes single string, rejects empty."""
    assert input_cls(sequences=["MKTL", "ACGT"]).sequences == ["MKTL", "ACGT"]
    assert input_cls(sequences="MKTL").sequences == ["MKTL"]
    with pytest.raises(ValidationError, match="sequences must not be empty"):
        input_cls(sequences=[])


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_config_contract(input_cls, config_cls, output_cls):
    """Config has shared fields (batch_size, device)."""
    config = config_cls()
    assert hasattr(config, "batch_size")
    assert hasattr(config, "device")


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_output_contract(input_cls, config_cls, output_cls):
    """Output inherits base, has scores field, supports csv/json export."""
    assert issubclass(output_cls, CausalModelScoringOutput)
    scores = [SequenceScores(metrics={"log_likelihood": -1.5, "perplexity": 4.48})]
    output = output_cls(scores=scores)
    assert output.scores[0].metrics["log_likelihood"] == -1.5
    assert {"csv", "json"} <= set(output.output_format_options)
