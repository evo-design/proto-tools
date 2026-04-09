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


# ── Sample Input Contract ─────────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_input_inherits_base(input_cls, config_cls, output_cls):
    assert issubclass(input_cls, CausalModelSampleInput)


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_input_normalizes_single_string(input_cls, config_cls, output_cls):
    inp = input_cls(prompts="ATCG")
    assert inp.prompts == ["ATCG"]


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_input_rejects_empty(input_cls, config_cls, output_cls):
    with pytest.raises(ValidationError, match="prompts must not be empty"):
        input_cls(prompts=[])


# ── Sample Config Contract ────────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_config_inherits_base(input_cls, config_cls, output_cls):
    assert issubclass(config_cls, CausalModelSampleConfig)


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_config_has_shared_fields(input_cls, config_cls, output_cls):
    config = config_cls()
    assert hasattr(config, "temperature")
    assert hasattr(config, "top_p")
    assert hasattr(config, "batch_size")
    assert hasattr(config, "device")
    assert hasattr(config, "prepend_prompt")


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_config_rejects_zero_temperature(input_cls, config_cls, output_cls):
    with pytest.raises(ValidationError):
        config_cls(temperature=0.0)


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_config_rejects_invalid_top_p(input_cls, config_cls, output_cls):
    with pytest.raises(ValidationError):
        config_cls(top_p=0.0)
    with pytest.raises(ValidationError):
        config_cls(top_p=1.5)


# ── Sample Output Contract ────────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_output_inherits_base(input_cls, config_cls, output_cls):
    assert issubclass(output_cls, CausalModelSampleOutput)


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_output_has_sequences_field(input_cls, config_cls, output_cls):
    output = output_cls(sequences=["ATCG", "GCTA"])
    assert output.sequences == ["ATCG", "GCTA"]


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_output_supports_fasta_export(input_cls, config_cls, output_cls):
    output = output_cls(sequences=["ATCG"])
    assert "fasta" in output.output_format_options


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SAMPLE_TOOLS)
def test_sample_output_supports_txt_export(input_cls, config_cls, output_cls):
    output = output_cls(sequences=["ATCG"])
    assert "txt" in output.output_format_options


# ── Scoring Input Contract ────────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_input_has_sequences_field(input_cls, config_cls, output_cls):
    inp = input_cls(sequences=["MKTL", "ACGT"])
    assert inp.sequences == ["MKTL", "ACGT"]


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_input_normalizes_single_string(input_cls, config_cls, output_cls):
    inp = input_cls(sequences="MKTL")
    assert inp.sequences == ["MKTL"]


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_input_rejects_empty(input_cls, config_cls, output_cls):
    with pytest.raises(ValidationError, match="sequences must not be empty"):
        input_cls(sequences=[])


# ── Scoring Config Contract ───────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_config_has_shared_fields(input_cls, config_cls, output_cls):
    config = config_cls()
    assert hasattr(config, "batch_size")
    assert hasattr(config, "device")


# ── Scoring Output Contract ───────────────────────────────────────────────────


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_output_inherits_base(input_cls, config_cls, output_cls):
    assert issubclass(output_cls, CausalModelScoringOutput)


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_output_has_scores_field(input_cls, config_cls, output_cls):
    scores = [SequenceScores(metrics={"log_likelihood": -1.5, "perplexity": 4.48})]
    output = output_cls(scores=scores)
    assert len(output.scores) == 1
    assert output.scores[0].metrics["log_likelihood"] == -1.5


@pytest.mark.parametrize("input_cls,config_cls,output_cls", SCORING_TOOLS)
def test_scoring_output_supports_csv_and_json_export(input_cls, config_cls, output_cls):
    scores = [SequenceScores(metrics={"log_likelihood": -1.5})]
    output = output_cls(scores=scores)
    assert "csv" in output.output_format_options
    assert "json" in output.output_format_options
