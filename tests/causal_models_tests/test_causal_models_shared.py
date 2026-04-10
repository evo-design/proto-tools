"""Cross-tool contract and behavioral tests for causal model shared base classes.

Verifies that all 4 causal model tools (evo1, evo2, progen2, progen3) honor
the shared Input/Config/Output contracts and runtime behavioral invariants.
"""

import math

import pytest
from pydantic import ValidationError

from proto_tools.tools.causal_models.evo1 import (
    Evo1SampleConfig,
    Evo1SampleInput,
    Evo1SampleOutput,
    Evo1ScoringConfig,
    Evo1ScoringInput,
    Evo1ScoringOutput,
    run_evo1_sample,
    run_evo1_score,
)
from proto_tools.tools.causal_models.evo2 import (
    Evo2SampleConfig,
    Evo2SampleInput,
    Evo2SampleOutput,
    Evo2ScoringConfig,
    Evo2ScoringInput,
    Evo2ScoringOutput,
    run_evo2_sample,
    run_evo2_score,
)
from proto_tools.tools.causal_models.progen2 import (
    ProGen2SampleConfig,
    ProGen2SampleInput,
    ProGen2SampleOutput,
    ProGen2ScoringConfig,
    ProGen2ScoringInput,
    ProGen2ScoringOutput,
    run_progen2_sample,
    run_progen2_score,
)
from proto_tools.tools.causal_models.progen3 import (
    ProGen3SampleConfig,
    ProGen3SampleInput,
    ProGen3SampleOutput,
    ProGen3ScoringConfig,
    ProGen3ScoringInput,
    ProGen3ScoringOutput,
    run_progen3_sample,
    run_progen3_score,
)
from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleConfig,
    CausalModelSampleInput,
    CausalModelSampleOutput,
    CausalModelScoringOutput,
    SequenceScores,
)
from proto_tools.utils import PROTEIN_AMINO_ACIDS
from proto_tools.utils.tool_instance import ToolInstance
from tests.conftest import _gpu_available

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


_DNA_CHARS = set("ACGTN")
_PROTEIN_CHARS = set(PROTEIN_AMINO_ACIDS)

SAMPLE_TOOLS_GPU = [
    pytest.param(
        run_evo1_sample,
        Evo1SampleInput(prompts=["ATCGATCG"]),
        lambda **kw: Evo1SampleConfig(num_tokens=20, verbose=False, **kw),
        "ATCGATCG",
        _DNA_CHARS,
        id="evo1",
    ),
    pytest.param(
        run_evo2_sample,
        Evo2SampleInput(prompts=["ATCGATCG"]),
        lambda **kw: Evo2SampleConfig(
            model_checkpoint="evo2_7b", num_tokens=20, verbose=False, print_generation=False, **kw
        ),
        "ATCGATCG",
        _DNA_CHARS,
        id="evo2",
    ),
    pytest.param(
        run_progen2_sample,
        ProGen2SampleInput(prompts=["MKAL"]),
        lambda **kw: ProGen2SampleConfig(model_checkpoint="progen2-small", max_length=30, verbose=False, **kw),
        "MKAL",
        _PROTEIN_CHARS,
        id="progen2",
    ),
    pytest.param(
        run_progen3_sample,
        ProGen3SampleInput(prompts=["MKAL"]),
        lambda **kw: ProGen3SampleConfig(model_checkpoint="progen3-112m", max_new_tokens=20, verbose=False, **kw),
        "MKAL",
        _PROTEIN_CHARS,
        id="progen3",
    ),
]

SCORING_TOOLS_GPU = [
    pytest.param(
        run_evo1_score, Evo1ScoringInput(sequences=["ATCGATCGATCG"]), Evo1ScoringConfig(verbose=False), id="evo1"
    ),
    pytest.param(
        run_evo2_score,
        Evo2ScoringInput(sequences=["ATCGATCGATCG"]),
        Evo2ScoringConfig(model_checkpoint="evo2_7b", verbose=False),
        id="evo2",
    ),
    pytest.param(
        run_progen2_score,
        ProGen2ScoringInput(sequences=["MKTLVIVTGA"]),
        ProGen2ScoringConfig(model_checkpoint="progen2-small", verbose=False),
        id="progen2",
    ),
    pytest.param(
        run_progen3_score,
        ProGen3ScoringInput(sequences=["MKTLVIVTGA"]),
        ProGen3ScoringConfig(model_checkpoint="progen3-112m", verbose=False),
        id="progen3",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _persist_tools(request):
    """Auto-cache tool workers for GPU tests to avoid repeated model loads."""
    if request.config.getoption("--cpu") or not _gpu_available():
        yield
        return
    with ToolInstance.persist():
        yield


@pytest.mark.uses_gpu
@pytest.mark.parametrize("run_fn,inputs,config_factory,prompt,valid_chars", SAMPLE_TOOLS_GPU)
def test_sampling_invariants(run_fn, inputs, config_factory, prompt, valid_chars):
    """Verify prepend_prompt behavior and valid sequence characters."""
    # prepend_prompt=True: output starts with prompt, all chars valid
    result = run_fn(inputs=inputs, config=config_factory(prepend_prompt=True))
    assert result.success, f"Tool failed: {result.errors}"
    seq = result.sequences[0]
    assert seq.startswith(prompt), f"prepend_prompt=True: expected '{prompt}' prefix, got '{seq[:20]}'"
    invalid = set(seq.upper()) - valid_chars
    assert not invalid, f"Invalid characters in output: {invalid}"

    # prepend_prompt=False: output does NOT start with prompt, chars still valid
    result_no = run_fn(inputs=inputs, config=config_factory(prepend_prompt=False))
    assert result_no.success, f"Tool failed: {result_no.errors}"
    seq_no = result_no.sequences[0]
    assert not seq_no.startswith(prompt), f"prepend_prompt=False: unexpected '{prompt}' prefix in '{seq_no[:20]}'"
    assert len(seq_no) > 0
    invalid_no = set(seq_no.upper()) - valid_chars
    assert not invalid_no, f"Invalid characters without prepend: {invalid_no}"


@pytest.mark.uses_gpu
@pytest.mark.parametrize("run_fn,inputs,config", SCORING_TOOLS_GPU)
def test_scoring_invariants(run_fn, inputs, config):
    """Verify scoring metrics: required keys, bounds, and math consistency."""
    result = run_fn(inputs=inputs, config=config)
    assert result.success, f"Tool failed: {result.errors}"

    for i, score in enumerate(result.scores):
        missing = {"log_likelihood", "avg_log_likelihood", "perplexity"} - set(score.metrics)
        assert not missing, f"Score {i} missing metrics: {missing}"
        assert score.log_likelihood < 0, f"Score {i}: log_likelihood should be negative"
        assert score.perplexity >= 1.0, f"Score {i}: perplexity should be >= 1.0"
        expected_ppl = math.exp(-score.avg_log_likelihood)
        assert abs(score.perplexity - expected_ppl) / max(expected_ppl, 1e-10) < 1e-4, (
            f"Score {i}: perplexity {score.perplexity} != exp(-avg_ll) {expected_ppl}"
        )
