"""Tests for Evo 2 in-context DNA design."""

import pytest
from pydantic import ValidationError

import proto_tools.tools.causal_models.evo2.evo2_icl as evo2_icl
from proto_tools.tools.causal_models.evo2.evo2_icl import Evo2ICLConfig, Evo2ICLInput, run_evo2_icl
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache
from tests.conftest import benchmark_twice

_EXAMPLES = ["ACGTACGT", "TGCATGCA"]
_TWO_HUNDRED_BP_EXAMPLES = ["ACGT" * 50, "TGCA" * 50] * 8


@pytest.fixture(autouse=True)
def fresh_cache():
    """Prevent seeded mock outputs from leaking between tests."""
    token = _program_tool_cache.set(ToolCache())
    yield
    _program_tool_cache.reset(token)


@pytest.fixture
def mock_sampler(monkeypatch):
    """Capture standalone dispatch and allow tests to supply sampled continuations."""
    calls = []
    sampled = []

    def dispatch(toolkit, payload, *, instance=None, config=None):
        calls.append((toolkit, payload, instance, config))
        return {"sequences": list(sampled)}

    monkeypatch.setattr(evo2_icl.ToolInstance, "dispatch", staticmethod(dispatch))
    return calls, sampled


# ── Prompt construction and limits ────────────────────────────────────────────


@pytest.mark.parametrize("container", [list, tuple])
def test_evo2_icl_input_normalizes_preserving_order_and_multiplicity(container):
    """Whitespace and casing normalize without changing the demonstration sequence."""
    inputs = Evo2ICLInput(examples=container([" ac\ngt\t", "tgca", "AC GT"]))

    assert inputs.examples == ["ACGT", "TGCA", "ACGT"]
    assert inputs.prompt == "NACGTNTGCANACGTN"


@pytest.mark.parametrize(
    ("examples", "prompt"),
    [("acgt", "NACGTN"), (["ACGT", "TGCA"], "NACGTNTGCAN")],
)
def test_evo2_icl_input_prompt_uses_one_n_per_boundary(examples, prompt):
    """Single and multiple examples share the same flanked, single-N format."""
    assert Evo2ICLInput(examples=examples).prompt == prompt


@pytest.mark.parametrize("invalid", ["", " \n", "ACGN", "ACGU", ">example\nACGT"])
def test_evo2_icl_input_rejects_invalid_examples(invalid):
    """Examples require DNA bases, with N reserved for the assembled separator."""
    with pytest.raises(ValidationError, match="Example 2 must"):
        Evo2ICLInput(examples=["ACGT", invalid])


def test_evo2_icl_input_prompt_budget_counts_separators():
    """The prompt cap includes both flanks and the separator between examples."""
    inputs = Evo2ICLInput(examples=["A" * 16_383, "C" * 16_382])
    assert len(inputs.prompt) == 32_768

    with pytest.raises(ValidationError, match="assembled prompt must fit within 32,768 bases"):
        Evo2ICLInput(examples=["A" * 16_383, "C" * 16_383])


def test_evo2_icl_config_generation_budget_combines_count_and_length():
    """The hosted-work limit allows its boundary and rejects the next base per design."""
    config = Evo2ICLConfig(num_sequences=1000, sequence_length=1000)
    assert config.num_sequences * config.sequence_length == 1_000_000

    with pytest.raises(ValidationError, match="at most 1,000,000 generated bases"):
        Evo2ICLConfig(num_sequences=1000, sequence_length=1001)


# ── Standalone sampling and output handling ────────────────────────────────────


def test_evo2_icl_dispatches_one_joint_prompt_for_each_design(mock_sampler):
    """FASTA examples jointly condition all designs rather than each producing a run."""
    calls, sampled = mock_sampler
    sampled.extend(["AAAA", "CCCC", "GGGG", "TTTT"])
    inputs = Evo2ICLInput(examples=["ACGT", "TGCA", "ACGT"])

    result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=4, sequence_length=4, device="cpu"))

    assert len(calls) == 1
    assert calls[0][1]["prompts"] == ["NACGTNTGCANACGTN"] * 4
    assert result.sequences == sampled
    assert result.metadata["num_examples"] == 3
    assert result.metadata["prompt_length"] == len(inputs.prompt)
    assert result.metadata["num_generated"] == 4
    assert result.tool_id == "evo2-icl"


@pytest.mark.parametrize("seed", [None, 63])
def test_evo2_icl_dispatch_propagates_sampling_controls_and_seed(mock_sampler, seed):
    """The new operation uses the existing sampler with the supplied decoding controls."""
    calls, sampled = mock_sampler
    sampled.extend(["AAAA", "CCCC"])
    inputs = Evo2ICLInput(examples=_EXAMPLES)
    config = Evo2ICLConfig(
        num_sequences=2,
        sequence_length=4,
        temperature=0.7,
        top_k=2,
        top_p=0.8,
        batch_size=2,
        device="cpu",
        verbose=False,
        timeout=900,
        seed=seed,
    )

    result = run_evo2_icl(inputs, config)

    toolkit, payload, instance, dispatched_config = calls[0]
    assert toolkit == "evo2"
    assert instance is None
    assert dispatched_config.timeout == 900
    assert payload == {
        "operation": "sample",
        "prompts": ["NACGTACGTNTGCATGCAN"] * 2,
        "model_checkpoint": "evo2_7b",
        "local_path": None,
        "top_k": 2,
        "top_p": 0.8,
        "temperature": 0.7,
        "max_new_tokens": 4,
        "cached_generation": True,
        "force_prompt_threshold": 2048,
        "max_seqlen": None,
        "skip_special_tokens": False,
        "stop_at_eos": True,
        "old_kv_cache": None,
        "return_kv_cache": False,
        "return_logits": False,
        "batch_size": 2,
        "device": "cpu",
        "verbose": False,
        "seed": seed,
    }
    assert result.metadata["seed"] == seed


@pytest.mark.parametrize(("batch_size", "threshold"), [(1, 4096), (2, 2048), (3, 1024), (8, 512)])
def test_evo2_icl_bounds_parallel_prefill_without_truncating_context(mock_sampler, batch_size, threshold):
    """Batch-aware prefill stays a power of two while forwarding the complete prompt."""
    calls, sampled = mock_sampler
    sampled.append("ACGT")
    inputs = Evo2ICLInput(examples=["C" * 32_766])

    result = run_evo2_icl(
        inputs,
        Evo2ICLConfig(num_sequences=1, sequence_length=4, batch_size=batch_size, device="cpu"),
    )

    payload = calls[0][1]
    assert payload["force_prompt_threshold"] == threshold
    assert threshold * batch_size <= 4096
    assert payload["prompts"] == [inputs.prompt]
    assert len(payload["prompts"][0]) == 32_768
    assert result.metadata["prefill_threshold"] == threshold


def test_evo2_icl_filters_invalid_generations_with_yield_warning(mock_sampler):
    """Only full-length ACGT continuations survive, in their original sampling order."""
    _, sampled = mock_sampler
    sampled.extend(["AAAA", "ACGN", "AC", "ACGTAC", "TTTT", "acgt"])

    result = run_evo2_icl(
        Evo2ICLInput(examples=_EXAMPLES),
        Evo2ICLConfig(num_sequences=6, sequence_length=4, device="cpu"),
    )

    assert result.sequences == ["AAAA", "TTTT"]
    assert result.metadata["num_requested"] == 6
    assert result.metadata["num_generated"] == 2
    assert result.metadata["num_omitted"] == 4
    assert len(result.warnings) == 1
    assert "Returned 2 of 6 requested designs" in result.warnings[0]
    assert "4 generations were short or contained non-ACGT tokens" in result.warnings[0]


def test_evo2_icl_rejects_all_invalid_generations(mock_sampler):
    """A run without any usable continuation raises an actionable error."""
    _, sampled = mock_sampler
    sampled.extend(["NNNN", "AC"])

    with pytest.raises(ValueError, match="No full-length A/C/G/T designs were generated"):
        run_evo2_icl(
            Evo2ICLInput(examples=_EXAMPLES),
            Evo2ICLConfig(num_sequences=2, sequence_length=4, device="cpu"),
        )


@pytest.mark.parametrize("sequences", [[], ["AAAA"], ["AAAA", "CCCC", "GGGG"]])
def test_evo2_icl_rejects_wrong_sampler_result_count(mock_sampler, sequences):
    """A malformed sampler response cannot silently change the requested workload."""
    _, sampled = mock_sampler
    sampled.extend(sequences)

    with pytest.raises(ValueError, match=f"Evo 2 returned {len(sequences)} generations for 2 requested designs"):
        run_evo2_icl(
            Evo2ICLInput(examples=_EXAMPLES),
            Evo2ICLConfig(num_sequences=2, sequence_length=4, device="cpu"),
        )


def test_export_fasta_contains_only_generated_designs(mock_sampler, tmp_path):
    """Exported FASTA excludes examples, separators, and invalid continuations."""
    from Bio import SeqIO

    _, sampled = mock_sampler
    sampled.extend(["AAAA", "NCCC", "TTTT"])
    result = run_evo2_icl(
        Evo2ICLInput(examples=_EXAMPLES),
        Evo2ICLConfig(num_sequences=3, sequence_length=4, device="cpu"),
    )

    result.export("icl_designs", export_path=tmp_path, file_format="fasta")

    records = list(SeqIO.parse(tmp_path / "icl_designs.fasta", "fasta"))
    assert [str(record.seq) for record in records] == ["AAAA", "TTTT"]
    assert len({record.id for record in records}) == 2


def test_evo2_icl_generates_one_thousand_designs(mock_sampler):
    """Sixteen 200-bp examples can condition the full allowance with the default batch."""
    calls, sampled = mock_sampler
    sampled.extend(("ACGT"[i % 4] * 200) for i in range(1000))

    result = run_evo2_icl(
        Evo2ICLInput(examples=_TWO_HUNDRED_BP_EXAMPLES),
        Evo2ICLConfig(num_sequences=1000, sequence_length=200, device="cpu"),
    )

    assert len(calls) == 1
    assert len(calls[0][1]["prompts"]) == 1000
    assert set(calls[0][1]["prompts"]) == {"N" + "N".join(_TWO_HUNDRED_BP_EXAMPLES) + "N"}
    assert calls[0][1]["batch_size"] == 1
    assert calls[0][1]["force_prompt_threshold"] == 4096
    assert result.sequences == sampled
    assert len(result.results) == 1000


def test_evo2_icl_full_context_is_preserved_with_long_design(mock_sampler):
    """A full prompt remains intact when conditioning a long design within the work limit."""
    calls, sampled = mock_sampler
    sampled.append("A" * 4096)
    inputs = Evo2ICLInput(examples=["C" * 32_766])

    result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=1, sequence_length=4096, device="cpu"))

    assert len(calls[0][1]["prompts"][0]) == 32_768
    assert calls[0][1]["max_new_tokens"] == 4096
    assert len(result.sequences[0]) == 4096


def test_evo2_icl_work_budget_accepts_boundary_and_rejects_one_extra_prompt_base(mock_sampler):
    """Sequential prompt work shares the hosted budget with newly generated DNA."""
    calls, sampled = mock_sampler
    sampled.extend(["A" * 200] * 1000)
    config = Evo2ICLConfig(num_sequences=1000, sequence_length=200, device="cpu")
    inputs = Evo2ICLInput(examples=["C" * 4894])

    result = run_evo2_icl(inputs, config)

    assert len(inputs.prompt) == 4896
    assert len(result.results) == 1000
    assert len(calls) == 1
    with pytest.raises(ValueError, match="example set is too large for this many designs"):
        run_evo2_icl(Evo2ICLInput(examples=["C" * 4895]), config)
    assert len(calls) == 1, "Over-budget context must be rejected before sampler dispatch"


def test_evo2_icl_maximum_prompt_requires_lower_design_count(mock_sampler):
    """The maximum input supports 30 long designs and rejects 31 without GPU work."""
    calls, sampled = mock_sampler
    sampled.extend(["A" * 4096] * 30)
    inputs = Evo2ICLInput(examples=["C" * 32_766])

    result = run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=30, sequence_length=4096, device="cpu"))

    assert len(result.results) == 30
    assert calls[0][1]["prompts"] == [inputs.prompt] * 30
    with pytest.raises(ValueError, match="example set is too large for this many designs"):
        run_evo2_icl(inputs, Evo2ICLConfig(num_sequences=31, sequence_length=4096, device="cpu"))
    assert len(calls) == 1, "An invalid design count must not start GPU generation"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark("evo2-icl")
@pytest.mark.uses_gpu
def test_evo2_icl_benchmark(request):
    """Benchmark two 200-bp pretrained-7B designs, checking cold and warm worker outputs."""
    inputs = Evo2ICLInput(examples=_TWO_HUNDRED_BP_EXAMPLES)
    config = Evo2ICLConfig(num_sequences=2, sequence_length=200, verbose=False)
    result = benchmark_twice(request, "evo2", lambda: run_evo2_icl(inputs, config))

    assert result.tool_id == "evo2-icl"
    assert result.metadata["model_checkpoint"] == "evo2_7b"
    assert 1 <= len(result.sequences) <= 2
    assert all(len(sequence) == 200 and set(sequence) <= set("ACGT") for sequence in result.sequences)
    assert result.metadata["num_requested"] == 2
    assert result.metadata["num_generated"] == len(result.sequences)
