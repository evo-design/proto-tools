"""tests/causal_models_tests/test_iglm.py.

Tests for IgLM.
"""

import csv
import random
import subprocess

import numpy as np
import pytest

from proto_tools.tools import (
    IgLMInfillConfig,
    IgLMInfillInput,
    IgLMSampleConfig,
    IgLMSampleInput,
    IgLMScoringConfig,
    IgLMScoringInput,
    IgLMSpan,
    run_iglm_infill,
    run_iglm_sample,
    run_iglm_score,
)
from proto_tools.utils import PROTEIN_AMINO_ACIDS
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("iglm")

# 1JPT (humanized anti-tissue-factor Fab D3H44) heavy chain variable domain.
_HEAVY_CHAIN = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATIS"
    "ADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS"
)
# CDR H3 of the heavy chain above: residues 99-106, 1-indexed and inclusive.
_CDR_H3 = "DTAAYFDY"
_CDR_H3_START = 99
_CDR_H3_END = 106

# Trastuzumab kappa light chain variable domain.
_LIGHT_CHAIN = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
# CDR L3 of the light chain above.
_CDR_L3 = "QQHYTTPPT"

# A fixed shuffle of the real heavy chain: same length and residue composition, so a
# scoring gap between the two is attributable to residue *order* alone.
_SCRAMBLED_HEAVY = "".join(random.Random(0).sample(_HEAVY_CHAIN, len(_HEAVY_CHAIN)))


def _error_text(exc: BaseException) -> str:
    """Full text of a dispatch failure, including the standalone subprocess's stderr."""
    parts = (str(exc), getattr(exc, "stderr", None), getattr(exc, "stdout", None))
    return "\n".join(str(part) for part in parts if part)


def _cdr_h3_span() -> IgLMSpan:
    """Fresh span covering the CDR H3 of ``_HEAVY_CHAIN``."""
    return IgLMSpan(start=_CDR_H3_START, end=_CDR_H3_END)


def _loop_variants(parent: str, loop: str, n: int, seed: int) -> list[str]:
    """Build ``n`` distinct chains from ``parent`` by re-writing ``loop`` in place.

    Keeps the real framework and the loop's position and length, so the variants are
    length-matched antibody chains rather than arbitrary protein strings.
    """
    rng = random.Random(seed)
    start = parent.index(loop)
    prefix, suffix = parent[:start], parent[start + len(loop) :]
    variants = {parent}
    while len(variants) < n:
        variants.add(prefix + "".join(rng.choice(PROTEIN_AMINO_ACIDS) for _ in range(len(loop))) + suffix)
    return sorted(variants)


# ── Span validation ──────────────────────────────────────────────────────────


def test_iglm_span_rejects_end_before_start():
    """A span whose end precedes its start is rejected, naming both coordinates."""
    with pytest.raises(ValueError, match=r"end \(98\) must be >= start \(106\)"):
        IgLMSpan(start=106, end=98)


def test_iglm_infill_input_rejects_span_past_sequence_end():
    """A span reaching past its parent sequence is rejected, naming the sequence index."""
    with pytest.raises(ValueError, match="span 99-200 runs past the end of sequence 0 "):
        IgLMInfillInput(sequences=[_HEAVY_CHAIN], spans=[IgLMSpan(start=99, end=200)])


def test_iglm_infill_input_rejects_span_count_mismatch():
    """Infilling pairs one span with one sequence; the error reports both counts."""
    with pytest.raises(ValueError, match="got 1 spans for 2 sequences"):
        IgLMInfillInput(sequences=[_HEAVY_CHAIN, _LIGHT_CHAIN], spans=[_cdr_h3_span()])


def test_iglm_score_input_bounds_checks_spans_after_broadcasting():
    """Scoring broadcasts one span across every sequence, then bounds-checks each pairing."""
    with pytest.raises(ValueError, match="span 99-106 runs past the end of sequence 1 "):
        IgLMScoringInput(sequences=[_HEAVY_CHAIN, _HEAVY_CHAIN[:50]], spans=_cdr_h3_span())


# ── Conditioning broadcast and normalization ─────────────────────────────────


def test_iglm_conditioning_broadcast_lengths():
    """One conditioning value fans out to every item, N values stay 1:1, anything else raises."""
    broadcast = IgLMSampleInput(prompts=["EVQ", "DIQ", "QVQ"], chains=["heavy"], species=["camel"])
    assert broadcast.chains == ["heavy", "heavy", "heavy"]
    assert broadcast.species == ["camel", "camel", "camel"]

    paired = IgLMSampleInput(prompts=["EVQ", "DIQ", "QVQ"], chains=["heavy", "light", "heavy"])
    assert paired.chains == ["heavy", "light", "heavy"]

    with pytest.raises(ValueError, match=r"'chains' has length 2; expected 1 \(broadcast to all\) or 3"):
        IgLMSampleInput(prompts=["EVQ", "DIQ", "QVQ"], chains=["heavy", "light"])


def test_iglm_inputs_normalize_single_values_to_lists():
    """A bare chain, sequence, or span is accepted and normalized to a one-element list."""
    infill = IgLMInfillInput(sequences=_HEAVY_CHAIN, spans=_cdr_h3_span(), chains="heavy")
    assert infill.sequences == [_HEAVY_CHAIN]
    assert infill.spans == [IgLMSpan(start=99, end=106)]
    assert infill.chains == ["heavy"]

    scoring = IgLMScoringInput(sequences=_HEAVY_CHAIN, spans={"start": 99, "end": 106}, species="mouse")
    assert scoring.spans == [IgLMSpan(start=99, end=106)]
    assert scoring.species == ["mouse"]


# ── Dispatch payloads ────────────────────────────────────────────────────────


def test_iglm_infill_dispatch_converts_spans_and_conditioning(monkeypatch):
    """1-indexed inclusive spans reach the worker as 0-indexed half-open ranges, in input order."""
    payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        assert toolkit == "iglm"
        assert config is not None
        payloads.append(payload)
        count = len(payload["sequences"])
        return {
            "sequences": [["MKTLV", "MKTLVVV"] for _ in range(count)],
            "infilled_residues": [["MK", "MKTL"] for _ in range(count)],
        }

    monkeypatch.setattr(
        "proto_tools.tools.causal_models.iglm.iglm_infill.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_iglm_infill(
        IgLMInfillInput(
            sequences=[_HEAVY_CHAIN, _LIGHT_CHAIN],
            spans=[_cdr_h3_span(), IgLMSpan(start=1, end=5)],
            chains=["heavy", "light"],
            species=["human", "camel"],
        ),
        IgLMInfillConfig(seed=11),
    )

    assert result.success
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["sequences"] == [_HEAVY_CHAIN, _LIGHT_CHAIN]
    assert payload["infill_ranges"] == [[98, 106], [0, 5]]
    assert payload["chain_tokens"] == ["[HEAVY]", "[LIGHT]"]
    assert payload["species_tokens"] == ["[HUMAN]", "[CAMEL]"]

    # The converted ranges must slice exactly the residues the caller named.
    start, end = payload["infill_ranges"][0]
    assert _HEAVY_CHAIN[start:end] == _CDR_H3
    light_start, light_end = payload["infill_ranges"][1]
    assert _LIGHT_CHAIN[light_start:light_end] == "DIQMT", "Residue 1 is index 0 and residue 5 is included"

    # Outbound: each design's span starts where the caller's did and ends on its own length,
    # since IgLM picks how many residues to write.
    assert result.results[0].spans == [IgLMSpan(start=99, end=100), IgLMSpan(start=99, end=102)]
    assert result.results[1].spans == [IgLMSpan(start=1, end=2), IgLMSpan(start=1, end=4)]


def test_iglm_score_dispatch_converts_spans_and_conditioning(monkeypatch):
    """Scoring converts spans the same way and sends ``None`` wherever no span was given."""
    payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        assert toolkit == "iglm"
        payloads.append(payload)
        count = len(payload["sequences"])
        metrics = {"log_likelihood": -10.0, "avg_log_likelihood": -1.0, "perplexity": 2.718}
        return {
            "metrics": [dict(metrics) for _ in range(count)],
            "logits": None,
            "vocab": list(PROTEIN_AMINO_ACIDS),
        }

    monkeypatch.setattr(
        "proto_tools.tools.causal_models.iglm.iglm_score.ToolInstance.dispatch",
        fake_dispatch,
    )

    config = IgLMScoringConfig(batch_size=1)

    mixed = run_iglm_score(
        IgLMScoringInput(
            sequences=[_HEAVY_CHAIN, _LIGHT_CHAIN],
            spans=[_cdr_h3_span(), None],
            chains=["heavy", "light"],
            species=["human", "mouse"],
        ),
        config,
    )
    assert mixed.success
    assert payloads[-1]["infill_ranges"] == [[98, 106], None]
    assert payloads[-1]["chain_tokens"] == ["[HEAVY]", "[LIGHT]"]
    assert payloads[-1]["species_tokens"] == ["[HUMAN]", "[MOUSE]"]

    whole = run_iglm_score(
        IgLMScoringInput(sequences=[_HEAVY_CHAIN, _LIGHT_CHAIN], chains=["heavy", "light"]),
        config,
    )
    assert whole.success
    assert payloads[-1]["infill_ranges"] == [None, None]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

# ── Sampling tests ───────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_iglm_sample_generates_distinct_sequences_per_prompt():
    """Each prompt gets its own bundle of ``num_sequences_per_input`` distinct continuations."""
    prompts = [_HEAVY_CHAIN[:16], _LIGHT_CHAIN[:16]]
    result = run_iglm_sample(
        IgLMSampleInput(prompts=prompts, chains=["heavy", "light"], species=["human"]),
        IgLMSampleConfig(num_sequences_per_input=3, seed=0, verbose=False),
    )
    validate_output(result)

    assert result.tool_id == "iglm-sample"
    assert len(result.results) == len(prompts)
    assert len(result.sequences) == 3 * len(prompts)

    for prompt, bundle in zip(prompts, result.results, strict=True):
        assert len(bundle.sequences) == 3
        assert len(set(bundle.sequences)) == 3, "Sequences within a bundle must be distinct"
        for sequence in bundle.sequences:
            assert sequence.startswith(prompt)
            assert len(sequence) > len(prompt)
            assert set(sequence).issubset(set(PROTEIN_AMINO_ACIDS))


@pytest.mark.uses_gpu
def test_iglm_sample_prepend_prompt_controls_the_returned_prefix():
    """``prepend_prompt`` decides whether the prompt is kept on the returned sequence."""
    prompt = _HEAVY_CHAIN[:16]
    inputs = IgLMSampleInput(prompts=[prompt], chains=["heavy"])

    kept = run_iglm_sample(inputs, IgLMSampleConfig(num_sequences_per_input=2, seed=0, verbose=False))
    dropped = run_iglm_sample(
        inputs,
        IgLMSampleConfig(num_sequences_per_input=2, seed=0, prepend_prompt=False, verbose=False),
    )

    assert kept.success and dropped.success
    assert all(sequence.startswith(prompt) for sequence in kept.sequences)
    assert not any(sequence.startswith(prompt) for sequence in dropped.sequences)
    assert all(len(sequence) > 0 for sequence in dropped.sequences)


@pytest.mark.uses_gpu
def test_iglm_sample_from_conditioning_alone():
    """The default empty prompt generates from the chain and species tokens with no prefix."""
    result = run_iglm_sample(
        IgLMSampleInput(chains=["heavy"], species=["human"]),
        IgLMSampleConfig(num_sequences_per_input=2, seed=5, verbose=False),
    )

    assert result.success
    assert len(result.results) == 1
    for sequence in result.results[0].sequences:
        assert len(sequence) > 50, "A heavy chain generated from scratch should be a full variable domain"
        assert set(sequence).issubset(set(PROTEIN_AMINO_ACIDS))


@pytest.mark.uses_gpu
def test_iglm_sample_seeded_reproducibility():
    """The same seed reproduces the same sequences; a different seed diverges."""
    inputs = IgLMSampleInput(prompts=[_HEAVY_CHAIN[:12]], chains=["heavy"])

    first = run_iglm_sample(inputs, IgLMSampleConfig(num_sequences_per_input=2, seed=1234, verbose=False))
    again = run_iglm_sample(inputs, IgLMSampleConfig(num_sequences_per_input=2, seed=1234, verbose=False))
    other = run_iglm_sample(inputs, IgLMSampleConfig(num_sequences_per_input=2, seed=4321, verbose=False))

    assert first.success and again.success and other.success
    assert first.sequences == again.sequences
    assert first.sequences != other.sequences


@pytest.mark.uses_gpu
def test_iglm_sample_export_fasta(tmp_path):
    """A real generation result round-trips to FASTA, one record per generated sequence."""
    result = run_iglm_sample(
        IgLMSampleInput(prompts=[_HEAVY_CHAIN[:12], _LIGHT_CHAIN[:12]], chains=["heavy", "light"]),
        IgLMSampleConfig(num_sequences_per_input=2, seed=0, verbose=False),
    )
    assert result.success

    result.export(name="iglm_designs", export_path=tmp_path, file_format="fasta")
    lines = (tmp_path / "iglm_designs.fasta").read_text().splitlines()

    assert [line for line in lines if line.startswith(">")] == [
        ">input_0_seq_0",
        ">input_0_seq_1",
        ">input_1_seq_0",
        ">input_1_seq_1",
    ]
    assert [line for line in lines if not line.startswith(">")] == result.sequences


# ── Infilling tests ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_iglm_infill_rewrites_the_span_and_keeps_the_framework():
    """Designs keep the parent's prefix and suffix, and the loop between them is re-written."""
    result = run_iglm_infill(
        IgLMInfillInput(sequences=[_HEAVY_CHAIN], spans=[_cdr_h3_span()], chains=["heavy"]),
        IgLMInfillConfig(num_sequences_per_input=4, seed=0, verbose=False),
    )
    validate_output(result)

    assert result.tool_id == "iglm-infill"
    assert len(result.results) == 1

    bundle = result.results[0]
    prefix, suffix = _HEAVY_CHAIN[: _CDR_H3_START - 1], _HEAVY_CHAIN[_CDR_H3_END:]
    assert len(bundle.sequences) == 4
    assert len(bundle.infilled_residues) == len(bundle.sequences)
    assert len(bundle.spans) == len(bundle.sequences)

    for sequence, infilled, span in zip(bundle.sequences, bundle.infilled_residues, bundle.spans, strict=True):
        assert sequence.startswith(prefix), "Framework before the span must survive re-design"
        assert sequence.endswith(suffix), "Framework after the span must survive re-design"
        assert sequence == prefix + infilled + suffix
        assert set(infilled).issubset(set(PROTEIN_AMINO_ACIDS))
        # The reported span must locate the written residues in the design it belongs to.
        assert sequence[span.start - 1 : span.end] == infilled
        assert span.start == _CDR_H3_START

    # IgLM picks the span length, so designs are not pinned to the 8 residues they replaced.
    assert {len(infilled) for infilled in bundle.infilled_residues} != {len(_CDR_H3)}, (
        "Re-design should explore loop lengths other than the native one"
    )
    assert sum(infilled == _CDR_H3 for infilled in bundle.infilled_residues) <= 1


@pytest.mark.uses_gpu
def test_iglm_infill_rejects_designs_that_ran_out_of_max_length():
    """A budget too small to finish the span fails loudly instead of returning a clipped loop.

    ``forced_eos_token_id`` puts the terminator at the last budgeted position, so a
    generation that ran out of room still passes upstream's well-formedness check — for
    infilling, where the mask and separator come from the prompt, it always does. Without
    the force-termination check those candidates come back as two-residue "designs".
    """
    # One-shot dispatch re-raises as CalledProcessError and the persistent worker as
    # RuntimeError, both carrying the standalone's text, so the message is what is asserted.
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)) as excinfo:
        run_iglm_infill(
            IgLMInfillInput(sequences=[_HEAVY_CHAIN], spans=[_cdr_h3_span()]),
            # The masked parent alone is 112 tokens, leaving almost no room for the span.
            IgLMInfillConfig(num_sequences_per_input=3, max_length=115, seed=0, verbose=False),
        )

    message = _error_text(excinfo.value)
    assert "ran out of max_length" in message
    assert "raise max_length" in message


@pytest.mark.uses_gpu
def test_iglm_infill_designs_feed_straight_into_scoring():
    """The spans an infill run reports are accepted as-is by iglm-score, one score per design."""
    designs = run_iglm_infill(
        IgLMInfillInput(sequences=[_HEAVY_CHAIN], spans=[_cdr_h3_span()], chains=["heavy"]),
        IgLMInfillConfig(num_sequences_per_input=3, seed=7, verbose=False),
    )
    assert designs.success
    bundle = designs.results[0]

    scored = run_iglm_score(
        IgLMScoringInput(sequences=bundle.sequences, spans=bundle.spans, chains=["heavy"]),
        IgLMScoringConfig(verbose=False),
    )
    validate_output(scored)
    assert_metrics_in_spec(scored)

    assert len(scored.scores) == len(bundle.sequences)
    for score in scored.scores:
        assert score["perplexity"] >= 1.0


# ── Scoring tests ────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_iglm_score_prefers_a_real_heavy_chain_over_a_scrambled_one():
    """A real human heavy chain scores better than a length-matched shuffle of itself."""
    result = run_iglm_score(
        IgLMScoringInput(sequences=[_HEAVY_CHAIN, _SCRAMBLED_HEAVY], chains=["heavy"], species=["human"]),
        IgLMScoringConfig(verbose=False),
    )
    validate_output(result)
    assert_metrics_in_spec(result)

    assert result.tool_id == "iglm-score"
    assert len(result.scores) == 2

    for score in result.scores:
        assert score["perplexity"] >= 1.0
        assert score["avg_log_likelihood"] <= 0.0
        np.testing.assert_allclose(score["perplexity"], np.exp(-score["avg_log_likelihood"]), rtol=1e-5)

    native, scrambled = result.scores
    assert native["perplexity"] < scrambled["perplexity"], (
        f"Native chain should be more antibody-like than its shuffle: "
        f"{native['perplexity']} vs {scrambled['perplexity']}"
    )
    assert native["log_likelihood"] > scrambled["log_likelihood"]


@pytest.mark.uses_gpu
def test_iglm_score_span_conditional_differs_from_whole_sequence():
    """Scoring a span conditionally scores fewer tokens than scoring the whole chain."""
    config = IgLMScoringConfig(verbose=False)
    whole = run_iglm_score(IgLMScoringInput(sequences=[_HEAVY_CHAIN], chains=["heavy"]), config)
    span = run_iglm_score(
        IgLMScoringInput(sequences=[_HEAVY_CHAIN], spans=[_cdr_h3_span()], chains=["heavy"]),
        config,
    )

    assert whole.success and span.success
    assert_metrics_in_spec(whole)
    assert_metrics_in_spec(span)

    assert span.scores[0]["avg_log_likelihood"] != whole.scores[0]["avg_log_likelihood"]
    assert span.scores[0]["log_likelihood"] > whole.scores[0]["log_likelihood"], (
        "The span sums over ~8 residues while the whole chain sums over ~115"
    )


@pytest.mark.uses_gpu
def test_iglm_score_is_invariant_to_batching():
    """Right padding must not leak: a mixed-length batch scores each chain as if alone."""
    inputs = IgLMScoringInput(sequences=[_HEAVY_CHAIN, _LIGHT_CHAIN], chains=["heavy", "light"])

    batched = run_iglm_score(inputs, IgLMScoringConfig(batch_size=8))
    unbatched = run_iglm_score(inputs, IgLMScoringConfig(batch_size=1))

    for batched_score, single_score in zip(batched.scores, unbatched.scores, strict=True):
        assert batched_score["avg_log_likelihood"] == pytest.approx(single_score["avg_log_likelihood"], abs=1e-5)
        assert batched_score["perplexity"] == pytest.approx(single_score["perplexity"], rel=1e-5)


@pytest.mark.uses_gpu
def test_iglm_score_export_csv(tmp_path):
    """A real scoring result round-trips to CSV, one row of scalar metrics per sequence."""
    result = run_iglm_score(
        IgLMScoringInput(sequences=[_HEAVY_CHAIN, _LIGHT_CHAIN], chains=["heavy", "light"]),
        IgLMScoringConfig(verbose=False),
    )
    assert result.success

    result.export(name="iglm_scores", export_path=tmp_path, file_format="csv")
    rows = list(csv.DictReader((tmp_path / "iglm_scores.csv").read_text().splitlines()))

    assert len(rows) == 2
    assert {"log_likelihood", "avg_log_likelihood", "perplexity"}.issubset(rows[0])
    for row, score in zip(rows, result.scores, strict=True):
        np.testing.assert_allclose(float(row["perplexity"]), score["perplexity"], rtol=1e-6)


# ── Benchmarks ───────────────────────────────────────────────────────────────


@pytest.mark.benchmark("iglm-sample")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_iglm_sample_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark iglm-sample on 20 real antibody N-terminal prompts x 5 sequences (cold + warm)."""
    heavy_prompts = [_HEAVY_CHAIN[:length] for length in range(3, 23, 2)]
    light_prompts = [_LIGHT_CHAIN[:length] for length in range(3, 23, 2)]
    prompts = heavy_prompts + light_prompts
    inputs = IgLMSampleInput(
        prompts=prompts,
        chains=["heavy"] * len(heavy_prompts) + ["light"] * len(light_prompts),
    )
    config = IgLMSampleConfig(num_sequences_per_input=5, batch_size=32, seed=0, verbose=False)

    result = benchmark_twice(request, "iglm", lambda: run_iglm_sample(inputs=inputs, config=config))

    assert result.success
    assert result.tool_id == "iglm-sample"
    assert len(result.results) == len(prompts)
    assert len(result.sequences) == 5 * len(prompts)
    for sequence in result.sequences:
        assert len(sequence) > 0, "Generated sequence should be non-empty"


@pytest.mark.benchmark("iglm-infill")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_iglm_infill_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark iglm-infill re-designing CDR H3 on 24 parent heavy chains x 4 designs (cold + warm)."""
    parents = _loop_variants(_HEAVY_CHAIN, _CDR_H3, n=24, seed=0)
    inputs = IgLMInfillInput(
        sequences=parents,
        spans=[_cdr_h3_span() for _ in parents],
        chains=["heavy"],
    )
    config = IgLMInfillConfig(num_sequences_per_input=4, batch_size=32, seed=0, verbose=False)

    result = benchmark_twice(request, "iglm", lambda: run_iglm_infill(inputs=inputs, config=config))

    assert result.success
    assert result.tool_id == "iglm-infill"
    assert len(result.results) == len(parents)
    for bundle in result.results:
        assert len(bundle.sequences) == 4
        assert len(bundle.infilled_residues) == 4


@pytest.mark.benchmark("iglm-score")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_iglm_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark iglm-score on 200 antibody chains: 100 heavy and 100 light variants (cold + warm)."""
    heavy = _loop_variants(_HEAVY_CHAIN, _CDR_H3, n=100, seed=1)
    light = _loop_variants(_LIGHT_CHAIN, _CDR_L3, n=100, seed=2)
    inputs = IgLMScoringInput(
        sequences=heavy + light,
        chains=["heavy"] * len(heavy) + ["light"] * len(light),
    )
    config = IgLMScoringConfig(batch_size=32, return_logits=False, verbose=False)

    result = benchmark_twice(request, "iglm", lambda: run_iglm_score(inputs=inputs, config=config))
    assert_metrics_in_spec(result)

    assert result.success
    assert result.tool_id == "iglm-score"
    assert len(result.scores) == 200
    for score in result.scores:
        assert score["perplexity"] >= 1.0
