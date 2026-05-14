"""E2E tests for the stochastic-iterable cache/dedup routing in @tool.

Exercises the framework's behavior against three CPU-only mocks
(batched stochastic, serial stochastic, batched deterministic) so the
routing logic can be verified end-to-end in milliseconds without GPU
or model load.
"""

import pytest

from proto_tools.tools.testing.mock_iterable_deterministic import (
    MockIterableDeterministicConfig,
    MockIterableDeterministicInput,
    run_mock_iterable_deterministic,
)
from proto_tools.tools.testing.mock_iterable_stochastic import (
    MockIterableStochasticConfig,
    MockIterableStochasticInput,
    run_mock_iterable_stochastic,
)
from proto_tools.tools.testing.mock_iterable_stochastic_serial import (
    MockIterableStochasticSerialConfig,
    MockIterableStochasticSerialInput,
    run_mock_iterable_stochastic_serial,
)
from proto_tools.utils.tool_cache import ToolCache, _program_tool_cache


@pytest.fixture
def fresh_cache():
    """Install a fresh program-scoped cache for the duration of the test."""
    cache = ToolCache()
    token = _program_tool_cache.set(cache)
    yield cache
    _program_tool_cache.reset(token)


# ── stochastic + iterable + duplicates ──────────────────────────────────────


def test_stochastic_duplicates_bypass_dedup_and_diverge(fresh_cache):
    """[p, p, p] + seed=42 produces three pairwise-distinct outputs, reproducibly.

    Verifies: (1) dedup is skipped for stochastic iterables — tool sees
    all 3 items; (2) per-item RNG advancement diversifies them in the
    same call; (3) a fresh recompute with the same seed gives the same
    triple.
    """
    inputs = MockIterableStochasticInput(prompts=["p", "p", "p"])
    config = MockIterableStochasticConfig(seed=42)

    r1 = run_mock_iterable_stochastic(inputs, config)
    fresh_cache.clear()  # force a real recompute on the second call
    r2 = run_mock_iterable_stochastic(inputs, config)

    # Tool function received all 3 items (no dedup collapse upstream).
    assert r1.items_processed == 3
    assert r2.items_processed == 3

    # Per-item RNG advancement → three distinct completions.
    assert len(r1.completions) == 3
    assert len(set(r1.completions)) == 3, f"Expected 3 distinct completions, got {r1.completions}"

    # Same seed across independent runs → identical outputs.
    assert r1.completions == r2.completions
    assert r1 is not r2, "cache was cleared, so r2 must be a fresh object"


def test_stochastic_duplicates_seeded_repeat_hits_whole_call_cache(fresh_cache):
    """Repeat call with same input + same seed returns the cached object.

    Verifies stochastic iterables use the whole-call cache when seeded.
    Distinguishes "cache hit" (same object identity) from "deterministic
    recompute" (different object, same content) — the previous test
    covers the latter.
    """
    inputs = MockIterableStochasticInput(prompts=["p", "p", "p"])
    config = MockIterableStochasticConfig(seed=42)

    r1 = run_mock_iterable_stochastic(inputs, config)
    r2 = run_mock_iterable_stochastic(inputs, config)

    assert r2 is r1, "second call should return the cached object"
    assert r1.completions == r2.completions


def test_stochastic_duplicates_unseeded_skips_cache(fresh_cache):
    """seed=None disables cache entirely — repeat calls produce different results."""
    inputs = MockIterableStochasticInput(prompts=["p", "p", "p"])
    config = MockIterableStochasticConfig()  # seed=None

    r1 = run_mock_iterable_stochastic(inputs, config)
    r2 = run_mock_iterable_stochastic(inputs, config)

    # Each call diverges internally — all 3 completions distinct.
    assert len(set(r1.completions)) == 3
    assert len(set(r2.completions)) == 3
    # The two calls produce different result sets (cache skipped).
    assert r1.completions != r2.completions, "seed=None must bypass cache; repeat call should not hit"


# ── stochastic + iterable + unique items ────────────────────────────────────


def test_stochastic_unique_items_seeded(fresh_cache):
    """[p1, p2, p3] + seed=42 produces 3 outputs, reproducible across recomputes."""
    inputs = MockIterableStochasticInput(prompts=["aa", "bb", "cc"])
    config = MockIterableStochasticConfig(seed=42)

    r1 = run_mock_iterable_stochastic(inputs, config)
    fresh_cache.clear()
    r2 = run_mock_iterable_stochastic(inputs, config)

    assert r1.items_processed == 3
    assert len(r1.completions) == 3
    # Reproducible across independent runs of the tool's algorithm.
    assert r1.completions == r2.completions


# ── deterministic + iterable + duplicates ───────────────────────────────────


def test_deterministic_duplicates_collapse_via_dedup(fresh_cache):
    """[p, p, p] for a deterministic tool: dedup collapses to [p], stitch expands."""
    inputs = MockIterableDeterministicInput(prompts=["p", "p", "p"])
    config = MockIterableDeterministicConfig()

    r1 = run_mock_iterable_deterministic(inputs, config)
    fresh_cache.clear()
    r2 = run_mock_iterable_deterministic(inputs, config)

    # Dedup collapsed [p, p, p] → [p] before the tool function ran.
    assert r1.items_processed == 1
    assert r2.items_processed == 1

    # Framework expanded the single result back to 3 entries — same token in every slot.
    assert len(r1.scores) == 3
    assert all(s == r1.scores[0] for s in r1.scores), f"Expected 3 copies, got {r1.scores}"

    # Reproducible across independent recomputes.
    assert r1.scores == r2.scores


def test_deterministic_unique_items_pass_through(fresh_cache):
    """[p1, p2, p3] for a deterministic tool: no dedup needed, tool sees all 3."""
    inputs = MockIterableDeterministicInput(prompts=["aa", "bbb", "cccc"])
    config = MockIterableDeterministicConfig()

    r1 = run_mock_iterable_deterministic(inputs, config)
    fresh_cache.clear()
    r2 = run_mock_iterable_deterministic(inputs, config)

    assert r1.items_processed == 3
    assert len(r1.scores) == 3
    assert r1.scores == r2.scores  # argmax decoding reproduces bit-exactly


# ── internal batching invariant ─────────────────────────────────────────────


def test_internal_batching_invariant_across_batch_sizes(fresh_cache):
    """Same seed + same prompts → same completions, regardless of batch_size.

    The tool's RNG advances linearly across all items in the call,
    independent of how it chunks them internally. If the framework
    re-seeded between batches (which it must not), outputs would change
    with batch_size.
    """
    prompts = ["a", "b", "c", "d", "e"]
    inputs = MockIterableStochasticInput(prompts=prompts)

    r_bs1 = run_mock_iterable_stochastic(
        inputs,
        MockIterableStochasticConfig(seed=42, batch_size=1),
    )
    r_bs5 = run_mock_iterable_stochastic(
        inputs,
        MockIterableStochasticConfig(seed=42, batch_size=5),
    )

    # Different batch_size → different cache key, both calls actually ran.
    assert r_bs1.completions == r_bs5.completions, (
        f"Internal batching should not change outputs for the same seed.\n"
        f"  bs=1: {r_bs1.completions}\n"
        f"  bs=5: {r_bs5.completions}"
    )


def test_internal_batched_duplicates_diverge_within_batch(fresh_cache):
    """Duplicates inside one internal batch still diverge.

    With batch_size=5 and 5 identical prompts, all 5 are processed in
    one internal batch yet still produce distinct outputs because the
    per-item ``rng.choice`` call advances RNG between batch elements.
    This is the analogue of ``torch.multinomial`` advancing the global
    RNG state between batch positions inside a single forward pass.

    A pure dedup-collapse bug would produce ``["p_X"] * 5``. A
    per-item-unroll bug would produce 5 outputs with different seeds.
    The expected behavior is one seed → linear RNG advance → mostly
    distinct draws (with rare vocab collisions tolerated).
    """
    inputs = MockIterableStochasticInput(prompts=["p"] * 5)
    config = MockIterableStochasticConfig(seed=42, batch_size=5)

    r1 = run_mock_iterable_stochastic(inputs, config)
    fresh_cache.clear()
    r2 = run_mock_iterable_stochastic(inputs, config)

    assert r1.items_processed == 5
    assert len(set(r1.completions)) == 5, f"Expected 5 distinct completions, got {r1.completions}"
    assert r1.completions == r2.completions  # reproducible


# ── stochastic + iterable + NO internal batching (serial) ───────────────────


def test_stochastic_serial_duplicates_bypass_dedup_and_diverge(fresh_cache):
    """Serial stochastic tool: [p, p, p] + seed=42 still diverges, no batching needed.

    The framework's stochastic-iterable routing must work the same for
    tools that batch internally and tools that don't. Mirrors
    ProGen3-sample's structure: a pure ``for prompt in prompts`` loop
    where per-item RNG advancement happens between iterations.
    """
    inputs = MockIterableStochasticSerialInput(prompts=["p", "p", "p"])
    config = MockIterableStochasticSerialConfig(seed=42)

    r1 = run_mock_iterable_stochastic_serial(inputs, config)
    fresh_cache.clear()
    r2 = run_mock_iterable_stochastic_serial(inputs, config)

    # Tool function received all 3 items (no dedup collapse upstream).
    assert r1.items_processed == 3
    assert r2.items_processed == 3

    # Per-item RNG advancement → three distinct completions.
    assert len(set(r1.completions)) == 3, f"Expected 3 distinct, got {r1.completions}"

    # Same seed across independent runs → identical outputs.
    assert r1.completions == r2.completions


def test_stochastic_serial_seeded_repeat_hits_whole_call_cache(fresh_cache):
    """Serial stochastic tool: repeat call with same seed returns cached object."""
    inputs = MockIterableStochasticSerialInput(prompts=["a", "b", "c"])
    config = MockIterableStochasticSerialConfig(seed=42)

    r1 = run_mock_iterable_stochastic_serial(inputs, config)
    r2 = run_mock_iterable_stochastic_serial(inputs, config)

    assert r2 is r1, "second call should return the cached object"
    assert r1.completions == r2.completions


def test_stochastic_serial_unseeded_skips_cache(fresh_cache):
    """Serial stochastic tool: seed=None disables cache."""
    inputs = MockIterableStochasticSerialInput(prompts=["p", "p", "p"])
    config = MockIterableStochasticSerialConfig()  # seed=None

    r1 = run_mock_iterable_stochastic_serial(inputs, config)
    r2 = run_mock_iterable_stochastic_serial(inputs, config)

    # Each call diverges internally — all 3 completions distinct.
    assert len(set(r1.completions)) == 3
    assert len(set(r2.completions)) == 3
    # Calls differ from each other (no cache reuse).
    assert r1.completions != r2.completions


def test_stochastic_serial_matches_batched_output_for_same_seed(fresh_cache):
    """Serial and batched mocks produce identical outputs for the same prompts + seed.

    Both implementations share the contract: one ``random.Random(seed)``
    stream, one ``rng.choice`` per item. The presence or absence of an
    internal batch loop is an implementation detail invisible to the
    RNG. This is the strongest possible check that the framework's
    routing is consistent across these two real-world tool shapes.
    """
    prompts = ["a", "b", "a", "c", "a"]

    r_batched = run_mock_iterable_stochastic(
        MockIterableStochasticInput(prompts=prompts),
        MockIterableStochasticConfig(seed=42, batch_size=2),
    )
    r_serial = run_mock_iterable_stochastic_serial(
        MockIterableStochasticSerialInput(prompts=prompts),
        MockIterableStochasticSerialConfig(seed=42),
    )

    assert r_batched.completions == r_serial.completions, (
        f"Batched vs serial tools should produce identical output for the same seed.\n"
        f"  batched: {r_batched.completions}\n"
        f"  serial:  {r_serial.completions}"
    )


# ── coverage-gap fills ──────────────────────────────────────────────────────


def test_stochastic_unique_seeded_repeat_hits_whole_call_cache(fresh_cache):
    """Stochastic + unique iterable items + seed=42: repeat call hits whole-call cache.

    Parallel to ``test_stochastic_duplicates_seeded_repeat_hits_whole_call_cache``
    but with unique inputs — confirms that the whole-call cache works
    for unique items as well as duplicates. The cache key is computed
    from the full input list, so the second call hits regardless of
    whether the list has duplicates.
    """
    inputs = MockIterableStochasticInput(prompts=["aa", "bb", "cc"])
    config = MockIterableStochasticConfig(seed=42)

    r1 = run_mock_iterable_stochastic(inputs, config)
    r2 = run_mock_iterable_stochastic(inputs, config)

    assert r2 is r1, "second call should return the cached object"
    assert r1.completions == r2.completions


def test_stochastic_unique_items_unseeded_skips_cache(fresh_cache):
    """Stochastic + unique iterable items + seed=None: each call produces different results.

    Mirrors ``test_stochastic_duplicates_unseeded_skips_cache`` but with
    unique inputs — verifies the cache-skip-when-unseeded rule applies
    regardless of whether the iterable has duplicates.
    """
    inputs = MockIterableStochasticInput(prompts=["aa", "bb", "cc"])
    config = MockIterableStochasticConfig()  # seed=None

    r1 = run_mock_iterable_stochastic(inputs, config)
    r2 = run_mock_iterable_stochastic(inputs, config)

    # Each call diverges internally.
    assert len(set(r1.completions)) >= 2
    # The two calls produce different result sets (cache skipped).
    assert r1.completions != r2.completions


def test_stochastic_mixed_duplicates_with_internal_batching(fresh_cache):
    """[s1, s1, s2, s3, s4] + seed=42 + batch_size=2: all 5 items reach the tool, duplicates diverge.

    This is the literal example from the seeding.md "Iterable input —
    tool's internal batching" diagram. The two ``s1`` positions are
    fed to the tool together (in the first internal batch), the tool's
    per-item RNG advancement makes them diverge, and the full result is
    reproducible across recomputes.
    """
    inputs = MockIterableStochasticInput(prompts=["s1", "s1", "s2", "s3", "s4"])
    config = MockIterableStochasticConfig(seed=42, batch_size=2)

    r1 = run_mock_iterable_stochastic(inputs, config)
    fresh_cache.clear()
    r2 = run_mock_iterable_stochastic(inputs, config)

    # No dedup collapse — all 5 items reached the tool.
    assert r1.items_processed == 5
    assert len(r1.completions) == 5

    # The duplicate s1 prompts at positions 0/1 produce different completions.
    assert r1.completions[0] != r1.completions[1], f"Duplicates should diverge, got {r1.completions}"

    assert r1.completions == r2.completions  # reproducible


def test_cache_key_isolates_by_input_content(fresh_cache):
    """Different prompts with same seed produce non-colliding cache entries.

    The whole-call cache key includes the full input iterable, so two
    calls with different prompts must NOT falsely hit each other's
    cache entry — even when seed and config are identical. Different
    prompts also yield different prompt-derived logits, so the sampled
    tokens differ as well.
    """
    config = MockIterableStochasticConfig(seed=42)

    r_aaa = run_mock_iterable_stochastic(
        MockIterableStochasticInput(prompts=["aa", "aa", "aa"]),
        config,
    )
    r_bbb = run_mock_iterable_stochastic(
        MockIterableStochasticInput(prompts=["bb", "bb", "bb"]),
        config,
    )

    # Distinct cache entries — not the same object, different content.
    assert r_aaa is not r_bbb
    assert r_aaa.completions != r_bbb.completions
