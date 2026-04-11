"""tests/masked_models_tests/masking_tests/test_masking_strategies.py.

Tests for masking strategies.
"""

import logging
import warnings

import pytest
from pydantic import ValidationError

from proto_tools.tools.masked_models.masking import (
    MaskingStrategy,
    apply_masking_strategy,
)
from proto_tools.tools.masked_models.masking.base import (
    MASK_TOKEN,
    _resolve_count,
    mutable_mask,
    weighted_sample,
)
from proto_tools.tools.masked_models.masking.maskers import MASKERS
from proto_tools.tools.masked_models.shared_data_models import (
    MaskedModelInput,
)

# CPU-testable strategy instances for shared invariant tests
CPU_STRATEGIES = [
    pytest.param(MaskingStrategy(), id="default"),
    pytest.param(MaskingStrategy(num_mutations=3), id="num_mutations"),
    pytest.param(MaskingStrategy(mask_fraction=0.5), id="mask_fraction"),
]


# ── Shared invariants (parameterized over strategies) ─────────────────────────


def test_fixed_positions_never_violated():
    """No strategy should ever mutate a fixed position."""
    seq = "ABCDEFGHIJ"
    fixed = [1, 3, 5]
    for strategy_factory in [
        lambda: MaskingStrategy(fixed_positions=fixed),
        lambda: MaskingStrategy(num_mutations=3, fixed_positions=fixed),
        lambda: MaskingStrategy(mask_fraction=0.5, fixed_positions=fixed),
    ]:
        result = strategy_factory().mask([seq])
        for pos in fixed:
            assert result[0][pos - 1] == seq[pos - 1]


@pytest.mark.parametrize("strategy", CPU_STRATEGIES)
def test_output_length_matches_input(strategy):
    """Every strategy must preserve sequence length."""
    seq = "ABCDEFGHIJ"
    result = strategy.mask([seq])
    assert len(result[0]) == len(seq)


@pytest.mark.parametrize("strategy", CPU_STRATEGIES)
def test_only_mask_tokens_introduced(strategy):
    """Masked positions become '_'; non-masked positions keep original character."""
    seq = "ABCDEFGHIJ"
    result = strategy.mask([seq])
    for orig, out in zip(seq, result[0], strict=False):
        assert out in (orig, MASK_TOKEN)


@pytest.mark.parametrize("strategy", CPU_STRATEGIES)
def test_batch_size_preserved(strategy):
    """Output list length must equal input list length."""
    seqs = ["ABCDEF", "GHIJKLMNOP", "XYZ"]
    result = strategy.mask(seqs)
    assert len(result) == len(seqs)


@pytest.mark.parametrize("strategy", CPU_STRATEGIES)
def test_existing_masks_never_unmasked(strategy):
    """Positions that are already '_' in the input must stay '_' in the output."""
    seq = "AB_DE_GH"
    result = strategy.mask([seq])
    for i, c in enumerate(seq):
        if c == MASK_TOKEN:
            assert result[0][i] == MASK_TOKEN


# ── Mask-all behavior ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seq, fixed, expected",
    [
        pytest.param("ABCDE", None, "_____", id="no_fixed"),
        pytest.param("ABCDE", [1, 3, 5], "A_C_E", id="some_fixed"),
        pytest.param("ABCDEF", [], "______", id="empty_fixed"),
        pytest.param("ABC", [1, 2, 3], "ABC", id="all_fixed"),
        pytest.param("A_C", None, "___", id="existing_masks"),
    ],
)
def test_mask_all(seq, fixed, expected):
    """mask_fraction=1.0 masks every designable position."""
    result = MaskingStrategy(
        mask_fraction=1.0,
        fixed_positions=fixed,
    ).mask([seq])
    assert result[0] == expected


def test_no_designable_positions_returns_unchanged():
    """When all positions are fixed, default masking returns the input unchanged."""
    result = MaskingStrategy(fixed_positions=[1, 2, 3]).mask(["ABC"])
    assert result == ["ABC"]


def test_mask_all_batch():
    """Batch masking applies the same fixed_positions uniformly to all sequences."""
    result = MaskingStrategy(
        mask_fraction=1.0,
        fixed_positions=[1],
    ).mask(["ABC", "DEFGH"])
    assert result[0] == "A__"
    assert result[1] == "D____"


# ── mutable_mask helper ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seq, fixed, expected",
    [
        pytest.param("ABCDE", None, [True, True, True, True, True], id="all_mutable"),
        pytest.param("A_C_E", None, [True, False, True, False, True], id="existing_masks"),
        pytest.param("ABCDE", [2, 4], [True, False, True, False, True], id="fixed_only"),
        pytest.param("A_CDE", [4], [True, False, True, False, True], id="masks_and_fixed"),
    ],
)
def test_mutable_mask(seq, fixed, expected):
    """mutable_mask returns True for positions that are designable (not '_' and not fixed)."""
    assert mutable_mask(seq, fixed) == expected


# ── Count resolution ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "num_mutations, mask_fraction, n_designable, expected",
    [
        pytest.param(None, None, 100, 30, id="default_30pct"),
        pytest.param(5, None, 100, 5, id="exact"),
        pytest.param(None, 0.15, 100, 15, id="fraction_15pct"),
        pytest.param(None, 0.15, 3, 1, id="fraction_rounds_to_min_1"),
        pytest.param(None, 0.01, 10, 1, id="tiny_fraction_clamps_to_1"),
    ],
)
def test_resolve_count(num_mutations, mask_fraction, n_designable, expected):
    """_resolve_count picks num_mutations > mask_fraction > default(30%)."""
    assert _resolve_count(num_mutations, mask_fraction, n_designable) == expected


def test_num_mutations_and_mask_fraction_mutually_exclusive():
    """Setting both num_mutations and mask_fraction raises ValueError."""
    with pytest.raises(ValueError, match="not both"):
        MaskingStrategy(num_mutations=3, mask_fraction=0.5)


def test_mask_fraction_applies_to_designable_not_full_length():
    """mask_fraction scales by designable count, not total sequence length."""
    result = MaskingStrategy(
        mask_fraction=0.5,
        fixed_positions=[1, 2, 3, 4],
    ).mask(["ABCDEFGHIJ"])
    # 10 chars, 4 fixed -> 6 designable -> round(6 * 0.5) = 3 masks
    assert result[0].count("_") == 3
    assert result[0][:4] == "ABCD"


# ── MaskingStrategy count behavior ──────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, seq, expected_count",
    [
        pytest.param({}, "ABCDEFGHIJ", 3, id="default_30pct"),
        pytest.param({"num_mutations": 3}, "ABCDEFGHIJ", 3, id="num_mutations"),
        pytest.param({"mask_fraction": 0.5}, "ABCDEFGHIJ", 5, id="mask_fraction"),
    ],
)
def test_masking_count(kwargs, seq, expected_count):
    """MaskingStrategy produces the correct number of '_' tokens."""
    result = MaskingStrategy(**kwargs).mask([seq])
    assert result[0].count("_") == expected_count
    assert len(result[0]) == len(seq)


@pytest.mark.parametrize(
    "num_mutations, seq, fixed",
    [
        pytest.param(10, "ABC", None, id="exceeds_length"),
        pytest.param(4, "ABCDE", [1, 2], id="exceeds_designable"),
    ],
)
def test_too_many_mutations_raises(num_mutations, seq, fixed):
    """Requesting more mutations than designable positions raises ValueError."""
    with pytest.raises(ValueError, match="mutable positions"):
        MaskingStrategy(
            num_mutations=num_mutations,
            fixed_positions=fixed,
        ).mask([seq])


def test_variable_length_batch_with_fraction():
    """mask_fraction adapts to each sequence's designable count independently."""
    result = MaskingStrategy(mask_fraction=0.5).mask(["ABCD", "ABCDEFGHIJ"])
    assert result[0].count("_") == 2  # round(4 * 0.5) = 2
    assert result[1].count("_") == 5  # round(10 * 0.5) = 5


# ── Sampling helpers ─────────────────────────────────────────────────────────


def test_weighted_sample_basic():
    """Uniform weights produce k unique samples from the eligible set."""
    result = weighted_sample([0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0], 2)
    assert len(result) == 2
    assert len(set(result)) == 2
    assert all(r in [0, 1, 2, 3] for r in result)


def test_weighted_sample_k_equals_n():
    """When k == len(eligible), all positions are returned."""
    assert set(weighted_sample([5, 10], [0.5, 0.5], 2)) == {5, 10}


def test_weighted_sample_heavily_weighted():
    """Extreme weight differences make sampling deterministic."""
    results = [weighted_sample([0, 1, 2], [100.0, -100.0, -100.0], 1) for _ in range(50)]
    assert all(r == [0] for r in results)


# ── apply_masking_strategy helper ────────────────────────────────────────────


class _MockConfig:
    """Minimal config-like object for testing apply_masking_strategy."""

    def __init__(self, masking_strategy, device="cuda", seed=None):
        self.masking_strategy = masking_strategy
        self.device = device
        self.seed = seed


def test_apply_masking_strategy_masks_unmasked_sequences():
    """Applies masking strategy when no '_' tokens are present."""
    config = _MockConfig(MaskingStrategy(num_mutations=2))
    inputs = MaskedModelInput(sequences=["MKTLLIFLA"])
    result = apply_masking_strategy(config, inputs)
    assert result.sequences[0].count("_") == 2
    assert len(result.sequences[0]) == len("MKTLLIFLA")


def test_apply_masking_strategy_skips_premasked():
    """Leaves sequences unchanged when they already contain '_'."""
    config = _MockConfig(MaskingStrategy(num_mutations=5))
    inputs = MaskedModelInput(sequences=["MKT_LIFLA"])
    result = apply_masking_strategy(config, inputs)
    assert result.sequences == ["MKT_LIFLA"]


def test_apply_masking_strategy_warns_custom_strategy_ignored(caplog):
    """Warns when a non-default strategy is ignored due to pre-masking."""
    config = _MockConfig(MaskingStrategy(num_mutations=3))
    inputs = MaskedModelInput(sequences=["MKT_LIFLA"])
    with caplog.at_level(logging.WARNING):
        apply_masking_strategy(config, inputs)
    assert "ignoring custom masking_strategy" in caplog.text


def test_apply_masking_strategy_no_warning_default_strategy(caplog):
    """No warning when default strategy is used with pre-masked input."""
    config = _MockConfig(MaskingStrategy())
    inputs = MaskedModelInput(sequences=["MKT_LIFLA"])
    with caplog.at_level(logging.WARNING):
        apply_masking_strategy(config, inputs)
    assert "ignoring" not in caplog.text


def test_apply_masking_strategy_does_not_mutate_original():
    """Returns a new inputs object; original is untouched."""
    config = _MockConfig(MaskingStrategy(num_mutations=2))
    inputs = MaskedModelInput(sequences=["MKTLLIFLA"])
    result = apply_masking_strategy(config, inputs)
    assert inputs.sequences == ["MKTLLIFLA"]
    assert result is not inputs


# ── Method field and validation ──────────────────────────────────────────────


def test_method_field_default():
    """Default method is 'random'."""
    assert MaskingStrategy().method == "random"


def test_method_field_roundtrip():
    """Method field survives serialization."""
    s = MaskingStrategy(method="entropy", model_name="esm2")
    data = s.model_dump()
    assert data["method"] == "entropy"
    assert data["model_name"] == "esm2"
    restored = MaskingStrategy(**data)
    assert restored.method == "entropy"
    assert restored.model_name == "esm2"


def test_invalid_method_raises():
    """Unknown method raises a validation error."""
    with pytest.raises(ValidationError, match="Input should be 'random', 'entropy' or 'max-logit'"):
        MaskingStrategy(method="bogus")


def test_invalid_model_for_method():
    """Unsupported model_name for a method raises a validation error."""
    with pytest.raises(ValidationError, match="Input should be 'esm2' or 'esm3'"):
        MaskingStrategy(method="entropy", model_name="bogus_model")


def test_model_fields_ignored_for_random():
    """Setting model_name with method='random' emits a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MaskingStrategy(method="random", model_name="esm2")
    assert len(w) == 1
    assert "ignored" in str(w[0].message)


# ── Temperature ───────────────────────────────────────────────────────────────


def test_temperature_default():
    """Default temperature is 1.0."""
    assert MaskingStrategy().temperature == 1.0


def test_temperature_low_is_greedy():
    """Very low temperature makes selection deterministic (greedy).

    We use a mock masker that gives position 0 a much higher score.
    With low temperature, position 0 should always be selected.
    """
    from unittest.mock import patch

    from proto_tools.tools.masked_models.masking.maskers import Masker

    class GreedyTestMasker(Masker):
        supported_models = None

        def score(self, sequences, position_score_fn=None):
            # Position 0 gets score 2, rest get score 1
            return [[2.0] + [1.0] * (len(seq) - 1) for seq in sequences]

    strategy = MaskingStrategy(method="random", num_mutations=1, temperature=0.01)
    with patch.dict("proto_tools.tools.masked_models.masking.maskers.MASKERS", {"random": GreedyTestMasker}):
        results = [strategy.mask(["ABCDEFGHIJ"])[0] for _ in range(20)]
    # Position 0 (A) should be masked in all runs with very low temperature
    assert all(r[0] == "_" for r in results)


def test_temperature_high_is_uniform():
    """Very high temperature makes score differences negligible.

    With extreme temperature, even heavily biased scores produce
    near-uniform sampling.
    """
    from unittest.mock import patch

    from proto_tools.tools.masked_models.masking.maskers import Masker

    class BiasedTestMasker(Masker):
        supported_models = None

        def score(self, sequences, position_score_fn=None):
            # Position 0 gets a much higher score
            return [[100.0] + [1.0] * (len(seq) - 1) for seq in sequences]

    strategy = MaskingStrategy(method="random", num_mutations=1, temperature=1000.0)
    with patch.dict("proto_tools.tools.masked_models.masking.maskers.MASKERS", {"random": BiasedTestMasker}):
        masked_positions = []
        for _ in range(100):
            result = strategy.mask(["ABCDEFGHIJ"])[0]
            pos = result.index("_")
            masked_positions.append(pos)
    # With very high temperature, we should see variety (not always position 0)
    assert len(set(masked_positions)) > 1


def test_temperature_serialization_roundtrip():
    """Temperature survives serialization."""
    s = MaskingStrategy(temperature=0.5)
    data = s.model_dump()
    assert data["temperature"] == 0.5
    restored = MaskingStrategy(**data)
    assert restored.temperature == 0.5


# ── Masker persistence ────────────────────────────────────────────────────────


def test_masker_is_reused_across_calls():
    """The masker instance persists across multiple .mask() calls."""
    strategy = MaskingStrategy(num_mutations=2)
    strategy.mask(["ABCDEFGHIJ"])
    masker_first = strategy._masker
    strategy.mask(["KLMNOPQRST"])
    masker_second = strategy._masker
    assert masker_first is masker_second


def test_masker_not_set_before_first_call():
    """_masker is None until .mask() is called."""
    strategy = MaskingStrategy()
    assert strategy._masker is None


# ── E2E tests (parameterized over all methods) ──────────────────────────────

_E2E_NUM_MUTATIONS = 3
_E2E_FIXED_POSITIONS = [1, 2, 3]  # 1-indexed: M, K, T are protected

_ALL_METHODS = [
    pytest.param(key, marks=pytest.mark.uses_gpu, id=key)
    if MASKERS[key].supported_models is not None
    else pytest.param(key, id=key)
    for key in MASKERS
]


def _create_strategy(method, num_mutations=None, mask_fraction=None, fixed_positions=None):
    """Create a MaskingStrategy, adding model_name if the method needs one."""
    kwargs = {"method": method, "fixed_positions": fixed_positions}
    if num_mutations is not None:
        kwargs["num_mutations"] = num_mutations
    if mask_fraction is not None:
        kwargs["mask_fraction"] = mask_fraction
    if MASKERS[method].supported_models is not None:
        kwargs["model_name"] = "esm2"
    return MaskingStrategy(**kwargs)


def _validate_masked_output(original, masked, num_mutations, fixed_positions=None):
    """Assert correct mask count, preserved context, and fixed positions."""
    assert len(masked) == len(original)
    assert masked.count(MASK_TOKEN) == num_mutations
    for i, (o, m) in enumerate(zip(original, masked, strict=False)):
        if m != MASK_TOKEN:
            assert m == o, f"Position {i}: expected '{o}', got '{m}'"
    # Fixed positions (1-indexed) must never be masked
    if fixed_positions:
        for pos in fixed_positions:
            assert masked[pos - 1] == original[pos - 1], f"Fixed position {pos} was masked: '{masked[pos - 1]}'"


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_method_num_mutations(method):
    """Each method masks exactly num_mutations positions, respects.

    fixed_positions, and handles a batch of different-length sequences.
    """
    sequences = ["MKTAYIAK", "MKTAYIAKQR", "MKTAYIAKQRQISFVK"]
    strategy = _create_strategy(
        method,
        num_mutations=_E2E_NUM_MUTATIONS,
        fixed_positions=_E2E_FIXED_POSITIONS,
    )
    result = strategy.mask(sequences)
    assert len(result) == len(sequences)
    for orig, masked in zip(sequences, result, strict=False):
        _validate_masked_output(
            orig,
            masked,
            _E2E_NUM_MUTATIONS,
            fixed_positions=_E2E_FIXED_POSITIONS,
        )


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_method_mask_fraction(method):
    """Each method masks ~50% of designable positions, respects.

    fixed_positions, and handles a batch of different-length sequences.
    """
    sequences = ["MKTAYIAK", "MKTAYIAKQR", "MKTAYIAKQRQISFVK"]
    fraction = 0.5
    strategy = _create_strategy(
        method,
        mask_fraction=fraction,
        fixed_positions=_E2E_FIXED_POSITIONS,
    )
    result = strategy.mask(sequences)
    assert len(result) == len(sequences)
    for orig, masked in zip(sequences, result, strict=False):
        # Designable = total - fixed positions present in this sequence
        n_fixed = sum(1 for p in _E2E_FIXED_POSITIONS if p <= len(orig))
        n_designable = len(orig) - n_fixed
        expected_masks = max(1, round(n_designable * fraction))
        _validate_masked_output(
            orig,
            masked,
            expected_masks,
            fixed_positions=_E2E_FIXED_POSITIONS,
        )
