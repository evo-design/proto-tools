"""Tests for BaseConfig seed field and reproducibility guarantees."""

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils.base_config import BaseConfig
from tests.tool_infra_tests.pytest_helpers import (
    CHIMERA_ONLY_KEYS,
    EXCLUDED_CATEGORIES,
    build_inputs_and_config,
    parse_min_gpu_count,
)

# ============================================================================
# BaseConfig seed mechanics
# ============================================================================


def test_resolved_seed_auto_assigns():
    """seed=None auto-assigns distinct resolved_seed values."""
    c1, c2 = BaseConfig(), BaseConfig()
    assert c1.seed is None and c2.seed is None
    assert isinstance(c1.resolved_seed, int)
    assert c1.resolved_seed != c2.resolved_seed


@pytest.mark.parametrize("seed", [0, 42])
def test_resolved_seed_uses_explicit(seed):
    """Explicit seed flows through to resolved_seed."""
    assert BaseConfig(seed=seed).resolved_seed == seed


def test_cache_key_with_seed():
    """Seed appears in cache key when set, absent when None."""
    assert '"seed": 42' in BaseConfig(seed=42).cache_key()
    assert '"seed": 43' in BaseConfig(seed=43).cache_key()
    assert "seed" not in BaseConfig().cache_key()
    # Two None-seed configs produce the same cache key
    assert BaseConfig().cache_key() == BaseConfig().cache_key()


# ============================================================================
# Masking strategy reproducibility
# ============================================================================


def test_masking_strategy_seeded():
    """Same seed → same mask positions; different seeds → different positions."""
    from proto_tools.tools.masked_models.masking import MaskingStrategy

    strategy = MaskingStrategy(method="random", mask_fraction=0.5)
    seqs = ["ACGTACGTACGTACGT"]
    assert strategy.mask(seqs, seed=42) == strategy.mask(seqs, seed=42)
    assert strategy.mask(seqs, seed=42) != strategy.mask(seqs, seed=99)


# ============================================================================
# Exhaustive seed reproducibility (all registered tools)
# ============================================================================

# Tools excluded from seed reproducibility tests due to upstream non-determinism
# that cannot be fixed on our end (external subprocess without deterministic mode).
# See: https://github.com/RosettaCommons/foundry/issues/170
#
# - rfdiffusion3-design: external subprocess without a deterministic mode.
# - protenix-prediction: protenix's CLI ``--seed`` is honoured, but the
#   cuequivariance triangle multiplication / attention kernels accumulate
#   floating-point ops non-deterministically across runs. Coordinates drift by
#   ~1-2 mÅ even with the same seed. Forcing the torch fallback kernel would
#   restore determinism at a significant speed cost, so we exclude instead.
# - chai1-prediction: deterministic across fresh subprocesses, but consecutive
#   dispatches inside the same persistent worker drift by ~3 mÅ with a fully
#   consistent delta. Root cause appears to be hidden CUDA/JIT state inside
#   chai_lab that torch does not expose a reset API for — neither pre-loading
#   ESM, early CUBLAS_WORKSPACE_CONFIG, empty_cache+synchronize, nor the
#   existing set_torch_seed+use_deterministic_algorithms path eliminates the
#   drift. User impact is negligible (well below bond-length noise) and a full
#   warmup at load time would add significant first-call latency.
# - progen3-sample, progen3-score: MoE forward-pass non-determinism in the
#   grouped-gemm CUDA library used by megablocks. A weight-hash diagnostic
#   confirms two fresh subprocesses load bit-identical weights, so the drift
#   is entirely in the MoE forward pass (~1e-3 in log-likelihood, amplified
#   to completely different sequences via autoregressive sampling). The
#   persistent variant passes because the same CUDA context gives the same
#   grouped-gemm trajectory on consecutive calls. Upstream acknowledged:
#   - https://github.com/Profluent-AI/progen3/issues/6
#   - https://github.com/databricks/megablocks/issues/83
_SEED_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "rfdiffusion3-design",
        "protenix-prediction",
        "chai1-prediction",
        "progen3-sample",
        "progen3-score",
    }
)


def _build_seed_test_params() -> list:
    """Build pytest parametrize params for seed reproducibility across all tools.

    Unlike env-report tests, this does NOT deduplicate by directory — seed
    behavior is per-tool, not per-environment.
    """
    params = []

    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        if spec.category in EXCLUDED_CATEGORIES:
            continue
        if spec.key in _SEED_EXCLUDED_KEYS:
            continue
        if spec.example_input is None:
            continue

        marks: list = []

        if spec.uses_gpu:
            gpu_count = parse_min_gpu_count(spec.device_count)
            marks.append(pytest.mark.uses_gpu(gpu_count))

        if spec.key in CHIMERA_ONLY_KEYS:
            marks.append(pytest.mark.only_chimera)

        params.append(pytest.param(spec, id=spec.key, marks=marks))

    return params


@pytest.mark.exhaustive
@pytest.mark.parametrize("spec", _build_seed_test_params())
def test_all_tools_seed_reproducibility(spec: ToolSpec, tmp_path):
    """Same seed + same input produces identical output for every registered tool."""
    inputs, config = build_inputs_and_config(spec, tmp_path, {"seed": 42})

    r1 = spec.function(inputs, config)
    assert r1.success, f"First run of {spec.key} failed: {r1.errors}"

    r2 = spec.function(inputs, config)
    assert r2.success, f"Second run of {spec.key} failed: {r2.errors}"

    r1.approx_equal(r2)


@pytest.mark.exhaustive
@pytest.mark.parametrize("spec", _build_seed_test_params())
def test_all_tools_seed_reproducibility_persistent(spec: ToolSpec, tmp_path):
    """Same seed produces identical output across dispatches to a persistent worker."""
    from proto_tools.utils.tool_instance import ToolInstance

    # Skip pure-Python tools — they run in-process, so ``ToolInstance.persist_tool``
    # has nothing to persist and would raise.
    if not spec.has_standalone_env:
        pytest.skip(f"{spec.key} has no standalone env — persistent worker not applicable")

    inputs, config = build_inputs_and_config(spec, tmp_path, {"seed": 42})
    tool_dir = spec.source_file.parent.name

    with ToolInstance.persist_tool(tool_dir):
        r1 = spec.function(inputs, config)
        assert r1.success, f"First run of {spec.key} failed: {r1.errors}"

        r2 = spec.function(inputs, config)
        assert r2.success, f"Second run of {spec.key} failed: {r2.errors}"

        r1.approx_equal(r2)
