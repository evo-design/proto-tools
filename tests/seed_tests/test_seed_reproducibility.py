"""Tests for BaseConfig seed field and reproducibility guarantees."""

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils.base_config import BaseConfig
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.pytest_helpers import (
    EXCLUDED_CATEGORIES,
    SKIP_CI_TOOLKITS,
    build_inputs_and_config,
    parse_min_gpu_count,
)

# ============================================================================
# BaseConfig seed mechanics
# ============================================================================


def test_cache_key_with_seed():
    """Seed appears in cache key when set, absent when None."""
    assert '"seed": 42' in BaseConfig(seed=42).cache_key()
    assert '"seed": 43' in BaseConfig(seed=43).cache_key()
    assert '"seed":' not in BaseConfig().cache_key()
    # Two None-seed configs produce the same cache key
    assert BaseConfig().cache_key() == BaseConfig().cache_key()


def test_derive_per_item_seeds():
    """Distinct + reproducible per base seed; unseeded falls back to a fresh random base."""
    a = BaseConfig(seed=42).derive_per_item_seeds(5)
    assert a == BaseConfig(seed=42).derive_per_item_seeds(5)  # reproducible
    assert len(set(a)) == 5  # pairwise distinct
    assert a != BaseConfig(seed=99).derive_per_item_seeds(5)  # base-seed sensitive
    assert len(set(BaseConfig().derive_per_item_seeds(5))) == 5  # unseeded fallback still distinct


# ============================================================================
# Masking strategy reproducibility
# ============================================================================


def test_masking_strategy_seeded():
    """Same seed → same mask positions; different seeds → different positions."""
    from proto_tools.transforms.masking import MaskingStrategy

    strategy = MaskingStrategy(method="random", mask_fraction=0.5)
    seqs = ["ACGTACGTACGTACGT"]
    assert strategy.mask(seqs, seed=42) == strategy.mask(seqs, seed=42)
    assert strategy.mask(seqs, seed=42) != strategy.mask(seqs, seed=99)


# ============================================================================
# Extensive seed reproducibility (all registered tools)
# ============================================================================

# Three exclusion sets, depending on which variants of the seed reproducibility
# test a tool fails. The goal is to keep test coverage for the variant that
# *does* pass when the failure is variant-specific, instead of dropping a tool
# from both tests.
#
# All exclusions are due to upstream non-determinism we cannot fix on our end
# without unacceptable speed costs (e.g. forcing slower deterministic CUDA
# kernels or float32 outputs).

# Tools where BOTH variants fail. Excluded from collection entirely.
#
# - rfdiffusion3-design: external subprocess without a deterministic mode.
#   Upstream: https://github.com/RosettaCommons/foundry/issues/170.
# - rf3-prediction: foundry CLI tool like rfdiffusion3-design, no deterministic
#   mode; a fresh ``rf3 fold`` subprocess per fold makes cross-call comparisons
#   cross-process, hitting intermittent bimodal CUDA kernel drift.
# - protenix-prediction: protenix's CLI ``--seed`` is honoured, but the
#   cuequivariance triangle multiplication / attention kernels accumulate
#   floating-point ops non-deterministically across runs. Coordinates drift by
#   ~1-2 mÅ even with the same seed. Upstream:
#   - https://github.com/bytedance/Protenix/issues/116
#   - https://github.com/bytedance/Protenix/issues/119
# - alphafold2-gradient: JAX bfloat16 / CUDA autotune drift (ColabDesign's
#   default ``global_config.bfloat16=True``) amplified by gradient backprop
#   beyond tolerances even within a single persistent worker (~12% relative
#   error on gradient values).
# - alphafold3-prediction: AF3 honours the seed via ``modelSeeds`` →
#   ``jax.random.PRNGKey(seed)`` in run_alphafold.py, but the diffusion +
#   pairformer stack hits JAX/CUDA autotune + bfloat16 reduction-order
#   non-determinism. ~7 mÅ atomic-coordinate drift between runs (same
#   seed) — across both persistent and non-persistent variants. Mitigations
#   like ``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` would slow inference
#   2-5x without a guarantee of bit-exact reproducibility.
# - germinal-design: binder-design pipeline that runs AF2 hallucination +
#   re-prediction internally, inheriting the same JAX/CUDA non-determinism as
#   alphafold2/3 above. Cross-run drift changes which trajectory filters fire,
#   so even the ``pipeline_stats`` keys differ between runs; both variants fail.
# - bindcraft-design, freebindcraft-design: AF2 hallucination + MPNN pipeline
#   (same ColabDesign/JAX non-determinism as germinal-design); BindCraft also
#   samples binder length + trajectory seed from an unseeded numpy RNG, so
#   cross-call runs diverge in both trajectory choice and filter outcomes.
_SEED_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "rfdiffusion3-design",
        "rf3-prediction",
        "protenix-prediction",
        "alphafold2-gradient",
        "alphafold3-prediction",
        "germinal-design",
        "bindcraft-design",
        "freebindcraft-design",
    }
)

# Tools where ONLY the persistent variant fails. Fresh subprocesses with the
# same seed are reproducible, but consecutive dispatches inside the same
# persistent worker drift due to hidden CUDA/JIT state.
#
# - chai1-prediction: ~3 mÅ drift with a fully consistent delta. Hidden
#   CUDA/JIT state inside ``chai_lab`` that torch does not expose a reset API
#   for — neither pre-loading ESM, early ``CUBLAS_WORKSPACE_CONFIG``,
#   ``empty_cache + synchronize``, nor the existing ``set_torch_seed +
#   use_deterministic_algorithms`` path eliminates the drift. Upstream:
#   - https://github.com/chaidiscovery/chai-lab/issues/228
#   - https://github.com/chaidiscovery/chai-lab/issues/246
# - alphagenome-predict-variants: second dispatch crashes the persistent
#   worker. The tool is marked ``gpu_only=True`` so the eviction round-trip
#   test handles it via worker restart, but the persistent seed test has no
#   eviction to trigger that path.
# - esmfold2-prediction: ~2 mÅ atom drift between consecutive in-worker dispatches; root cause not isolated.
# - boltz2-prediction, boltz2-affinity: resident model; ~0.1 Å cross-dispatch drift from hidden CUDA/JIT state (same class as chai1-prediction).
_SEED_PERSISTENT_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "chai1-prediction",
        "alphagenome-predict-variants",
        "esmfold2-prediction",
        "boltz2-prediction",
        "boltz2-affinity",
    }
)

# Tools where ONLY the non-persistent variant fails. The same loaded model
# inside one worker process produces deterministic output, but two fresh
# subprocesses (with the same seed) end up with slightly different CUDA kernel
# autotune choices and drift at the kernel-numerics level.
#
# - progen3-sample, progen3-score: MoE forward-pass non-determinism in the
#   ``grouped-gemm`` CUDA library used by megablocks. A weight-hash diagnostic
#   confirmed two fresh subprocesses load bit-identical weights, so the drift
#   is entirely in the MoE forward pass (~1e-3 in log-likelihood, amplified to
#   completely different sequences via autoregressive sampling). Upstream:
#   - https://github.com/Profluent-AI/progen3/issues/6
#   - https://github.com/databricks/megablocks/issues/83
# - alphagenome-*: JAX bfloat16 forward pass with ``compute=bfloat16,
#   output=bfloat16``. Inter-process CUDA kernel variance produces underlying
#   float32 drift large enough to cross bfloat16 ULP boundaries (~2 ULPs in the
#   7-bit mantissa, 0.4-1.4% relative). The forward pass passes ``PRNGKey=None``
#   so it's mathematically deterministic, but JAX/CUDA autotune doesn't
#   guarantee bit-exact behaviour across processes.
# - esmfold2-prediction: ~7 mÅ atom drift across fresh subprocesses despite
#   matching seeds and standard determinism flags. Root cause not isolated.
# - proteinmpnn-sample: autoregressive JAX sampling — per-item seeds are
#   deterministic, but cross-process kernel numerics flip a few residues at
#   near-ties. Bounded + bimodal (two fixed sequences ~4 residues apart, same
#   class as the other JAX/autoregressive tools here). The persistent variant
#   is reproducible.
# - fampnn-pack: per-item seeds are deterministic, but a packed side-chain
#   coordinate drifts ~1 mÅ across fresh subprocesses (cross-process kernel
#   numerics). The persistent variant is reproducible.
_SEED_NON_PERSISTENT_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "progen3-sample",
        "progen3-score",
        "alphagenome-predict-intervals",
        "alphagenome-predict-sequences",
        "alphagenome-predict-variants",
        "alphagenome-score-intervals",
        "alphagenome-score-ism-variants-batch",
        "alphagenome-score-variants",
        "esmfold2-prediction",
        "proteinmpnn-sample",
        "fampnn-pack",
    }
)


def _build_seed_test_params() -> list:
    """Build pytest parametrize params for seed reproducibility across stochastic tools.

    Filtered to ``stochastic=True`` tools — same-seed-same-output is only
    meaningful where the tool's algorithm actually consumes the seed.
    Unlike env-report tests, this does NOT deduplicate by directory — seed
    behavior is per-tool, not per-environment.
    """
    params = []

    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        if not spec.stochastic:
            continue
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

        if spec.source_file.parent.name in SKIP_CI_TOOLKITS:
            marks.append(pytest.mark.skip_ci)

        params.append(pytest.param(spec, id=spec.key, marks=marks))

    return params


@pytest.mark.extensive
@pytest.mark.parametrize("spec", _build_seed_test_params())
def test_all_tools_seed_reproducibility(spec: ToolSpec, tmp_path):
    """Same seed + same input produces identical output for every registered tool."""
    if spec.key in _SEED_NON_PERSISTENT_EXCLUDED_KEYS:
        pytest.skip(
            f"{spec.key} is excluded from the non-persistent variant: drift across "
            f"fresh subprocesses (see _SEED_NON_PERSISTENT_EXCLUDED_KEYS comment)."
        )

    inputs, config = build_inputs_and_config(spec, tmp_path, {"seed": 42})

    r1 = spec.function(inputs, config)
    assert r1.success, f"First run of {spec.key} failed: {r1.errors}"
    assert_metrics_in_spec(r1)

    r2 = spec.function(inputs, config)
    assert r2.success, f"Second run of {spec.key} failed: {r2.errors}"
    assert_metrics_in_spec(r2)

    r1.approx_equal(r2)


@pytest.mark.extensive
@pytest.mark.parametrize("spec", _build_seed_test_params())
def test_all_tools_seed_reproducibility_persistent(spec: ToolSpec, tmp_path):
    """Same seed produces identical output across dispatches to a persistent worker."""
    from proto_tools.utils.tool_instance import ToolInstance

    # Skip pure-Python tools — they run in-process, so ``ToolInstance.persist_tool``
    # has nothing to persist and would raise.
    if not spec.has_standalone_env:
        pytest.skip(f"{spec.key} has no standalone env — persistent worker not applicable")

    if spec.key in _SEED_PERSISTENT_EXCLUDED_KEYS:
        pytest.skip(
            f"{spec.key} is excluded from the persistent variant: drift or "
            f"crash across consecutive worker calls (see "
            f"_SEED_PERSISTENT_EXCLUDED_KEYS comment)."
        )

    inputs, config = build_inputs_and_config(spec, tmp_path, {"seed": 42})
    tool_dir = spec.source_file.parent.name

    with ToolInstance.persist_tool(tool_dir):
        r1 = spec.function(inputs, config)
        assert r1.success, f"First run of {spec.key} failed: {r1.errors}"
        assert_metrics_in_spec(r1)

        r2 = spec.function(inputs, config)
        assert r2.success, f"Second run of {spec.key} failed: {r2.errors}"
        assert_metrics_in_spec(r2)

        r1.approx_equal(r2)
