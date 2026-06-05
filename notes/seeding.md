# Seeding

This note covers how seeds, caching, and dedup interact in proto-tools across deterministic and stochastic tools.

## The two categories

A `@tool()` registration either declares `stochastic=True` or it doesn't. The flag drives three framework behaviors:

1. Cache-skip when `seed is None`.
2. Skip dedup for iterable inputs.
3. Use whole-call cache instead of per-item cache.

`seed` lives on `BaseConfig`, so every config accepts a `seed` kwarg. Tools that don't actually sample (BLAST, DSSP, etc.) inherit the field but ignore it in their inference, so passing `seed=42` to them is a no-op.

### Deterministic tools (no flag)

Tools whose inference doesn't sample. Some may still call `set_torch_seed(config.seed)` defensively for persistent-worker consistency, but their output is mathematically determined by input + config (modulo GPU float noise). Cache and dedup behave normally, so they are always cacheable and identical iterable items deduplicate.

**Examples (by category):**

- **Sequence search & alignment**: BLAST, MMseqs2, ColabFold-search, Foldseek, Foldmason, mafft
- **HMM tools**: hmmsearch, hmmscan, jackhmmer, phmmer, nhmmer
- **Structure analysis & scoring**: DSSP, ipsae, pdockq2, structure-metrics, pyrosetta-energy, pyrosetta-sasa, pyrosetta-sap
- **Single-pass structure prediction**: ESMFold-prediction
- **Forward-pass scoring**: Evo1/2-score, ProGen2/3-score, ESM2/3-score, ProteinMPNN-score, LigandMPNN-score, ESM-IF1-score, FaMPNN-score, FaMPNN-score-all-mutations
- **Predictive models**: Borzoi-prediction, Borzoi-ensemble, Enformer-prediction
- **Annotation & lookup**: InterProScan, Ensembl, AlphaFold-DB lookup, CCD lookup, MinCED, AlphaMissense DB, AlphaGenome
- **Embeddings**: ESMC

### Stochastic tools (`stochastic=True`)

Tools whose algorithm explicitly samples (`torch.multinomial`, `torch.randn`, JAX `random.PRNGKey`, `pyrosetta.rg().set_seed`, etc.). Different seeds produce different outputs.

**Examples (by category):**

- **Language model samplers**: Evo1-sample, Evo2-sample, ProGen2-sample, ProGen3-sample, ESM2-sample, ESM3-sample, AbLang-sample
- **Gradient / interpretability tools**: ESM2-gradient, AbLang-gradient, ESMFold-gradient, ProteinMPNN-gradient
- **Inverse folding samplers**: ProteinMPNN-sample, LigandMPNN-sample, ESM-IF1-sample, FaMPNN-sample, FaMPNN-pack
- **Iterative / diffusion structure predictors**: AlphaFold2-binder, AlphaFold3-prediction, Boltz2-prediction, Chai1-prediction, Protenix-prediction
- **Deterministic predictors** (no `stochastic` flag): AlphaFold2-prediction. The seed only varies output when the supplied MSA exceeds `num_msa=512` rows (via cluster subsampling), since single-sequence and shallow-MSA inputs are a pure function of sequence + recycle count.
- **Design tools**: Bindcraft-design, Germinal-design, RFDiffusion3-design
- **Structure dynamics**: BioEmu-sample
- **Monte Carlo scoring**: PyRosetta-relax, PyRosetta-interface-analyzer
- **Random generators**: random-nucleotide-sample, random-protein-sample

## RNG vs hardware non-determinism

Two unrelated sources of variation can make a tool appear "non-deterministic." Only the first is controllable by a seed, and only the first warrants the `stochastic=True` flag.

1. **Algorithmic RNG.** The tool's inference calls `torch.multinomial`, `torch.randn`, JAX `random.PRNGKey`, or any other explicit RNG that meaningfully affects the output. A seed controls this. Tools in this category get `stochastic=True`.
2. **Hardware / numerical noise.** Bit-level variation in floating-point arithmetic from fused GPU kernels (FlashAttention, cuDNN autotune, JAX bfloat16, reduction-order in matmul). The algorithm is mathematically deterministic, but the chip doesn't compute it bit-exact between runs. **No seed controls this.** The only fix is `torch.use_deterministic_algorithms(True)`, which carries significant performance cost and is intentionally not enabled. `BaseToolOutput.approx_equal` is the contract for this, treating outputs as "equal" up to small float noise.

Reproducibility and diversification testing (`test_seed_reproducibility.py` and `test_seed_diversification.py`) runs only against `stochastic=True` tools. Unflagged tools are assumed reproducible by construction, and testing them would amount to testing GPU determinism, which is out of scope.

## Behavior for deterministic tools (no flag)

Seed is accepted but ignored (no-op), and cache and dedup are always on, so behavior is identical regardless of whether `seed` is passed.

### Non-iterable input

```
Input(itemA) + Config() ==> Output(resultA)
```

On a repeat call, the result is the same and the cache hits.

### Iterable input with unique items

```
Input([i1, i2, i3]) + Config() ==> Output([r1, r2, r3])
```

On a repeat call, the result is the same for each input, with per-item cache hits.

### Iterable input with duplicate items

```
Input([i1, i1, i1]) + Config()
  =(Cache dedup: input run only once)=>
  Output([r1, r1, r1])
```

On a repeat call, the result is the same for each input.

### Iterable input with the tool's internal batching

For tools that batch internally (e.g., score tools running model forward passes on chunks of inputs):

```
Input([s1, s1, s2, s3, s4]) + Config()
  =(Cache dedup: only one s1 passed forward)=>
  Tool internal batches: [s1, s2], [s3, s4]
  Tool internal results: [r1, r2], [r3, r4]
  =(Stitch + expand dedup)=>
  Output([r1, r1, r2, r3, r4])
```

There are two distinct layers, because the framework's dedup runs before the tool sees the input, while the tool's internal batching (`batch_size` config) runs over the post-dedup list.

## Behavior for stochastic tools (`stochastic=True`) with `seed=None`

When `seed` is `None`, the cache and dedup are both skipped. Each call draws a fresh random base.

### Non-iterable input

```
Input(itemA) + Config(seed=None) ==> Output(resultX)
```

A repeat call produces a different random result.

### Iterable input with unique items

```
Input([i1, i2, i3]) + Config(seed=None) ==> Output([x1, x2, x3])
```

A repeat call produces different results.

### Iterable input with duplicate items

`Input([i1, i1, i1]) + Config(seed=None)`

↓ *NO CACHE USE for stochastic configs: each position treated as distinct*

Tool sees `[i1, i1, i1]`

↓

`Output([X1, Y1, Z1])`

Different results for each position. Repeat calls produce yet another fresh set (no reproducibility, which is the whole point of `seed=None`).

**Caveat**: this diverse-output behavior depends on the tool's internal RNG advancement (see [Per-item diversification is tool-specific](#per-item-diversification-is-tool-specific) below). For tools whose upstream sampler does not advance RNG per item, identical inputs produce identical outputs even with `seed=None`, and the framework cannot change that.

## Behavior for stochastic tools (`stochastic=True`) with `seed=2`

When `seed` is set, output is reproducible across calls. Cache + dedup are enabled.

### Non-iterable input

```
Input(itemA) + Config(seed=2) ==> Output(resultA)
```

On a repeat call, the result is the same and the cache hits.

### Iterable input with unique items

```
Input([i1, i2, i3]) + Config(seed=2) ==> Output([r1, r2, r3])
```

On a repeat call, the result is the same for each input, with a whole-call cache hit.

### Iterable input with duplicate items

`Input([i1, i1, i1]) + Config(seed=2)`

↓ *NO DEDUP for stochastic iterables*

Tool sees `[i1, i1, i1]`

↓ Tool sets `torch.manual_seed(2)` once

↓ Tool `forward(batch)` → logits

↓ `multinomial(probs[0])` → `r1` *(consumes RNG, advances state)*

↓ `multinomial(probs[1])` → `x1` *(different, advanced state)*

↓ `multinomial(probs[2])` → `y1` *(different again)*

`Output([r1, x1, y1])`

Diverse outputs (per-item RNG advancement) AND reproducible across calls (same seed → same multinomial draw sequence).

The mechanism works as follows. One seed enters the tool, and per-item sampling primitives (`torch.multinomial` or equivalent) consume and advance the global RNG state between batch elements, so identical inputs in the same batch diverge. Same seed → same sequence of multinomial draws → reproducible across calls.

### Iterable input with the tool's internal batching

`Input([s1, s1, s2, s3, s4]) + Config(seed=2)`

↓ *NO DEDUP for stochastic iterables: s1 duplicates passed forward*

Tool internal batches (batch_size=2): `[s1, s1]`, `[s2, s3]`, `[s4]`

↓

Tool internal results: `[r1, x1]`, `[r2, r3]`, `[r4]`

↓

`Output([r1, x1, r2, r3, r4])`

The duplicates in the same internal batch (`[s1, s1]`) diverge because the tool's per-item multinomial advances RNG between them.

## Architectural invariants

Three invariants underpin the design.

### 1. The seed is set once per dispatch, not per element

The framework MUST NOT call `set_torch_seed`, derive per-item seeds, or otherwise inject seed manipulation between batch elements. The seed that enters a `stochastic=True` tool's inference is the user-supplied `config.seed` (or a fresh random base when `seed is None`), set **once** at the top of the tool's inference.

**Enforcement:** architectural. `tool_registry.py` does not contain unroll/re-seed logic. Re-introducing it should be visible at code-review time.

### 2. Per-item diversification is the tool's responsibility

Once one seed enters the tool, the tool's existing per-item sampling primitives (`torch.multinomial`, JAX `random.split`, autoregressive decode loops, etc.) consume and advance RNG between items, which is what makes identical inputs in one batched call produce diverse outputs. Many wrapped samplers do not advance RNG per item, and for those the framework cannot make identical inputs diverge (see [Per-item diversification is tool-specific](#per-item-diversification-is-tool-specific)).

**Enforcement:** functional. `test_seed_diversification.py` parameterizes across all `stochastic=True` tools and asserts three pairwise-distinct outputs from three identical inputs.

### 3. GPU batching is preserved

A tool receiving an N-item iterable produces N outputs via the same forward-pass batching it would use for distinct items. The framework does not split an N-item dispatch into N single-item calls, because that would erase the throughput advantage of batched generation and the amortization of fixed per-call overhead (model warmup, KV-cache setup).

**Enforcement:** architectural, same code-path as invariant 1. The diversification test does **not** directly verify that batching is preserved, because a framework that serialized batches would still pass the diversification check and would simply run slower. Code review on changes to `tool_registry.py` and any new dispatch helpers is the catching mechanism.

## Per-item diversification is tool-specific

The "skip dedup, the tool diversifies naturally" behavior above relies on each tool's own sampling code advancing its RNG between items within a batched call. That is a property of the wrapped upstream library, not something proto-tools controls. The framework sets one seed at the top of the tool's inference, and everything after that is internal to the tool, so when an upstream sampler does not advance its RNG per item, identical inputs in one call return identical outputs even with `seed=None`.

This is an inherent consequence of wrapping third-party samplers, not a gap we expect to close. A good number of stochastic tools do not diversify per item, and there is no general framework-level fix, because the framework cannot reach into an upstream library's sampling loop. Treat per-item diversification and exact reproducibility as best-effort and tool-specific rather than guarantees the framework makes.

### ESMFold

ESMFold has two registered tools in `esmfold.py`. The prediction tool is a single-pass forward pass and does not carry the flag, while the gradient tool reads `config.seed` and is registered with `stochastic=True`. They share a config-class hierarchy (`ESMFoldGradientConfig` extends `ESMFoldConfig`) and both inherit `seed` from `BaseConfig`, though the prediction tool simply ignores the inherited field.

## Reproducibility caveats

A few subtleties worth knowing:

1. **Ordering matters.** Reproducibility holds for `(inputs in order, config)`. `[i1, i2, i3]` and `[i2, i1, i3]` with the same seed produce different outputs because per-item RNG advancement is sequential.
2. **Bit-exact reproducibility on GPU is not promised.** Float noise from fused kernels, cuDNN autotune, and bfloat16 reductions can cause small variation between runs. `BaseToolOutput.approx_equal` (used by `test_seed_reproducibility.py`) handles this contractually.
3. **Per-item cache is incompatible with skip-dedup for `stochastic=True` tools.** Per-item cache keys are position-independent (`(item, config)`), but for stochastic tools two identical items in one call produce different outputs (per-item RNG advancement), so caching item 0's result as `(i1, seed=2)` and later reusing it at position 1 would be wrong. Stochastic tools therefore use whole-call cache only (key = `(full_input_iterable, config)`), not per-item cache. The realistic call pattern for samplers doesn't exercise partial-batch re-runs, so the loss is minimal.

## Testing

Reproducibility and diversification tests parameterize across **`stochastic=True` tools only**. Unflagged tools are assumed reproducible by construction, and testing them would amount to testing GPU determinism, which is out of scope.

Cross-tool parametrized tests are marked `@pytest.mark.extensive`. They run opt-in via `pytest --ext` / `--extensive` (not on every PR), and they are not enabled by `--all` or `--slow`. This matches the existing pattern for combinatorial tests that fan out across every registered tool.

`tests/seed_tests/test_seed_diversification.py` parameterizes across all `stochastic=True` tools and asserts:

- `[i1, i1, i1] + seed=2` produces three pairwise-distinct outputs (the diversification claim, meaning one distinct result per input position).
- `[i1, i1, i1] + seed=2` is reproducible across calls (per-batch reproducibility, meaning running the same call twice gives the same triple).
- `seed=None` repeat calls produce different results (cache must skip).

`tests/seed_tests/test_seed_reproducibility.py` filters its parametrization to `stochastic=True` tools. Same-seed-same-output is meaningful only there.

`tests/seed_tests/test_stochastic_iterable_routing.py` exercises the framework's cache / dedup / routing using three pure-CPU mock tools in `proto_tools/tools/testing/`:

- `mock-iterable-stochastic`: internal batched sampling
- `mock-iterable-stochastic-serial`: pure serial loop, no batching
- `mock-iterable-deterministic`: deterministic per-prompt scoring

These mocks expose an `items_processed` field so tests can directly observe whether the framework dedup'd before the tool ran, and they run in under a second total and aren't marked `extensive`.
