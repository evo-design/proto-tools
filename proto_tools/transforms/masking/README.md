# Masking

## Overview

Shared position-selection utility used by masked protein language model tools (ESM2, ESM3) to decide *which* residues to hide before the model predicts replacements. It is not a standalone tool — it provides the `MaskingStrategy` config class and helper functions that the sampler tools consume internally.

## Background

Masked language modelling over proteins works by blanking out a subset of positions in a sequence and asking the model to predict them. The quality of the resulting samples depends heavily on *which* positions get masked:

- Masking random positions is cheap and unbiased, but ignores the model's own uncertainty.
- Masking positions where the model is least certain (highest-entropy) concentrates the budget on residues the model thinks are most flexible — useful for design.
- Masking positions where the model most confidently predicts a *different* amino acid (low max-logit of the wild-type residue) concentrates on likely mutations — useful for directed editing.

This utility centralises that logic in one place so ESM2 and ESM3 share the same strategy surface, the same defaults, and the same reproducibility guarantees.

## How It Works

`MaskingStrategy` is a Pydantic config object with three fields that matter at selection time: `method`, `mask_fraction` (or `num_mutations`), and `fixed_positions`. Tools that use it pass the object into `apply_masking_strategy(config, inputs, position_score_fn)`, which:

1. Computes the set of *designable* positions (the full sequence minus anything in `fixed_positions`).
2. Resolves how many positions to mask. Priority: `num_mutations` (exact count) > `mask_fraction` (proportion of designable positions) > a 30% default.
3. Scores each designable position according to `method`:
   - `"random"` — uniform scores → uniform random sampling
   - `"entropy"` — Shannon entropy of the model's per-position logit distribution (higher = more uncertain)
   - `"max-logit"` — negated max logit of the wild-type amino acid (higher = model confidently prefers *something else*)
4. Draws positions via weighted sampling (`weighted_sample`). A score temperature (`score_temperature`) controls how sharply the sampler concentrates on the top-scoring positions when `method` is `"entropy"` or `"max-logit"`.
5. Applies the mask to the sequence with `apply_mask`, returning the masked string ready for the downstream model to fill in.

All randomness flows through an explicit `np.random.RandomState`, so a given `(method, mask_fraction, fixed_positions, seed)` tuple is fully reproducible.

## Quick Start Examples

```python
from proto_tools.transforms.masking import MaskingStrategy

# Default: uniform random, 30% of designable positions.
MaskingStrategy().mask(["MKTLLIFLA"])
# → ["M_TLLIFLA"]

# Entropy-guided, exact count, model-scored.
MaskingStrategy(
    method="entropy",
    model_name="esm2",
    num_mutations=3,
).mask(["MKTLLIFLA"])
# → ["MK_LL_F_A"]  (three highest-entropy positions)

# Mask a larger editable subset while preserving fixed positions.
MaskingStrategy(mask_fraction=0.5, fixed_positions=[1]).mask(["MKTLLIFLA"])
# → ["MK_L__F_A"]
```

The masked strings are handed directly to ESM2 / ESM3 sampling tools; see the [ESM2](../../tools/masked_models/esm2/README.md) and [ESM3](../../tools/masked_models/esm3/README.md) example notebooks for end-to-end usage.

## Best Practices & Gotchas

- **Don't set both `num_mutations` and `mask_fraction`.** The config validates this at construction time and raises.
- **`mask_fraction` is applied to the *designable* count**, not the full sequence length. With `fixed_positions=[1, 2, 3]` on a 100-residue sequence, `mask_fraction=0.3` masks ~29 positions, not 30.
- **`entropy` and `max-logit` require a model to score with.** Set `model_name="esm2"` (or `"esm3"`) on the strategy. Without a score function, the two methods silently degrade to uniform random.
- **Use explicit seeds for reproducibility.** The samplers use a `np.random.RandomState`; pass a seed through the tool's config to get byte-identical masked outputs across runs.
- **Score temperature only affects the scored methods.** Adjusting `score_temperature` when `method="random"` has no effect.

## References

- [ESM (Evolutionary Scale Modeling) research](https://www.evolutionaryscale.ai/)
- [Masked language modelling — original BERT paper](https://arxiv.org/abs/1810.04805) (the technique that pLMs inherit)
- Shannon entropy as an uncertainty measure: [Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory))

## Related Tools

- [ESM2](../../tools/masked_models/esm2/README.md) — Meta AI's protein language model; uses this masking utility for sampling.
- [ESM3](../../tools/masked_models/esm3/README.md) — ESM3 (multimodal protein language model); also uses this masking utility.
