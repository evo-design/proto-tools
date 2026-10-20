# Gradient relaxation: the soft/hard contract

Every gradient tool turns a continuous `(L, V)` sequence state into the representation its model
actually consumes. This page is the reference for that step: one formula, one meaning for `soft`
and `hard`, and the same defaults everywhere.

For the surrounding contract — `temperature`, probability mode, the `SequenceLogits` hierarchy,
vocabulary order, outputs — see `notes/gradients.md`.

## The formula

With `X` the supplied matrix, `T` the temperature, and `preprocess` whatever a particular model
applies to logits before its softmax (the identity for all but AlphaFold2):

```
P     = softmax(preprocess(X) / T)       # or X itself when temperature is None
D     = one_hot(argmax(P))
D_ste = stop_gradient(D - P) + P
U     = (1 - soft) * X + soft * P
M     = (1 - hard) * U + hard * D_ste
```

`M` is what reaches the model. The defaults are `temperature=1.0`, `soft=1.0`, `hard=0.0`, which
reduce `M` to a plain `softmax(X / T)`.

- **`soft`** interpolates from the raw input matrix (0) to the softmax probabilities (1). The
  `(1 - soft)` branch is the caller's own matrix, not an argmax one-hot, so it keeps a
  derivative: the gradient at `soft=0` is the upstream vector itself.
- **`hard`** mixes in a straight-through estimator: the forward value moves toward a one-hot
  while the gradient still flows through `P`. At `hard=1` both the forward value and the
  surrogate gradient are independent of `soft`, so `hard` supersedes it.

Differentiating `M` gives an identity route scaled by `(1 - hard) * (1 - soft)` plus a softmax
route scaled by `(1 - hard) * soft + hard`. No documented setting leaves a zero gradient, and a
check pins that: a relaxation that forwarded a constant would let an optimizer run without ever
moving, and nothing would raise.

`preprocess` touches only the softmax branch. AlphaFold2's `alpha` scaling and bias therefore
change `P` and `D_ste` but not the raw-input branch, which is what the formula above specifies.

## What does not move with the blend

- **Masked-model targets.** ESM2, AbLang, CodonFM, Minerva, and GLM2 take each site's target
  from `P`, so the objective does not shift with `soft` or `hard`.
- **ProteinMPNN's negative-log-likelihood labels.** They are the decoded sequence — a hard
  one-hot at every setting. Softmax is monotonic, so they come from `argmax(X)` directly and
  need no branch for probability mode.

## Implementations

| Where | What |
|---|---|
| `standalone_helpers.relaxation.relaxed_sequence` | The torch definition; every torch backend calls it. |
| `standalone_helpers.relaxation.relaxed_sequence_jax` | The JAX twin, for environments with no torch (ProteinMPNN). |
| `standalone_helpers.mixed_mlm` | Re-derives the blend for mixed protein/DNA models, whose `P` is normalized within each position's modality block. Every term is masked, so the raw-input branch cannot weight a forbidden column. |
| ColabDesign's `soft_seq` | AlphaFold2 relaxes inside ColabDesign, on both the upstream and Germinal backends. |

The helpers import their array library lazily and are not re-exported from the package
`__init__`, so an environment that has only one of torch and JAX can still import
`standalone_helpers`.

Four implementations is three more than anyone wants, and each exists for a reason an import
cannot remove: a missing library, a modality-masked softmax, an upstream that owns its own
reparameterization. What keeps them one contract is that each is checked against the same
formula, and the torch and JAX definitions are additionally pinned to one shared table of
forward values and gradients.

## Probability mode

With `temperature=None` the supplied matrix is already `P`, so `U` mixes `X` with itself and the
straight-through term adds `X - stop_gradient(X)`: the forward value is `X` wherever `hard < 1`,
and the gradient is the upstream vector at every setting. Probability-space gradients survive the
controls. AlphaFold2 expresses this mode as `soft=0` inside ColabDesign, whose `soft_seq` always
softmaxes, and rejects the options that would then have no effect rather than ignoring them.

## Choosing settings

`soft=1, hard=0` is the default because it is the only setting whose forward value is a
probability distribution and whose gradient is smooth. `hard=1` is the straight-through
estimator: use it when the forward pass should see the discrete sequence the design will become.

Lowering `soft` below 1 feeds raw, unnormalized values to a model trained on distributions. That
is mathematically well defined and differentiable, and it is outside the training distribution of
every model here. Whether any of these settings produces better designs is an experimental
question this repository does not answer.

## Checks

| File | Covers |
|---|---|
| `tests/tool_infra_tests/_relaxation_env_checks.py` | The torch definition: forward values and gradients across the blend grid, the straight-through route, probability mode, and the shared reference table. Runs in esm2's environment. |
| `tests/tool_infra_tests/_relaxation_jax_env_checks.py` | The JAX twin, against the same formula and the same reference table. Runs in proteinmpnn's environment. |
| `tests/tool_infra_tests/_alphafold2_relaxation_env_checks.py` | ColabDesign's `soft_seq` on both backends, with and without alpha scaling and bias. Runs in alphafold2's environment. |
| `tests/tool_infra_tests/_relaxation_reference.py` | The shared table itself, imported by the check scripts as a sibling module. |
| `tests/tool_infra_tests/test_gradient_contract.py` | Launches each suite with `run_in_env`, and covers the config bounds in-process. |

The numerical suites need no model weights and no GPU — the formula is pure arithmetic — but they
do need a built tool environment, so they are marked `integration`.
