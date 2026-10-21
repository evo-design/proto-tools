# Gradient tools

Proto's gradient tools evaluate a model objective and return its gradient with respect to a continuous representation of a sequence. Instead of a string of discrete tokens, the input is a matrix of shape `(L, V)`, with one row per sequence position and one column per token in Proto's interface vocabulary for that sequence type. A model's own tokenizer may be much larger; backends map between the two.

This page explains how to prepare those inputs, interpret the returned gradients, work with the shared input types, and control how a continuous state is relaxed before it reaches a model.

## Logits, probabilities, and temperature

The `temperature` field determines how a gradient tool interprets its input matrix:

| `temperature` | Input values | What the model uses | Returned gradient |
|---|---|---|---|
| Positive number (default `1.0`) | Logits | `softmax(logits / temperature)` | With respect to the supplied logits |
| `None` | Probabilities | The supplied distribution, unchanged | With respect to the supplied probabilities |

The field is called `logits` in both cases for API compatibility. `temperature` is part of the tool **Input**, rather than its Config, and must be a positive, finite number or `None`.

For most gradient-based sequence optimization, use the default. You can update logits freely between iterations, and softmax will always turn them into a normalized distribution. Lower temperatures concentrate probability on the highest-scoring tokens; higher temperatures make the distribution more uniform.

Set `temperature=None` when you need the model to evaluate an existing probability distribution *without changing it*. Each active row must contain finite, nonnegative values that sum to 1, within an absolute tolerance of `1e-6`. The returned gradient is then with respect to those probabilities. Directly updating probabilities with ordinary gradient descent may produce invalid distributions, so this mode is generally intended for evaluation or algorithms that explicitly maintain the probability constraints.

For example, ESM2 supports both forms:

```python
# Optimize logits: softmax is applied by the backend.
inputs = ESM2GradientInput(logits=my_logits, temperature=1.0)

# Evaluate an existing probability distribution without applying softmax again.
inputs = ESM2GradientInput(logits=my_probabilities, temperature=None)
```

## Starting from probabilities

If you have a probability distribution but want to optimize **logits**, use the input class's `from_probabilities()` factory:

```python
inputs = ESM2GradientInput.from_probabilities(
    my_probabilities,
    temperature=0.5,
)

# Other required tool inputs can be supplied in the same call.
inputs = ProteinMPNNGradientInput.from_probabilities(
    my_probabilities,
    temperature=0.5,
    structure=my_structure,
)
```

The factory computes `logits = temperature * log(probabilities)` and sets the matching temperature on the request. Applying `softmax(logits / temperature)` then recovers the original distribution, up to floating-point precision. Specifying both together avoids accidentally constructing logits at one temperature and evaluating them at another.

For ordinary sequence inputs, the factory accepts only **strictly positive** probabilities. An exact zero would require an infinite logit, and Proto does not transmit infinite logits. Mixed protein/DNA inputs are the exception: the conversion operates within each permitted modality, preserving the required zeros in other columns. It also does not silently replace zeros with small positive values, because that would change the supplied distribution.

If you need to preserve exact zeros, pass a valid distribution with `temperature=None` instead. If you need to *permanently forbid* tokens during optimization, use a constraint supported by the particular tool (for example, AlphaFold2's `excluded_amino_acids`). A common, user-facing masked-softmax input is not yet available.

The same factory pattern supports other input layouts: batched nucleotide tools accept a distribution per batch item, AbLang accepts `heavy_chain` and/or `light_chain` distributions, and mixed protein/DNA tools handle the allowed vocabulary at each position when converting probabilities.

## Shared sequence types

`SequenceLogits` is a Pydantic `RootModel` that validates matrix shape and finite values. Its vocabulary-specific subclasses define the number and ordering of columns:

| Type | Columns | Vocabulary |
|---|---:|---|
| `ProteinSequenceLogits` | 20 | `ACDEFGHIKLMNPQRSTVWY` |
| `NucleotideSequenceLogits` | 4 | `ACGT` |
| `CodonSequenceLogits` | 64 | DNA codons in lexicographic order |
| `MixedSequenceLogits` | 24 | `ACDEFGHIKLMNPQRSTVWYacgt` |

Callers can still supply ordinary nested lists. Pydantic parses them into the appropriate sequence type, and serialization produces the same nested-list JSON format. In Python, `.root` exposes the underlying matrix; `SequenceLogits` also supports indexing, iteration, `len()`, and equality against a plain nested list.

`GradientInput` owns temperature and the shared validation rules. Tool-specific input classes extend it with their required fields. AbLang, for example, holds an `AntibodyLogits` object containing separate heavy- and light-chain matrices; batched nucleotide tools hold a list of matrices. `GradientInput.sequence_logits()` exposes these matrices by field path so shared validation and tests can handle them consistently.

### Mixed protein/DNA inputs

Mixed-model tools additionally restrict which columns are valid at each position, based on the sequence template. Uppercase protein positions use the 20 amino acid columns, lowercase DNA positions use the four nucleotide columns, and strand markers or ambiguous positions can represent inactive context.

With `temperature=None`, active rows must sum to 1 and have zero probability in all disallowed columns. Inactive rows are instead required to be **all zeros**, in either temperature mode. The restrictions come from the internal `GradientInput.allowed_tokens()` hook, not from a public `allowed_mask` field.

## Relaxation: the `soft` and `hard` controls

Every gradient tool turns the supplied matrix into the representation its model consumes through the same two coefficients. They live on `RelaxationConfig`, which every gradient config inherits. With `X` the supplied matrix, `T` the temperature, and `preprocess` whatever a model applies to logits before its own softmax (the identity for all but AlphaFold2):

```
P     = softmax(preprocess(X) / T)       # or X itself when temperature is None
D_ste = stop_gradient(one_hot(argmax(P)) - P) + P
U     = (1 - soft) * X + soft * P
M     = (1 - hard) * U + hard * D_ste
```

`M` is what reaches the model. The defaults — `temperature=1.0`, `soft=1.0`, `hard=0.0` — reduce it to a plain `softmax(X / T)`.

- **`soft`** interpolates from the raw input matrix (0) to the softmax probabilities (1). The `(1 - soft)` branch is the caller's own matrix, not an argmax one-hot, so it keeps a derivative: at `soft=0` the gradient is the upstream vector itself.
- **`hard`** mixes in a straight-through estimator, whose forward value is a one-hot and whose gradient flows through `P`. At `hard=1` both the forward value and the surrogate gradient are independent of `soft`.

Differentiating `M` gives an identity route scaled by `(1 - hard) * (1 - soft)` plus a softmax route scaled by `(1 - hard) * soft + hard`, so no documented setting leaves a zero gradient. A relaxation that forwarded a constant would let an optimizer run without ever moving, and nothing would raise, so a check pins that too.

Use the default for ordinary optimization: it is the only setting whose forward value is a probability distribution and whose gradient is smooth. Set `hard=1.0` when the forward pass should see the discrete sequence the design will become. Lowering `soft` below 1.0 feeds raw, unnormalized values to a model trained on distributions — well defined and differentiable, but outside the training distribution of every model here. Whether any of these settings produces better designs is an experimental question this repository does not answer.

With `temperature=None` the supplied matrix is already `P`, so `U` mixes `X` with itself and the straight-through term adds `X - stop_gradient(X)`: the forward value is `X` wherever `hard < 1`, and the gradient is the upstream vector at every setting.

### What does not move with the blend

- **Masked-model targets.** ESM2, AbLang, CodonFM, Minerva, and GLM2 take each site's target from `P`, so the objective does not shift with `soft` or `hard`.
- **ProteinMPNN's negative-log-likelihood labels.** They are the decoded sequence — a hard one-hot at every setting. Softmax is monotonic, so they come from `argmax(X)` directly and need no branch for probability mode.

### Implementations

| Where | What |
|---|---|
| `standalone_helpers.relaxation.relaxed_sequence` | The torch definition; every torch backend calls it. |
| `standalone_helpers.relaxation.relaxed_sequence_jax` | The JAX twin, for environments with no torch (ProteinMPNN). |
| `standalone_helpers.mixed_mlm` | Re-derives the blend for mixed protein/DNA models, whose `P` is normalized within each position's modality block. Every term is masked, so the raw-input branch cannot weight a forbidden column. |
| ColabDesign's `soft_seq` | AlphaFold2 relaxes inside ColabDesign, on both the upstream and Germinal backends. `preprocess` is Germinal's `alpha` scaling and any `bias_redesign`, which therefore move `P` but not the raw-input branch. |

Each exists for a reason an import cannot remove — a missing array library, a modality-masked softmax, an upstream that owns its own reparameterization — so what keeps them one contract is that each is checked against the formula above, with the torch and JAX definitions additionally pinned to one shared table of forward values and gradients. Both helpers import their array library lazily and are not re-exported from the package `__init__`, so an environment with only one of torch and JAX can still import `standalone_helpers`.

AlphaFold2 expresses probability mode as `soft=0` inside ColabDesign, whose `soft_seq` always softmaxes, and rejects the options that would then have no effect — `soft`, `hard`, `excluded_amino_acids`, and `bias_redesign` — instead of ignoring them silently.

Backends also map Proto's public vocabulary order to their own native token order; those native orders remain defined with each toolkit rather than assumed to be identical across models.

## Outputs and testing

A `GradientOutput` contains `loss`, `metrics`, `vocab`, and `gradient`. The gradient uses the input's shape and Proto vocabulary order. When `compute_gradient=False`, `gradient` is `None` and the tool performs forward-only scoring without changing the loss or metrics.

AbLang is a special case: it combines heavy- and light-chain inputs and returns one `(Lh + Ll, 20)` gradient matrix, with the heavy-chain rows first. Split the result at the heavy-chain length to recover each chain's gradient.

Shared contract tests live in `tests/tool_infra_tests/test_gradient_contract.py`. They cover registered gradient tools, input validation, the common interface, and the relaxation controls, and they launch the numerical suites below through `run_in_env`. Backend-specific tests check numerical gradient agreement, including finite-difference comparisons where appropriate.

| File | Covers |
|---|---|
| `_relaxation_env_checks.py` | The torch definition: forward values and gradients across the blend grid, the straight-through route, probability mode, and the shared reference table. Runs in esm2's environment. |
| `_relaxation_jax_env_checks.py` | The JAX twin, against the same formula and the same reference table. Runs in proteinmpnn's environment. |
| `_alphafold2_relaxation_env_checks.py` | ColabDesign's `soft_seq` on both backends, with and without alpha scaling and bias. Runs in alphafold2's environment. |
| `_relaxation_reference.py` | The shared table itself, imported by the check scripts as a sibling module. |

Those suites need no model weights and no GPU — the formula is pure arithmetic — but they do need a built tool environment, so they are marked `integration`.
