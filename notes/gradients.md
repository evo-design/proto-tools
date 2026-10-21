# How gradient tools work in Proto

Proto's gradient tools let you optimize a biological sequence using derivatives from a predictive model. Instead of supplying only a discrete amino acid or nucleotide sequence, you supply a continuous matrix with one row per position and one column per possible token. The tool evaluates its objective and returns a loss and, when requested, a gradient that can be used to update the sequence.

This guide explains what goes into a gradient tool, how Proto turns that input into something the model can evaluate, and how to interpret the result. The same conventions apply across the gradient tools, with a few model-specific details described below.

## 1. Sequence inputs and temperature

A relaxed sequence is a matrix of shape `(L, V)`, where `L` is the sequence length and `V` is the size of Proto's vocabulary for that model. A protein sequence, for example, uses an `(L, 20)` matrix. Each row contains one value for every possible amino acid at that position.

The `temperature` field determines how the tool interprets those values:

| `temperature` | Input represents | Probability calculation | Returned gradient |
| --- | --- | --- | --- |
| Positive float (default `1.0`) | Unnormalized logits | `P = softmax(preprocess(X) / temperature)` | With respect to the supplied logits `X` |
| `None` | An existing probability distribution | No new softmax; `P = X` | With respect to the supplied probabilities `X` |

Here, `preprocess(X)` is normally just `X`. AlphaFold2 is an exception: ColabDesign can scale the logits and add bias before softmax.

The input field remains named `logits` in either mode for compatibility. `temperature` belongs to the tool's **Input**; the `soft` and `hard` controls discussed below belong to its **Config**.

For most gradient-based optimization, use a positive temperature. Logits can be updated without constraints, and softmax converts them into a valid probability distribution. Lower temperatures favor the highest-scoring token at each position; higher temperatures spread the probability more evenly.

```python
# Start from unnormalized amino acid scores.
inputs = ESM2GradientInput(logits=my_logits, temperature=1.0)

# Or provide normalized probabilities directly.
inputs = ESM2GradientInput(logits=my_probabilities, temperature=None)
```

**A note on terminology:** The probability matrix `P` is not always exactly what the predictive model receives. Proto can further blend it with raw logits or a hard one-hot sequence, depending on `soft` and `hard`.

## 2. How `soft` and `hard` change the model input

Every gradient tool exposes two relaxation controls through its config:

- **`soft`** blends the raw input matrix with its softmax probabilities. At `soft=0`, the blend uses raw values; at `soft=1`, it uses probabilities.
- **`hard`** blends in a one-hot sequence using a straight-through estimator (STE). At `hard=1`, the model evaluates a hard sequence, while backpropagation follows a differentiable surrogate.

Both controls are finite numbers between `0` and `1`. Their shared defaults are **`soft=1.0` and `hard=0.0`**. The default `temperature` is `1.0`.

### The calculation

Let `X` be the original input matrix, and let `P` be its probability representation. Proto constructs the following values:

```text
P     = softmax(preprocess(X) / temperature)
D     = one_hot(argmax(P))
D_ste = stop_gradient(D - P) + P

U     = (1 - soft) * X + soft * P
M     = (1 - hard) * U + hard * D_ste
```

`M` is the relaxed sequence representation passed to the model's input-processing pathway. In the forward pass, `D_ste` has the same values as `D`: exactly one selected token at each position. In the backward pass, its gradient follows `P` rather than the nondifferentiable `argmax` operation.

The settings correspond to these modes:

| `soft` | `hard` | Representation the model sees | How gradients flow back |
| ---: | ---: | --- | --- |
| `1` | `0` | Softmax probabilities `P` | Through softmax |
| `1` | `1` | Hard one-hot `D` | Through the STE surrogate |
| `0` | `0` | Raw input `X` | Directly through the input |
| `0.5` | `0` | `0.5 X + 0.5 P` | Through both raw and softmax branches |
| `1` | `0.5` | `0.5 P + 0.5 D` | Through softmax and the STE surrogate |

For example, consider a position with three possible tokens and logits `[2, 1, 0]`. At temperature `1`, softmax produces approximately `[0.665, 0.245, 0.090]` and the hard one-hot sequence is `[1, 0, 0]`.

| Mode | Example values passed forward |
| --- | --- |
| Raw (`soft=0, hard=0`) | `[2, 1, 0]` |
| Soft (`soft=1, hard=0`) | `[0.665, 0.245, 0.090]` |
| Hard STE (`soft=1, hard=1`) | `[1, 0, 0]` |
| Half raw, half soft (`soft=0.5, hard=0`) | `[1.333, 0.622, 0.045]` |
| Half soft, half hard (`soft=1, hard=0.5`) | `[0.833, 0.122, 0.045]` |

These numbers illustrate the shared blending formula without model-specific scaling or masking.

When `hard=1`, the result no longer depends on `soft`: the entire forward representation is one-hot, and its backward path uses the STE.

### Why keep raw and blended modes?

Raw and intermediate representations give an optimizer more freedom than a normalized probability matrix. ColabDesign uses this kind of progression during sequence design. But raw values can be outside the inputs on which other predictive models were trained. Proto supports these modes mathematically; that does **not** establish that they improve sequence design or produce better-calibrated predictions.

The shared implementation preserves a gradient path for each mode. That does not guarantee that every objective produces a nonzero gradient: a gradient can still be zero because of the model, the loss, or local saturation.

## 3. Using existing probabilities (`temperature=None`)

Set `temperature=None` when your input already contains probabilities and you don't want Proto to apply softmax again. Every active row must contain finite, nonnegative values that sum to `1.0` within an absolute tolerance of `1e-6`. The gradient is then expressed in **probability coordinates**, rather than logit coordinates.

To pass the distribution through unchanged, use the ordinary relaxation settings (`soft=1, hard=0`):

```python
inputs = ESM2GradientInput(logits=my_probabilities, temperature=None)
# Ordinary default config: soft=1.0, hard=0.0
```

There is an important distinction between **skipping softmax** and **skipping all relaxation**. In the generic formula, when `temperature=None`, `P=X` and the first blend reduces to `U=X`. If `hard` is greater than zero, however, the forward pass also mixes in the one-hot representation:

```text
M = (1 - hard) * X + hard * one_hot(argmax(X))  # forward values
```

The STE supplies a surrogate backward path. Therefore, `temperature=None` by itself does **not** imply that explicitly enabled hardening leaves the model input unchanged. Use `hard=0` when preserving the supplied distribution is the goal. AlphaFold2 handles this mode specially: its adapter bypasses ColabDesign's softmax branch and rejects incompatible relaxation options instead of silently ignoring them.

Because unconstrained gradient updates can make probabilities negative or cause rows to stop summing to one, direct probability optimization requires a method that maintains the simplex constraints.

## 4. Starting with probabilities but optimizing logits

If you have probabilities and want to optimize **logits**, use the gradient input class's `from_probabilities()` factory:

```python
inputs = ESM2GradientInput.from_probabilities(
    my_probabilities,
    temperature=0.5,
)

# Extra required inputs can be supplied in the same call.
inputs = ProteinMPNNGradientInput.from_probabilities(
    my_probabilities,
    temperature=0.5,
    structure=my_structure,
)
```

For strictly positive probabilities, the factory calculates:

```text
logits = temperature * log(probabilities)
```

It sets the matching temperature on the constructed input, so `softmax(logits / temperature)` reconstructs the original distribution (up to floating-point precision). This avoids accidentally converting at one temperature and evaluating at another.

The factory rejects exact zeros in ordinary sequence distributions rather than silently replacing them with small positive values. Exact zeros would require infinite logits, which Proto does not accept. To preserve zeros, supply the distribution directly with `temperature=None`. Permanently excluding tokens during logit optimization requires a model-supported constraint; Proto does not yet expose a universal public `allowed_mask`.

Mixed protein/DNA models are a special case: their factories convert probabilities within each position's permitted vocabulary block, preserving zeros outside that block. AbLang supports separate heavy- and light-chain distributions, and batched nucleotide tools accept a distribution for each batch item.

**AlphaFold2 caveat:** Its effective `P` can also depend on ColabDesign's logit scaling and bias. The formula above guarantees reconstruction for an ordinary softmax on the supplied logits, not necessarily for an additional model-specific preprocessing step.

## 5. Sequence types and vocabulary ordering

`SequenceLogits` is a Pydantic `RootModel` that validates matrix shape and finite values. Subclasses define the vocabulary and number of columns:

| Type | Columns | Order |
| --- | ---: | --- |
| `ProteinSequenceLogits` | 20 | `ACDEFGHIKLMNPQRSTVWY` |
| `NucleotideSequenceLogits` | 4 | `ACGT` |
| `CodonSequenceLogits` | 64 | DNA codons, lexicographic order |
| `MixedSequenceLogits` | 24 | `ACDEFGHIKLMNPQRSTVWYacgt` |

Callers can still send ordinary nested lists. Pydantic converts them into the appropriate types without changing the JSON request format. In Python, `.root` exposes the underlying list of rows.

`GradientInput` provides the shared temperature and validation rules. Tool-specific input classes add fields such as ProteinMPNN's structure or AbLang's paired heavy and light chains. `GradientInput.sequence_logits()` identifies the relaxed matrices in these different input shapes, allowing common validation and tests.

### Mixed protein/DNA models

Minerva and GLM2 combine protein and DNA positions in one `(L, 24)` matrix. Their sequence templates determine the permitted columns at each position: protein positions use the first 20 columns, DNA positions use the last four, and certain template positions are inactive context.

The models normalize `P` within each position's allowed vocabulary, and mask **all** terms in the relaxation calculation. Thus, even in raw mode, values in forbidden columns cannot contribute to the model input. Inactive context rows must be exactly zero in either temperature mode. In probability mode, active rows must sum to one and assign zero weight to forbidden columns.

These constraints come from an internal `allowed_tokens()` validation hook, not a user-facing `allowed_mask` field.

### AlphaFold2 / ColabDesign

AlphaFold2 uses ColabDesign's own `soft_seq()` implementation rather than the shared helper. The meaning of `soft` and `hard` is the same, but the distribution `P` may be computed from scaled and biased logits. For example, the Germinal backend can use `softmax((2 * X + bias) / T)`; the raw branch still uses the original `X`.

In probability mode, the adapter selects the raw-input branch internally so validated probabilities pass through unchanged. It rejects settings that would otherwise be ignored, including incompatible relaxation and bias/exclusion options.

## 6. What the tool returns

A gradient tool returns a `GradientOutput` with:

- `loss`: the scalar model objective being evaluated;
- `metrics`: additional model-specific scores;
- `vocab`: the public token order associated with the output;
- `gradient`: derivatives with respect to the supplied sequence matrix, in the same shape and vocabulary order.

With positive temperature, the gradient is with respect to **logits**. With `temperature=None`, it is with respect to **probabilities**. In a hard-STE mode, it is a **surrogate gradient**: the forward pass evaluates one-hot inputs, but the backward pass follows the differentiable probability pathway.

Setting `compute_gradient=False` runs forward-only scoring and returns `gradient=None`.

AbLang returns a single matrix for paired heavy and light chains. The rows are concatenated heavy-first, then light; split at the heavy-chain length to recover each chain's gradient.

### How the objective relates to the relaxation

The relaxation controls change the input representation evaluated by the model. They do not change the rule used to choose training or scoring targets. The masked models (ESM2, AbLang, CodonFM, Minerva, and GLM2) derive their targets from `P`. ProteinMPNN uses decoded sequence identities for its negative-log-likelihood labels.

That **does not** mean the reported loss is unchanged when you adjust `soft` or `hard`: changing the model input can change its predictions and therefore its loss and gradient. It only means the target-selection rule is unchanged for fixed sequence logits.

## 7. Implementation and tests

The common numerical behavior is implemented in several places because the model environments and frameworks differ:

| Implementation | Role |
| --- | --- |
| `standalone_helpers.relaxation.relaxed_sequence` | Shared PyTorch relaxation |
| `standalone_helpers.relaxation.relaxed_sequence_jax` | Equivalent JAX relaxation, used by ProteinMPNN |
| `standalone_helpers.mixed_mlm` | Modality-masked version for mixed protein/DNA inputs |
| ColabDesign's `soft_seq()` | AlphaFold2's upstream implementation of the same blending convention |

The PyTorch and JAX helpers are checked against the same table of expected forward values and gradients. ColabDesign is checked against the same mathematical contract, including its preprocessing. Tests also cover the STE backward path, intermediate blends, probability mode, and shared configuration defaults.

The registry-wide contract checks live in `tests/tool_infra_tests/test_gradient_contract.py`. The framework-specific checks—`_relaxation_env_checks.py`, `_relaxation_jax_env_checks.py`, and `_alphafold2_relaxation_env_checks.py`—run in their respective tool environments through `run_in_env`. These checks do not require model weights or a GPU, although they do require the corresponding environments to be built.
