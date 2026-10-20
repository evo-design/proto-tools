# Gradient tools

Proto's gradient tools evaluate a model objective and return its gradient with respect to a continuous representation of a sequence. Instead of a string of discrete tokens, the input is a matrix of shape `(L, V)`, with one row per sequence position and one column per token in Proto's interface vocabulary for that sequence type. A model's own tokenizer may be much larger; backends map between the two.

This page explains how to prepare those inputs, interpret the returned gradients, and work with the shared input types.

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

## Backend behavior

Backends share the same temperature and gradient-coordinate contract, and relax a discrete sequence the same way:

- **ESM2, AbLang, CodonFM, Minerva, GLM2, ESMFold, Parade, and Malinois** use the shared `soft` and `hard` blend coefficients from `RelaxationConfig`. ProteinMPNN still exposes the older `use_ste` boolean and is being migrated.
- **AlphaFold2** performs the same blend inside ColabDesign, which is where the convention comes from: `soft=0` selects the raw input matrix and `hard=1.0` is the straight-through estimator.

Shared relaxation helpers handle the standard softmax and straight-through operations. Backends also map Proto's public vocabulary order to their own native token order; those native orders remain defined with each toolkit rather than assumed to be identical across models.

AlphaFold2 performs its relaxation inside ColabDesign. In probability mode, the backend arranges for the supplied distribution to pass through unchanged. Options that would have no effect in that mode—including `soft`, `hard`, `excluded_amino_acids`, and `bias_redesign`—are rejected instead of silently ignored.

## Outputs and testing

A `GradientOutput` contains `loss`, `metrics`, `vocab`, and `gradient`. The gradient uses the input's shape and Proto vocabulary order. When `compute_gradient=False`, `gradient` is `None` and the tool performs forward-only scoring without changing the loss or metrics.

AbLang is a special case: it combines heavy- and light-chain inputs and returns one `(Lh + Ll, 20)` gradient matrix, with the heavy-chain rows first. Split the result at the heavy-chain length to recover each chain's gradient.

Shared contract tests live in `tests/tool_infra_tests/test_gradient_contract.py`. They cover registered gradient tools, input validation, and the common interface. Backend-specific tests check numerical gradient agreement, including finite-difference comparisons where appropriate.
