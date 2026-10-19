# Gradient tools

Every gradient tool differentiates a backend objective with respect to a *relaxed* sequence
state: a real-valued `(L, V)` matrix standing in for a discrete sequence, where `V` is the
tool's vocabulary size. This note is the canonical reference for the input and
gradient-coordinate contract they share, and for the places a backend is allowed to differ.

## The contract

A gradient input carries one `temperature` and one or more relaxed matrices. `temperature`
fixes both how the matrices are read and which space the returned gradient lives in:

| `temperature` | `logits` means | Backend applies | Returned gradient |
|---|---|---|---|
| positive float (default `1.0`) | unnormalized logits | `softmax(logits / T)` | logit-space |
| `None` | a probability distribution | nothing; used verbatim | probability-space |

The field is named `logits` in both modes. Each tool's README states what it means there.

Backends take the gradient with respect to the raw incoming matrix in both cases. It is
"logit-space" only because a softmax sits in the graph; remove the softmax and the same
derivative is with respect to the probabilities themselves.

`temperature` lives on the **Input**, not the Config: it describes the state being passed in,
not how the model runs. It must be positive and finite, or `None`.

## Probability-mode validation

With `temperature=None`, every **active** row of every matrix must be:

- finite (guaranteed for all rows by the matrix type itself),
- nonnegative, and
- summing to `1.0` within an absolute tolerance of `1e-6`.

**Inactive rows** — positions that are not design variables — are exempt from the simplex check
and must instead be exactly zero, in *both* temperature modes.

Which rows are active comes from an optional per-matrix `allowed_mask`: an `(L, V)` boolean grid
returned by `GradientInput.allowed_tokens()`, where `None` means every column is permitted
everywhere. A row that is entirely `False` is inactive. A row with some `True` entries is active
and must additionally leave its non-permitted columns at zero.

Only the mixed protein/DNA models override the hook today. They derive the mask from their
template string: columns 0-19 for an uppercase protein token, 20-23 for a lowercase DNA token,
and no permitted column for `+`/`-` strand markers or ambiguous symbols. Every other tool
inherits the unrestricted case.

`allowed_tokens()` is an internal validation hook, not a wire field. There is deliberately no
public `allowed_mask` input yet: a user-facing masking API should be supported consistently
across every applicable backend rather than tool by tool, and only the mixed models have a
masking seam today. Forbidden tokens are expressed in the meantime by passing exact zeros with
`temperature=None`, or through a tool's own field, such as AlphaFold2's `excluded_amino_acids`.

**Infinite logits are never transmitted.** Matrix construction rejects non-finite values, so a
`-inf` column is not a way to forbid a token.

## Starting from a probability distribution

`SequenceLogits.from_probabilities(P, temperature=T)` returns `T * log(P)`, which
`softmax(logits / T)` inverts exactly, so a caller can seed an optimizer from a distribution
and keep working in logit space.

It requires **strictly positive** probabilities. An exact zero maps to an infinite logit, so it
is refused rather than quietly floored — there is no epsilon substitution, because silently
changing the supplied distribution is worse than failing. The error names the two intents an
exact zero can carry:

- a zero that should stay zero: pass the distribution directly with `temperature=None`, which
  preserves it exactly;
- a token that should be forbidden for the whole run: that belongs in a masked-softmax
  interface, which does not exist yet.

## Class hierarchy

```
SequenceLogits                     (L, V) RootModel; shape, rectangularity, finiteness, VOCAB
├── ProteinSequenceLogits          V=20, ACDEFGHIKLMNPQRSTVWY
├── NucleotideSequenceLogits       V=4,  A,C,G,T
├── CodonSequenceLogits            V=64, lexicographic DNA codons
└── MixedSequenceLogits            V=24, ACDEFGHIKLMNPQRSTVWYacgt

AntibodyLogits                     composite; exposes iter_sequence_logits() for its chains
                                   heavy_chain / light_chain

GradientInput                      temperature, sequence_logits(), allowed_tokens(), validation
├── ProteinGradientInput           + logits: ProteinSequenceLogits
│   ├── ESM2GradientInput, ProteinMPNNGradientInput,
│   └── ESMFoldGradientInput, AlphaFold2GradientInput
├── BatchedNucleotideGradientInput + logits: list[NucleotideSequenceLogits]  ((B, L, 4))
│   └── MalinoisGradientInput, ParadeGradientInput
├── CodonFMGradientInput           + logits: CodonSequenceLogits
├── MixedSequenceGradientInput     + sequence template, logits: MixedSequenceLogits
└── AbLangGradientInput            + antibody: AntibodyLogits
```

A `SequenceLogits` is a Pydantic `RootModel` over `list[list[float]]`: it parses from and
serializes to ordinary nested lists, so the JSON request shape is exactly what it was before
the type existed. It supports `len()`, iteration and indexing over position rows; `.root` is
the plain nested list to put in a worker payload.

`GradientInput.sequence_logits()` returns every matrix on an input keyed by a dotted path
(`"logits"`, `"antibody.heavy_chain"`, `"logits[0]"`). That is how shared validation and the
cross-backend tests find the relaxed state without knowing a tool's field layout.

Vocabulary subclasses live beside the vocabulary they wrap — `CodonSequenceLogits` in the
CodonFM toolkit, `MixedSequenceLogits` in `mixed_data_models.py` — so there is one definition
of each column order.

## Backend relaxation

`standalone_helpers.relaxation.relaxed_probs()` is the shared conversion from a transported
matrix to the probabilities a model consumes:

```python
probs   = logits if temperature is None else softmax(logits / temperature)
relaxed = (1 - soft) * one_hot(argmax(probs)) + soft * probs
# hard > 0 mixes in a straight-through estimator
```

`resolve_temperature()` coerces the transported value, returning `None` for probability mode
and rejecting non-positive or non-finite temperatures.

Two families of backend consume it:

- **Masked-LM / STE** (esm2, ablang, codonfm, proteinmpnn, minerva, glm2): a boolean `use_ste`
  config switches between soft blended embeddings and a hard forward pass with soft-probability
  gradients. They call the softmax at `soft=1, hard=0`.
- **ColabDesign soft/hard** (esmfold, parade, malinois, alphafold2): float `soft` and `hard`
  configs interpolate continuously. `hard=1.0` is the same straight-through surrogate that
  `use_ste=True` selects in the other family; `soft` has no masked-LM analogue.

Both families are genuine, not an accident: keep the distinction explicit rather than forcing
one vocabulary onto the other.

### AlphaFold2

AlphaFold2 does not relax its own input. `temperature`, `soft` and `hard` are written into
ColabDesign's `opt` dict and `soft_seq()` performs the blend, including the Germinal fork's
`alpha` scaling and persistent bias.

Probability mode is still reachable without patching ColabDesign: `soft_seq()` blends its
softmax branch against the raw input, so `opt["soft"] = 0` makes the supplied distribution
reach AlphaFold verbatim, and `aux["grad"]["seq"]` is then the probability-space derivative.
The worker sets `soft=0, hard=0` and pins `temp` to keep the unused softmax branch finite.

Because the bias terms feed only that unused branch, `soft`, `hard`,
`excluded_amino_acids` and `bias_redesign` have no effect in probability mode. They are **refused** when combined with
`temperature=None` rather than silently ignored.

## Output

`GradientOutput` carries `gradient`, `loss`, `metrics` and `vocab`. The gradient matches the
input's shape and column order; `vocab` is the shared column ordering for both.

`gradient` is `None` when a tool runs with `compute_gradient=False`, which every backend
supports for forward-only scoring. The loss and metrics are identical in both modes.

AbLang is the one tool whose input holds several matrices and whose output holds one: the
chains are concatenated heavy-then-light into a single `(Lh + Ll, 20)` tensor, which is what
AbLang actually differentiates. Split the gradient back apart at `len(antibody.heavy_chain)`.

Workers may return the gradient either as nested lists or as a compressed-array dict;
`decompress_result` normalizes that before the Output model sees it.

## Tests

`tests/tool_infra_tests/test_gradient_contract.py` holds the cross-backend contract tests. They
are parametrized over every registered tool whose input inherits `GradientInput`, so a new
gradient backend is covered as soon as it is registered, and they assert that no `*-gradient`
tool sits outside the hierarchy. They are pure validation: CPU-only and environment-free.

Numerical agreement is checked per backend against finite differences; see
`tests/masked_models_tests/_mixed_checks.py` for the reference pattern.
