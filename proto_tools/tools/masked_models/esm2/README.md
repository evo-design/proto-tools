<a href="https://bio-pro.mintlify.app/tools/masked-models/esm2"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESM2

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview
ESM2 (Evolutionary Scale Modeling 2) is Meta AI's [protein language model](https://www.evolutionaryscale.ai/blog/esm-cambrian) trained on millions of protein sequences from [UniRef](https://www.uniprot.org/help/uniref). It provides sequence embeddings, per-position amino acid logits, sequence mutation (sampling), and sequence scoring (MLM pseudo-perplexity). ESM2 offers multiple model sizes from 8M to 15B parameters, balancing quality and computational cost.

This package also includes `esm2-gradient`, a relaxed-sequence gradient tool that computes the masked pseudo-log-likelihood objective over a continuous L×20 logits distribution and returns its gradient with respect to the input. It can be used as a differentiable structure-free pLM loss inside MCMC, gradient descent, or any optimization loop over relaxed protein sequences.

## Background

**What are protein language models?**
Protein language models (pLMs) learn the "grammar" of proteins from evolutionary data. They capture:
- **Sequence conservation**: Which residues are essential for function
- **[Co-evolution](https://en.wikipedia.org/wiki/Coevolution)**: Pairs of residues that evolve together (often in contact)
- **Structural constraints**: Patterns that define [secondary/tertiary structure](https://en.wikipedia.org/wiki/Protein_structure)
- **Functional motifs**: Binding sites, active sites, [post-translational modifications](https://en.wikipedia.org/wiki/Post-translational_modification)

**Why embeddings are useful:**
ESM2 embeddings encode rich biological information:
- Similar proteins cluster in embedding space
- Embeddings predict structure, function, and localization
- Mean-pooled embeddings work as fixed-length sequence descriptors

**Why logits are useful:**
Per-position logits indicate the model's confidence in each amino acid:
- High logits = evolutionarily preferred (conserved)
- Low logits = tolerated or deleterious positions
- Comparing wild-type vs mutant logits predicts variant effects
- Logits are returned over 20 canonical amino acids in fixed order: `ACDEFGHIKLMNPQRSTVWY`

## Tools

### ESM2 Embeddings (`esm2-embedding`)

Extract protein sequence embeddings and logits using ESM2.

Uses ESM2 from Meta AI to extract contextualized embeddings and per-position
logits for protein sequences. The model is automatically loaded on-demand.
Supports local GPU execution via isolated Python environments.

### ESM2 Gradient (`esm2-gradient`)

Compute ESM2 masked PLL gradient with respect to relaxed protein logits.

### ESM2 Sampling (`esm2-sample`)

Sample masked positions in protein sequences using ESM2.

The `preprocess` hook on :class:`ESM2SampleConfig` applies the masking
strategy before this function runs, so `inputs.sequences` already
contain `_` at positions to sample.

### ESM2 Scoring (`esm2-score`)

Score protein sequences using ESM2 language model.

Computes MLM pseudo-perplexity by masking each position individually and
computing $P(x_i | x_{-i})$. Uses batched processing for efficiency.

Ambiguous amino acids (X, B, Z, etc.) are excluded from the perplexity
calculation using the industry-standard exclusion strategy. Only positions
with standard amino acids (20 canonical AAs) contribute to log-likelihood
and perplexity metrics.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `esm2-embedding` | Extract embeddings and logits | Embeddings, logits, attention masks |
| `esm2-gradient` | Masked PLL gradient over relaxed logits | Gradient, loss, metrics, vocab |
| `esm2-sample` | Mutate sequences using model | Modified sequences |
| `esm2-score` | Score sequences via MLM pseudo-perplexity | Per-sequence metrics, optional logits |

## Model Variants

| Checkpoint | Parameters | Layers | Embedding Dim | Speed | Quality |
|------------|------------|--------|---------------|-------|---------|
| `esm2_t6_8M_UR50D` | 8M | 6 | 320 | Fastest | Basic |
| `esm2_t12_35M_UR50D` | 35M | 12 | 480 | Fast | Good |
| `esm2_t30_150M_UR50D` | 150M | 30 | 640 | Medium | Better |
| `esm2_t33_650M_UR50D` | 650M | 33 | 1280 | Slower | High |
| `esm2_t36_3B_UR50D` | 3B | 36 | 2560 | Slow | Higher |
| `esm2_t48_15B_UR50D` | 15B | 48 | 5120 | Slowest | Best |

**Model selection guidance:**
1. **Default choice**: `esm2_t33_650M_UR50D` offers best quality/speed tradeoff for most tasks.
2. **Memory constrained**: Use `esm2_t30_150M_UR50D` or smaller.
3. **Maximum quality**: Use `esm2_t48_15B_UR50D` (requires >40GB GPU).

## Execution Modes

- **Local GPU/CPU**: Loads the model on-demand. Use `device="cuda"`, `"cpu"`, or `"mps"`.

## How It Works

ESM2 is a masked language model (similar to BERT) trained on protein sequences. It learns to predict masked amino acids from surrounding context, capturing evolutionary and structural patterns.

- **Embeddings**: Forward pass through the model produces per-position hidden states. Mean-pooling across positions yields a fixed-length sequence descriptor.
- **Sampling**: Positions are selected by a decoding method (entropy, max_logit, or random), masked, and resampled from the model's predicted distribution at a given temperature.
- **Scoring**: Each position is masked one at a time, and the model's log-probability of the true amino acid is recorded. Aggregated scores give pseudo-perplexity (lower = more "natural" sequence).
- **Gradient**: A relaxed `(L, 20)` distribution is mixed against ESM2's per-residue token embeddings to form a soft input, each amino-acid position is masked in turn, and a per-chunk backward pass accumulates the gradient of the mean masked negative log-likelihood with respect to the input logits while keeping model parameters frozen. The optional Straight-Through Estimator runs the forward pass on hard one-hot tokens while gradients still flow through the soft probabilities.

## Input Parameters

### Embeddings Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences (amino acid strings) |

### Sampling Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences to mutate |

### Scoring Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences to score |

### Gradient Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `List[List[float]]` | Relaxed sequence logits with shape `(L, 20)` in canonical amino-acid order `ACDEFGHIKLMNPQRSTVWY` |
| `temperature` | `float \| null` | Optional softmax temperature. When set, applies `softmax(input / temperature)` before gradient computation. When `null` (default), input is used as-is — callers provide a pre-computed distribution |

## Configuration

### Embeddings Tool (`ESM2EmbeddingsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |
| `verbose` | `bool` | `False` | Print progress messages |
| `return_logits` | `bool` | `False` | Include per-position logits in output |
| `repr_layer` | `int` | `-1` | Transformer layer index for embeddings (`-1` = last) |
| `truncation_seq_length` | `int` | `1022` | Truncate sequences exceeding this many residues (ESM2 cap is 1022) |

### Sampling Tool (`ESM2SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `masking_strategy` | `MaskingStrategy` | random 30% | Composite — see fields below |
| `sampling_method` | `Literal["single_pass", "iterative_refinement"]` | `"single_pass"` | `single_pass` fills every mask in one forward; `iterative_refinement` runs a MaskGIT-style loop |
| `temperature` | `float` | `1.0` | Softmax temperature for sampling |
| `top_p` | `float` | `1.0` | Nucleus threshold (iterative only); `1.0` disables |
| `num_steps` | `int` | `20` | Iterative-refinement decoding steps (iterative only) |
| `schedule` | `Literal["cosine", "linear"]` | `"cosine"` | Unmask schedule across rounds (iterative only) |
| `strategy` | `Literal["random", "entropy"]` | `"random"` | Per-round commit selection (iterative only) |
| `temperature_annealing` | `bool` | `True` | Anneal toward 0 across rounds (iterative only) |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `"cuda"` | Device for inference |
| `return_logits` | `bool` | `False` | Include per-position logits in output |

**`MaskingStrategy` fields** (nested, controls which positions to mask):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `Literal["random", "entropy", "max-logit"]` | `"random"` | Position-selection scoring method |
| `num_mutations` | `int \| None` | `None` | Exact number of positions to mask |
| `mask_fraction` | `float \| None` | `None` | Fraction of designable positions to mask (default ~30%) |
| `fixed_positions` | `list[int] \| None` | `None` | 1-indexed positions that must NOT be masked |
| `temperature` | `float` | `1.0` | Temperature for position selection (separate from sampling temperature) |

Use `sampling_method="iterative_refinement"` for higher-coherence joint sampling at multiple masked sites — slower (~num_steps× compute), but commits positions in rounds rather than independently.

### Scoring Tool (`ESM2ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `batch_size` | `int` | `1` | Masked variants per forward pass |
| `device` | `str` | `"cuda"` | Device for inference |
| `verbose` | `bool` | `False` | Print progress messages |
| `return_logits` | `bool` | `False` | Include per-position logits in output |

### Gradient Tool (`ESM2GradientConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `use_ste` | `bool` | `False` | When true, uses a Straight-Through Estimator: hard one-hot tokens in the forward pass with gradients flowing through soft probabilities. When false, uses soft blended embeddings directly |
| `compute_gradient` | `bool` | `True` | When true, runs backward pass and returns the gradient. Set `False` for forward-only log-likelihood scoring (e.g. MCMC proposal ranking); `gradient` is `None` in the output |
| `batch_size` | `int \| null` | `null` | AA positions per forward pass for batched PLL. `null` selects the backend default (32). Lower if OOM, higher for throughput |
| `seed` | `int \| null` | `null` | Optional PyTorch random seed for reproducibility |
| `device` | `str` | `"cuda"` | Device for inference |
| `verbose` | `bool` | `False` | Print progress messages |

### Parameter Guides

**Temperature guide for sampling:**
| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.5-0.7 | Conservative | Safer mutations |
| 1.0 | Standard | Model distribution |
| 1.5-2.0 | Creative | More diverse mutations |

## Output Specification

### ESM2EmbeddingsOutput (Embeddings)

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[SequenceEmbedding]` | Per-sequence embedding results (primary field) |

**`SequenceEmbedding` fields:**
| Field | Type | Description |
|-------|------|-------------|
| `mean_embedding` | `List[float]` | Mean-pooled embedding vector for one sequence |
| `attention_mask` | `List[int]` | Binary mask: 1 = valid position, 0 = padding |
| `logits` | `Optional[List[List[float]]]` | Per-position amino acid logits (seq_len, 20). Only present if `return_logits=True` |

### ESM2SampleOutput

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Mutated protein sequences |
| `logits` | `List[List[List[float]]]?` | Optional per-position logits (AA-only) |

### MaskedModelScoringOutput (Scoring)

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[MaskedModelScoringMetrics]` | One score object per input sequence |

**`MaskedModelScoringMetrics` fields:**
| Field | Type | Description |
|-------|------|-------------|
| `log_likelihood`, `avg_log_likelihood`, `perplexity` | `float` | Scalar metrics (attribute or mapping access) |
| `logits` | `List[List[float]]?` | Optional per-position logits (shape `[seq_len, 20]`) |
| `vocab` | `List[str]?` | Amino acid order (AA-only) |

### ESM2GradientOutput

| Field | Type | Description |
|-------|------|-------------|
| `gradient` | `List[List[float]] \| null` | Gradient matrix with the same `(L, 20)` shape and amino-acid column order as the input logits. `null` when `compute_gradient=False` (forward-only scoring) |
| `loss` | `float` | Mean negative log-likelihood over the L masked positions |
| `metrics` | `dict[str, Any]` | Auxiliary metrics: `log_likelihood`, `avg_log_likelihood`, `perplexity`, `sequence_length`, `model_checkpoint`, and `objective` (`"masked_pll"`) |
| `vocab` | `List[str]` | Amino-acid column ordering for both the input logits and the returned gradient — always canonical protein order `ACDEFGHIKLMNPQRSTVWY` |

## Interpreting Results

These thresholds are heuristics. Use them comparatively and validate for your task.

**For variant effect prediction (logit difference, heuristic):**
- **Likely deleterious**: Wild-type logit - mutant logit > 5
- **Possibly deleterious**: Difference > 2
- **Neutral**: Difference between -2 and 2
- **Possibly beneficial**: Difference < -2

**For sequence similarity (cosine similarity of embeddings):**
- **Highly similar**: > 0.9 (same protein family)
- **Related**: 0.7 - 0.9 (similar function)
- **Distant**: 0.5 - 0.7 (remote homology)
- **Unrelated**: < 0.5

**For scoring (pseudo-perplexity):**
- Lower perplexity indicates a more "natural" sequence according to the model
- Compare perplexities across variants rather than interpreting absolute values
- Perplexity is sensitive to sequence length; compare sequences of similar length

## Quick Start Examples

**Example 1: Extract embeddings for clustering**
```python
from proto_tools.tools.masked_models.esm2 import (
    ESM2EmbeddingsInput, ESM2EmbeddingsConfig, run_esm2_embeddings,
)
import numpy as np
from sklearn.cluster import KMeans

# Protein sequences
sequences = [
    "MVLSPADKTNVKAAW",
    "MVLSGEDKSNIKAAW",
    "GSSGSSGSSGSSGSS",
]

inputs = ESM2EmbeddingsInput(sequences=sequences)
config = ESM2EmbeddingsConfig(
    model_checkpoint="esm2_t33_650M_UR50D",
    batch_size=3,
    verbose=True
)

result = run_esm2_embeddings(inputs, config)

# Cluster embeddings
embeddings = np.array([r.mean_embedding for r in result.results])
kmeans = KMeans(n_clusters=2).fit(embeddings)
print(f"Cluster assignments: {kmeans.labels_}")
```

**Example 2: Score sequences (pseudo-perplexity)**
```python
from proto_tools.tools.masked_models.esm2 import ESM2ScoringInput, ESM2ScoringConfig, run_esm2_score

inputs = ESM2ScoringInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESM2ScoringConfig(batch_size=32, verbose=False)

result = run_esm2_score(inputs, config)
print(f"Perplexity: {result.scores[0].metrics['perplexity']:.3f}")
```

**Example 3: Guided sequence mutation**
```python
from proto_tools.tools.masked_models.esm2 import ESM2SampleInput, ESM2SampleConfig, run_esm2_sample

# Starting sequence
inputs = ESM2SampleInput(sequences=["MVLSPADKTNVKAAW"])

config = ESM2SampleConfig(
    model_checkpoint="esm2_t33_650M_UR50D",
    temperature=0.7,  # Conservative mutations
    decoding_method="entropy",  # Mutate uncertain positions
    num_mutations=3,  # 3 mutations per round
    verbose=True
)

result = run_esm2_sample(inputs, config)
print(f"Original: {inputs.sequences[0]}")
print(f"Mutated:  {result.sequences[0]}")
```

**Example 4: Batch processing**
```python
from proto_tools.tools.masked_models.esm2 import (
    ESM2EmbeddingsInput, ESM2EmbeddingsConfig, run_esm2_embeddings,
)

# Large batch of sequences
sequences = ["MVLSPADKTNVKAAW"] * 100

inputs = ESM2EmbeddingsInput(sequences=sequences)
config = ESM2EmbeddingsConfig(
    batch_size=16,  # Process 16 at a time
    verbose=True
)

result = run_esm2_embeddings(inputs, config)
print(f"Processed {len(result.results)} sequences")
```

**Example 5: Masked-PLL gradient over a relaxed sequence**
```python
from proto_tools.tools.masked_models.esm2 import (
    ESM2GradientInput, ESM2GradientConfig, run_esm2_gradient,
)
from proto_tools.utils import one_hot_protein_logits

# Seed the relaxed distribution from a discrete sequence (sharpness=2.0
# yields a biased-but-not-saturated softmax target).
logits = one_hot_protein_logits("MVLSPADKTNVKAAW", sharpness=2.0)

inputs = ESM2GradientInput(logits=logits, temperature=0.6)
config = ESM2GradientConfig(model_checkpoint="esm2_t33_650M_UR50D")

result = run_esm2_gradient(inputs, config)
print(f"Mean NLL:    {result.loss:.3f}")
print(f"Perplexity:  {result.metrics['perplexity']:.3f}")
print(f"Grad shape:  {len(result.gradient)} x {len(result.gradient[0])}")
# Step the relaxed sequence: logits ← logits − lr · gradient
```

**Example 6: Forward-only PLL scoring (no backward pass)**
```python
from proto_tools.tools.masked_models.esm2 import (
    ESM2GradientInput, ESM2GradientConfig, run_esm2_gradient,
)
from proto_tools.utils import one_hot_protein_logits

inputs = ESM2GradientInput(
    logits=one_hot_protein_logits("MVLSPADKTNVKAAW", sharpness=2.0),
    temperature=0.6,
)
config = ESM2GradientConfig(compute_gradient=False)  # skip backward pass

result = run_esm2_gradient(inputs, config)
assert result.gradient is None  # forward-only mode
print(f"avg log-likelihood: {result.metrics['avg_log_likelihood']:.3f}")
```

## Best Practices & Gotchas

**Batch processing:**

1. **Variable lengths**: Sequences are padded to batch max length - group similar-length sequences.

2. **OOM errors**: Reduce `batch_size` or use smaller model.

3. **Attention masks**: Always use masks to ignore padding positions in downstream analysis.

**Embedding usage:**

1. **Mean pooling**: `mean_embeddings` averages across sequence positions (ignoring padding).

2. **Per-position**: Extract from hidden states for residue-level tasks.

3. **Normalization**: Consider L2 normalizing embeddings for cosine similarity.

**Gradient tool:**

1. **Vocab order**: Input logits and the returned gradient share the canonical protein order `ACDEFGHIKLMNPQRSTVWY`. The tool maps to ESM2's tokenizer vocabulary internally.

2. **`temperature=None` vs. `1.0`**: With `temperature=None` (default), input logits are used as the soft distribution as-is — pass an already-normalized distribution. With `temperature` set, `softmax(logits / temperature)` is applied before the forward pass.

3. **STE vs. soft embeddings**: Set `use_ste=True` for stronger guidance toward discrete sequences (the forward sees hard one-hot tokens; gradients still flow through the soft probabilities). Leave `use_ste=False` for smooth optimization over the relaxed simplex.

4. **Forward-only mode**: Set `compute_gradient=False` to skip the backward pass entirely; `gradient` will be `None` but `loss` and `metrics` are populated. Useful for ranking MCMC proposals without paying the backward cost.

5. **Memory**: Gradient memory is dominated by the chunk size, not the sequence length — increase `batch_size` for shorter sequences, decrease on long inputs or smaller GPUs.

**Common mistakes:**

1. **Ignoring padding**: Always apply attention masks before averaging or analyzing logits.

2. **Empty sequences**: ESM2 tools require at least one non-empty sequence.

3. **Wrong vocab indices**: Tool logits are AA-only in order `ACDEFGHIKLMNPQRSTVWY` (no special tokens).

4. **Mixing checkpoints**: Embeddings from different model sizes are NOT comparable.

## References

**Primary publication:**
- Lin, Z. et al. (2023). "Evolutionary-scale prediction of atomic level protein structure with a language model." *Science*, 379(6637), 1123-1130. DOI: [10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

**Implementation:**
- GitHub: [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)
- Model hub: [https://huggingface.co/facebook](https://huggingface.co/facebook)

## Related Tools

**Tools often used together:**
- `esm3`: Newer generative model with structure prediction
- `inverse_folding/proteinmpnn`: Structure-conditioned sequence design
- `mmseqs2-clustering`: Cluster sequences before/after embedding analysis

**Alternative tools:**
- `progen2`: Autoregressive protein generation
- `esm3`: Can also do embeddings, but ESM2 is faster for embedding-only tasks
