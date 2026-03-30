<a href="https://bio-pro.mintlify.app/tools/masked-models/esm2"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESM2

## Overview
ESM2 (Evolutionary Scale Modeling 2) is Meta AI's [protein language model](https://www.evolutionaryscale.ai/blog/esm-cambrian) trained on millions of protein sequences from [UniRef](https://www.uniprot.org/help/uniref). It provides sequence embeddings, per-position amino acid logits, sequence mutation (sampling), and sequence scoring (MLM pseudo-perplexity). ESM2 offers multiple model sizes from 8M to 15B parameters, balancing quality and computational cost.

## When to Use This Tool

**Primary use cases:**
- Extracting protein sequence embeddings for similarity search, clustering, or ML features
- Zero-shot variant effect prediction using per-position logits
- Protein sequence mutation and guided design
- Sequence quality scoring (pseudo-perplexity)
- Transfer learning for protein property prediction

**When NOT to use this tool:**
- For DNA/RNA sequences: Use DNA-specific models (Evo2, Enformer)
- For structure prediction: Use ESMFold, Boltz2, or AlphaFold3
- For inverse folding: Use ProteinMPNN (structure-conditioned)
- For very long proteins (>2000 residues): Consider chunking or using ESM3

## Biological Background

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

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `esm2-embedding` | Extract embeddings and logits | Embeddings, logits, attention masks |
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

## Configuration

### Embeddings Tool (`ESM2EmbeddingsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |
| `verbose` | `bool` | `False` | Print progress messages |
| `return_logits` | `bool` | `False` | Include per-position logits in output |

### Sampling Tool (`ESM2SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `temperature` | `float` | 1.0 | Sampling temperature (lower = conservative) |
| `decoding_method` | `str` | `"entropy"` | Position selection: `"entropy"`, `"max_logit"`, `"random"` |
| `num_mutations` | `int` | 1 | Positions to mutate per iteration |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `"cuda"` | Device for inference |
| `return_logits` | `bool` | `False` | Include per-position logits in output |

### Scoring Tool (`ESM2ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `"esm2_t33_650M_UR50D"` | Model variant to use |
| `batch_size` | `int` | `1` | Masked variants per forward pass |
| `device` | `str` | `"cuda"` | Device for inference |
| `verbose` | `bool` | `False` | Print progress messages |
| `return_logits` | `bool` | `False` | Include per-position logits in output |

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
| `scores` | `List[SequenceScores]` | One score object per input sequence |

**SequenceScores fields:**
| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `Dict[str, float]` | Includes `log_likelihood`, `avg_log_likelihood`, `perplexity` |
| `logits` | `List[List[float]]?` | Optional per-position logits (shape `[seq_len, 20]`) |
| `vocab` | `List[str]?` | Amino acid order (AA-only) |

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
from bio_programming_tools.tools.masked_models.esm2 import (
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
from bio_programming_tools.tools.masked_models.esm2 import ESM2ScoringInput, ESM2ScoringConfig, run_esm2_score

inputs = ESM2ScoringInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESM2ScoringConfig(batch_size=32, verbose=False)

result = run_esm2_score(inputs, config)
print(f"Perplexity: {result.scores[0].metrics['perplexity']:.3f}")
```

**Example 3: Guided sequence mutation**
```python
from bio_programming_tools.tools.masked_models.esm2 import ESM2SampleInput, ESM2SampleConfig, run_esm2_sample

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
from bio_programming_tools.tools.masked_models.esm2 import (
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

## Best Practices & Gotchas

**Batch processing:**

1. **Variable lengths**: Sequences are padded to batch max length - group similar-length sequences.

2. **OOM errors**: Reduce `batch_size` or use smaller model.

3. **Attention masks**: Always use masks to ignore padding positions in downstream analysis.

**Embedding usage:**

1. **Mean pooling**: `mean_embeddings` averages across sequence positions (ignoring padding).

2. **Per-position**: Extract from hidden states for residue-level tasks.

3. **Normalization**: Consider L2 normalizing embeddings for cosine similarity.

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
- `mmseqs-clustering`: Cluster sequences before/after embedding analysis

**Alternative tools:**
- `progen2`: Autoregressive protein generation
- `esm3`: Can also do embeddings, but ESM2 is faster for embedding-only tasks
