# ESM3

## Overview
ESM3 is EvolutionaryScale's next-generation protein language model that combines sequence understanding with structure prediction and generation capabilities. Unlike ESM2, ESM3 is a generative model that can perform both embedding extraction and de novo protein structure prediction. The open model (`esm3_sm_open_v1`) provides embeddings, logits, sampling, and structure prediction in a unified framework.

## When to Use This Tool

**Primary use cases:**
- Extract protein sequence embeddings and per-position logits
- Score protein sequences via MLM pseudo-perplexity
- Mutate or generate protein sequences
- Predict 3D structure from sequence alone

**When NOT to use this tool:**
- For DNA/RNA sequences: use Evo2 or genomic models
- For highest-accuracy structures: use Boltz2, Chai1, or AlphaFold3
- For structure-conditioned design: use ProteinMPNN
- For fast embeddings-only tasks at scale: ESM2 is typically faster

## Model Variants

**Available checkpoints:**
- `esm3_sm_open_v1`: Small open-source ESM3 model (default)

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `esm3-embedding` | Extract embeddings and logits | Embeddings, logits, attention masks |
| `esm3-sample` | Mutate sequences using model | Modified sequences, optional logits |
| `esm3-structure-prediction` | Predict 3D structure | PDB strings, pLDDT, pTM |
| `esm3-score` | Score sequences with MLM pseudo-perplexity | Per-sequence metrics, optional logits |

## Package Layout

These are the primary modules in `bio_programming.bio_tools.tools.masked_models.esm3`:
- `esm3_embeddings.py`: embeddings + logits tool
- `esm3_structure_prediction.py`: structure prediction tool
- `esm3_sample.py`: sampling/mutation tool
- `esm3_score.py`: scoring tool
- `standalone/inference.py`: model wrapper and low-level inference

## Execution Modes

ESM3 can run locally or on the cloud runtime:
- Local GPU/CPU loads the model on-demand
- the cloud runtime GPU execution is used automatically when configured in the environment

## Inputs

All tools take a tool-specific input (e.g., `ESM3EmbeddingsInput`, `ESM3ScoringInput`) with one or more protein sequences:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences (amino acid strings) |

## Configurations

### Embeddings Tool (`ESM3EmbeddingsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `esm3_sm_open_v1` | Model variant |
| `batch_size` | `int` | `128` | Sequences per batch |
| `device` | `str` | `cuda` | `cuda`, `cpu`, or `mps` |
| `verbose` | `bool` | `False` | Print progress |
| `return_logits` | `bool` | `False` | Include per-position logits |

### Sampling Tool (`ESM3SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `esm3_sm_open_v1` | Model variant |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `decoding_method` | `str` | `entropy` | Position selection method |
| `num_mutations` | `int` | `1` | Mutations per iteration |
| `batch_size` | `Optional[int]` | `None` | Sequences per batch |
| `device` | `str` | `cuda` | Device |
| `verbose` | `bool` | `False` | Print progress |
| `return_logits` | `bool` | `False` | Include per-position logits |

### Structure Prediction Tool (`ESM3StructurePredictionConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `esm3_sm_open_v1` | Model variant |
| `batch_size` | `int` | `128` | Sequences per batch (use smaller for structures) |
| `device` | `str` | `cuda` | Device |
| `verbose` | `bool` | `False` | Print progress |

### Scoring Tool (`ESM3ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `esm3_sm_open_v1` | Model variant |
| `batch_size` | `int` | `32` | Masked variants per forward pass |
| `device` | `str` | `cuda` | Device |
| `verbose` | `bool` | `False` | Print progress |
| `return_logits` | `bool` | `False` | Include per-position logits |

## Outputs

### `ESM3EmbeddingsOutput` (Embeddings)

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `mean_embeddings` | `List[List[float]]` | `[num_seq, embed_dim]` | Mean-pooled sequence embeddings |
| `logits` | `List[List[List[float]]]` | `[num_seq, seq_len, vocab_size]` | Per-position logits |
| `attention_masks` | `List[List[int]]` | `[num_seq, seq_len]` | Valid position masks |
| `num_sequences` | `int` | - | Sequences processed |

### `ESM3SampleOutput`

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Mutated protein sequences |
| `logits` | `Optional[List[List[List[float]]]]` | Per-position logits (if requested) |

**Supported export formats:** `fasta`, `txt`, `json`

### `ESM3StructurePredictionOutput`

| Field | Type | Description |
|-------|------|-------------|
| `structures` | `List[Dict]` | Structures with `pdb_string`, `avg_plddt`, `ptm` |
| `num_sequences` | `int` | Sequences processed |

**Structure dict fields:**
| Key | Type | Description |
|-----|------|-------------|
| `sequence` | `str` | Input sequence |
| `pdb_string` | `str` | PDB format structure |
| `avg_plddt` | `float` | Average pLDDT (0-100) |
| `ptm` | `float` | Predicted TM-score (0-1) |

**Supported export formats:** `pdb`, `json`

### `MaskedModelScoringOutput` (Scoring)

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[SequenceScores]` | Per-sequence metrics and optional logits |

Each `SequenceScores` entry includes:
- `metrics`: `log_likelihood`, `avg_log_likelihood`, `perplexity`
- `logits`: per-position logits if `return_logits=True`
- `vocab`: list of 20 standard amino acids if `return_logits=True`

## Thresholds and Decision Boundaries

**Structure prediction confidence:**
| pLDDT | Interpretation |
|-------|----------------|
| `> 90` | Very high confidence |
| `70-90` | High confidence |
| `50-70` | Low confidence |
| `< 50` | Very low confidence (likely disordered) |

**pTM score:**
| pTM | Interpretation |
|-----|----------------|
| `> 0.8` | High confidence in overall fold |
| `0.5-0.8` | Moderate confidence |
| `< 0.5` | Low confidence |

**Variant effect (same as ESM2):**
- Logit difference `> 5`: likely deleterious
- Logit difference `> 2`: possibly deleterious

## Best Practices and Gotchas

**Structure prediction:**
- Use small `batch_size` values (1-4) to avoid OOM errors
- Very long sequences (>1000 residues) may fail or be low confidence
- Always check pLDDT and pTM before trusting the fold

**Embeddings and scoring:**
- Use `return_logits=False` unless you explicitly need logits

**Sampling:**
- Start with low `temperature` and small `num_mutations` for conservative edits
- Use `decoding_method="entropy"` for natural-looking mutations

## Quick Start Examples

**Example 1: Extract embeddings**
```python
from bio_programming.bio_tools.tools.masked_models.esm3 import ESM3EmbeddingsInput, ESM3EmbeddingsConfig, run_esm3_embeddings

inputs = ESM3EmbeddingsInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESM3EmbeddingsConfig(verbose=True)

result = run_esm3_embeddings(inputs, config)
print(f"Processed {result.num_sequences} sequences")
print(f"Embedding dim: {len(result.mean_embeddings[0])}")
```

**Example 2: Predict structure**
```python
from bio_programming.bio_tools.tools.masked_models.esm3 import (
    ESM3StructurePredictionInput, ESM3StructurePredictionConfig, run_esm3_structure_prediction,
)

inputs = ESM3StructurePredictionInput(sequences=["MVLSPADKTNVKAAWGKIGSHAGEYGAEALERMFLGFPTTKTYFPHFDLSH"])
config = ESM3StructurePredictionConfig(
    batch_size=1,  # Small for structure prediction
    verbose=True
)

result = run_esm3_structure_prediction(inputs, config)

# Check confidence
structure = result.structures[0]
print(f"Average pLDDT: {structure['avg_plddt']:.1f}")
print(f"pTM score: {structure['ptm']:.2f}")

# Save structure
with open("predicted.pdb", "w") as f:
    f.write(structure['pdb_string'])
```

**Example 3: Sequence mutation**
```python
from bio_programming.bio_tools.tools.masked_models.esm3 import ESM3SampleInput, ESM3SampleConfig, run_esm3_sample

inputs = ESM3SampleInput(sequences=["MVLSPADKTNVKAAW"])

config = ESM3SampleConfig(
    temperature=0.7,
    decoding_method="entropy",
    num_mutations=3
)

result = run_esm3_sample(inputs, config)
print(f"Original: {inputs.sequences[0]}")
print(f"Mutated:  {result.sequences[0]}")
```

**Example 4: Score sequences**
```python
from bio_programming.bio_tools.tools.masked_models.esm3 import ESM3ScoringInput, ESM3ScoringConfig, run_esm3_score

inputs = ESM3ScoringInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESM3ScoringConfig(batch_size=32)

result = run_esm3_score(inputs, config)
print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
```

**Example 5: Batch processing**
```python
from bio_programming.bio_tools.tools.masked_models.esm3 import (
    ESM3EmbeddingsInput, ESM3EmbeddingsConfig, run_esm3_embeddings,
)

# Large batch of sequences
sequences = ["MVLSPADKTNVKAAW"] * 100

inputs = ESM3EmbeddingsInput(sequences=sequences)
config = ESM3EmbeddingsConfig(
    batch_size=16,  # Process 16 at a time
    verbose=True
)

result = run_esm3_embeddings(inputs, config)
print(f"Processed {result.num_sequences} sequences")
```

## References

**Resources:**
- GitHub: https://github.com/evolutionaryscale/esm
- EvolutionaryScale: https://www.evolutionaryscale.ai/

## Related Tools

- `esm2`: Faster embeddings-only model
- `esmfold`: Dedicated structure prediction (based on ESM2)
- `progen2`: Autoregressive protein generation
- `inverse_folding/proteinmpnn`: Structure-conditioned sequence design
