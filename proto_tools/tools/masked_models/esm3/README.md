<a href="https://bio-pro.mintlify.app/tools/masked-models/esm3"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESM3

## Overview
ESM3 is EvolutionaryScale's next-generation [protein language model](https://www.evolutionaryscale.ai/blog/esm-cambrian) with sequence and structure modeling capabilities. This package's `esm3-sample` tool exposes masked sequence editing over supplied protein sequences. The open model (`esm3_sm_open_v1`) provides embeddings, logits, masked sampling, and scoring in a unified framework.

## Background

**What are protein language models?**
Protein language models (pLMs) learn the "grammar" of proteins from evolutionary data. ESM3 extends this by jointly modeling sequence and structure, capturing:
- **Sequence conservation**: Which residues are essential for function
- **[Co-evolution](https://en.wikipedia.org/wiki/Coevolution)**: Pairs of residues that evolve together (often in contact)
- **Structural constraints**: Patterns that define [secondary/tertiary structure](https://en.wikipedia.org/wiki/Protein_structure)
- **3D geometry**: Spatial relationships between residues

**Why ESM3 over ESM2?**
ESM3 is broader than ESM2 at the model-family level. In Proto Tools today, use `esm3-sample` for masked sequence editing and local refinement; for pure sequence embedding tasks, ESM2 is often faster.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `esm3-embedding` | Extract embeddings and logits | Embeddings, logits, attention masks |
| `esm3-sample` | Mutate/restore masked sequence positions using model | Modified sequences, optional logits |
| `esm3-score` | Score sequences with MLM pseudo-perplexity | Per-sequence metrics, optional logits |

## Model Variants

| Checkpoint | Description | Default |
|------------|-------------|---------|
| `esm3_sm_open_v1` | Small open-source ESM3 model | Yes |

Currently only the small open-source model is available. Larger models may become available through EvolutionaryScale's API.

## Execution Modes

- **Local GPU/CPU**: Loads the model on-demand. Use `device="cuda"`, `"cpu"`, or `"mps"`.

## How It Works

ESM3 uses a transformer architecture that jointly models protein sequence and structure. Key differences from ESM2:
- **Generative model family**: Can support sequence/structure generation modes, though this package exposes masked sequence sampling for supplied sequences
- **Structure tokens**: Encodes 3D structure as discrete tokens alongside sequence tokens
- **Multi-track architecture**: Processes sequence, structure, and function information in parallel

**Embeddings**: Forward pass produces per-position hidden states; mean-pooling yields fixed-length descriptors.

**Sampling**: Positions are selected by masking strategy (entropy, max-logit, or random), masked, and resampled from the model's distribution.

**Scoring**: Each position is masked one at a time to compute log-probability of the true amino acid, yielding pseudo-perplexity.

## Input Parameters

All tools take a tool-specific input with one or more protein sequences:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | Protein sequences (amino acid strings) |

## Configuration

### Embeddings Tool (`ESM3EmbeddingsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `esm3_sm_open_v1` | Model variant |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
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
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `cuda` | Device |
| `verbose` | `bool` | `False` | Print progress |
| `return_logits` | `bool` | `False` | Include per-position logits |

### Scoring Tool (`ESM3ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `esm3_sm_open_v1` | Model variant |
| `batch_size` | `int` | `1` | Masked variants per forward pass |
| `device` | `str` | `cuda` | Device |
| `verbose` | `bool` | `False` | Print progress |
| `return_logits` | `bool` | `False` | Include per-position logits |

### Parameter Guides

**Temperature guide for sampling:**
| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.5-0.7 | Conservative | Safer mutations |
| 1.0 | Standard | Model distribution |
| 1.5-2.0 | Creative | More diverse mutations |

## Output Specification

### ESM3EmbeddingsOutput (Embeddings)

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[SequenceEmbedding]` | Per-sequence embedding results (primary field) |

**`SequenceEmbedding` fields:**
| Field | Type | Description |
|-------|------|-------------|
| `mean_embedding` | `List[float]` | Mean-pooled embedding vector for one sequence |
| `attention_mask` | `List[int]` | Binary mask: 1 = valid position, 0 = padding |
| `logits` | `Optional[List[List[float]]]` | Per-position logits (seq_len, vocab_size). Only present if `return_logits=True` |

### ESM3SampleOutput

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Mutated protein sequences |
| `logits` | `Optional[List[List[List[float]]]]` | Per-position logits (if requested) |

Supported export formats: `fasta`, `txt`, `json`

### MaskedModelScoringOutput (Scoring)

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[MaskedModelScoringMetrics]` | Per-sequence metrics and optional logits |

Each `MaskedModelScoringMetrics` entry includes:
- `log_likelihood`, `avg_log_likelihood`, `perplexity` — access via attribute (`score.perplexity`) or mapping (`score["perplexity"]`)
- `logits`: per-position logits if `return_logits=True`
- `vocab`: list of 20 standard amino acids if `return_logits=True`

## Interpreting Results

**Variant effect (logit difference):**
- Logit difference > 5: likely deleterious
- Logit difference > 2: possibly deleterious
- Logit difference between -2 and 2: neutral
- Logit difference < -2: possibly beneficial

**Scoring (pseudo-perplexity):**
- Lower perplexity indicates a more "natural" sequence
- Compare perplexities across variants rather than interpreting absolute values

## Quick Start Examples

**Example 1: Extract embeddings**
```python
from proto_tools.tools.masked_models.esm3 import ESM3EmbeddingsInput, ESM3EmbeddingsConfig, run_esm3_embeddings

inputs = ESM3EmbeddingsInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESM3EmbeddingsConfig(verbose=True)

result = run_esm3_embeddings(inputs, config)
print(f"Processed {len(result.results)} sequences")
print(f"Embedding dim: {len(result.results[0].mean_embedding)}")
```

**Example 2: Sequence mutation**
```python
from proto_tools.tools.masked_models.esm3 import ESM3SampleInput, ESM3SampleConfig, run_esm3_sample

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

**Example 3: Score sequences**
```python
from proto_tools.tools.masked_models.esm3 import ESM3ScoringInput, ESM3ScoringConfig, run_esm3_score

inputs = ESM3ScoringInput(sequences=["MVLSPADKTNVKAAW", "GSSGSSGSS"])
config = ESM3ScoringConfig(batch_size=32)

result = run_esm3_score(inputs, config)
print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
```

**Example 4: Batch processing**
```python
from proto_tools.tools.masked_models.esm3 import (
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
print(f"Processed {len(result.results)} sequences")
```

## Best Practices & Gotchas

**Embeddings and scoring:**
- Use `return_logits=False` unless you explicitly need logits (saves memory)

**Sampling:**
- Start with low `temperature` and small `num_mutations` for conservative edits
- Use `decoding_method="entropy"` for natural-looking mutations

**Common mistakes:**
1. **Expecting ESM2-speed embeddings**: ESM3 is generally slower than ESM2 for embedding-only tasks.

## References

**Primary publication:**
- Hayes, T. et al. (2024). "Simulating 500 million years of evolution with a language model." *Science*. DOI: [10.1126/science.adk8946](https://doi.org/10.1126/science.adk8946)

**Implementation:**
- GitHub: [https://github.com/evolutionaryscale/esm](https://github.com/evolutionaryscale/esm)
- EvolutionaryScale: [https://www.evolutionaryscale.ai/](https://www.evolutionaryscale.ai/)

## Related Tools

**Tools often used together:**
- `esm2`: Faster embeddings-only model (use when you don't need structure)
- `inverse_folding/proteinmpnn`: Structure-conditioned sequence design

**Alternative tools:**
- `esmfold`: Dedicated structure prediction (based on ESM2)
- `progen2`: Autoregressive protein generation
