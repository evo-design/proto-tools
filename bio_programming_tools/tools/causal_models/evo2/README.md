# Evo2

## Overview
Evo2 is Arc Institute's DNA language model for genomic sequence generation and scoring. It performs autoregressive generation of DNA from prompts and can score sequences by log-likelihood. The tool supports local GPU/CPU execution and optional the cloud runtime GPU execution, with KV caching for efficient long generations in local mode.

## When to Use This Tool

**Primary use cases:**
- De novo DNA sequence generation from prompts
- Extending partial DNA sequences
- Generating coding sequences, promoters, and regulatory elements
- Scoring candidate DNA sequences with autoregressive likelihood

**When NOT to use this tool:**
- For protein sequences: use ESM2/ESM3 or ProGen2
- For transcriptional activity prediction: use Enformer or Borzoi
- For short oligo design: simpler tools may suffice

## Model Variants

**Available checkpoints (see `inference.py`):**
- `evo2_7b` (default)
- `evo2_40b`
- `evo2_7b_base`
- `evo2_40b_base`
- `evo2_1b_base`
- `evo2_7b_262k`
- `evo2_7b_microviridae`

Availability depends on installed weights and/or your Hugging Face access.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `evo2-sample` | Autoregressive DNA generation | Generated sequences, optional logits, optional KV caches |
| `evo2-score` | Autoregressive sequence scoring | Per-sequence metrics, optional logits |

## Package Layout

Modules in `bio_programming_tools.tools.causal_models.evo2`:
- `evo2_sample.py`: sampling/generation tool
- `evo2_score.py`: scoring tool
- `standalone/inference.py`: model wrapper and low-level inference

## Execution Modes

- **Local execution** loads the model on-demand and supports KV caching.
- **the cloud runtime execution** is used automatically when configured, but does **not** support KV cache reuse.

## Environment Setup References

Evo2/Vortex setup is complex. See:
- `https://github.com/ArcInstitute/evo2`
- `https://github.com/Zymrael/vortex`

## Inputs

### Sampling (`Evo2SampleInput`)
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompts` | `str` or `List[str]` | Prompt DNA sequence(s) for generation |

### Scoring (`Evo2ScoringInput`)
| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `str` or `List[str]` | DNA sequences to score |

## Configurations

### Sampling (`Evo2SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `evo2_7b` | Model checkpoint to use |
| `local_path` | `Optional[str]` | `None` | Local weights path (if not using HF) |
| `num_tokens` | `int` | `32` | Number of new tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `top_k` | `int` | `4` | Top-k sampling limit |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `prepend_prompt` | `bool` | `True` | Include prompt in output |
| `cached_generation` | `bool` | `True` | Use KV caching (local only) |
| `force_prompt_threshold` | `Optional[int]` | `None` | Prefill tokens before prompt forcing |
| `max_seqlen` | `Optional[int]` | `None` | Max sequence length for KV cache |
| `stop_at_eos` | `bool` | `True` | Stop at EOS token |
| `old_kv_cache` | `Optional[Dict]` | `None` | Continue generation from KV cache (local only) |
| `batch_size` | `Optional[int]` | `None` | Prompts per batch (all if None) |
| `device` | `str` | `cuda` | Device to run on |
| `keep_on_gpu` | `bool` | `True` | Keep model loaded after call |
| `print_generation` | `bool` | `True` | Print generation tokens (debug) |
| `verbose` | `bool` | `False` | Verbose logging |
| `return_logits` | `bool` | `False` | Include per-token logits |

### Scoring (`Evo2ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `evo2_7b` | Model checkpoint to use |
| `local_path` | `Optional[str]` | `None` | Local weights path (if not using HF) |
| `batch_size` | `Optional[int]` | `None` | Sequences per batch |
| `device` | `str` | `cuda` | Device to run on |
| `keep_on_gpu` | `bool` | `False` | Keep model loaded after call |
| `verbose` | `bool` | `False` | Verbose logging |
| `return_logits` | `bool` | `False` | Include per-position logits |

## Outputs

### `Evo2SampleOutput`

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Generated DNA sequences |
| `logits` | `Optional[List]` | Per-token logits (if requested) |
| `kv_caches` | `Optional[List[Dict]]` | KV caches for continued generation (local only) |

**Supported export formats:** `fasta`, `txt`, `json`

### `Evo2ScoringOutput`

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[SequenceScores]` | Per-sequence metrics and optional logits |

Each `SequenceScores` entry includes:
- `metrics`: `log_likelihood`, `avg_log_likelihood`, `perplexity`
- `logits`: per-position logits if `return_logits=True`
- `vocab`: byte-level vocabulary if `return_logits=True`

## Best Practices and Gotchas

**Prompt design:**
- Longer, biologically meaningful prompts provide better context.
- For batching, prompts should have the same length for maximum throughput.

**Generation parameters:**
- `temperature` 0.8–1.0 is a good default; lower for conservative sequences.
- `top_k=4` and `top_p` in 0.9–1.0 are reasonable starting points.

**Performance and memory:**
- Use `cached_generation=True` for long sequences (local only).
- Reduce `batch_size` if you hit OOM errors.

**Tokenization:**
- Evo2 uses a byte-level vocabulary (size 512). DNA bases map to ASCII values.

## Quick Start Examples

**Example 1: Basic DNA generation**
```python
from bio_programming_tools.tools.causal_models.evo2 import (
    run_evo2_sample, Evo2SampleInput, Evo2SampleConfig
)

inputs = Evo2SampleInput(prompts=["ATGCGTAAA"])
config = Evo2SampleConfig(
    num_tokens=500,
    temperature=0.8,
    verbose=True
)

result = run_evo2_sample(inputs, config)
print(f"Generated: {result.sequences[0][:100]}...")
print(f"Total length: {len(result.sequences[0])}")
```

**Example 2: Batched generation**
```python
from bio_programming_tools.tools.causal_models.evo2 import (
    run_evo2_sample, Evo2SampleInput, Evo2SampleConfig
)

inputs = Evo2SampleInput(prompts=[
    "ATGAAACGT",
    "ATGCCCGTT",
    "ATGGGGAAA"
])

config = Evo2SampleConfig(
    num_tokens=1000,
    temperature=1.0,
    batch_size=3
)

result = run_evo2_sample(inputs, config)
for i, seq in enumerate(result.sequences):
    print(f"Sequence {i+1}: {len(seq)} bp")
```

**Example 3: Continue generation with KV cache (local only)**
```python
from bio_programming_tools.tools.causal_models.evo2 import (
    run_evo2_sample, Evo2SampleInput, Evo2SampleConfig
)

inputs = Evo2SampleInput(prompts=["ATGCGTAAA"])
config = Evo2SampleConfig(num_tokens=200, cached_generation=True)

result = run_evo2_sample(inputs, config)

# Continue generation from cached state
next_config = Evo2SampleConfig(
    num_tokens=200,
    cached_generation=True,
    old_kv_cache=result.kv_caches[0]
)
next_inputs = Evo2SampleInput(prompts=[result.sequences[0]])
next_result = run_evo2_sample(next_inputs, next_config)
```

**Example 4: Score sequences**
```python
from bio_programming_tools.tools.causal_models.evo2 import (
    run_evo2_score, Evo2ScoringInput, Evo2ScoringConfig
)

inputs = Evo2ScoringInput(sequences=["ATCGATCG", "GCTAGCTA"])
config = Evo2ScoringConfig(batch_size=2)

result = run_evo2_score(inputs, config)
print(f"Perplexity: {result.scores[0].metrics['perplexity']}")
```

## References

**Resources:**
- GitHub: https://github.com/arcinstitute/evo2
- Arc Institute: https://arcinstitute.org/tools/evo
- Original Evo paper: https://www.science.org/doi/10.1126/science.ado9336

## Related Tools

- enformer/borzoi: score generated sequences for regulatory activity
- prodigal: find ORFs in generated sequences
- esm2/esm3: for protein sequence tasks
