# ProGen2

## Overview
ProGen2 is Salesforce's autoregressive protein language model for de novo protein sequence generation and scoring. Unlike masked language models (ESM2/ESM3), ProGen2 generates proteins left-to-right from a prompt and provides autoregressive likelihood scoring. The tool supports local GPU execution via a standalone venv and optional the cloud runtime GPU execution.

## When to Use This Tool

**Primary use cases:**
- De novo protein sequence generation
- Extending protein sequences from N-terminal prompts
- Generating antibody sequences (using `progen2-oas`)
- Scoring candidate protein sequences by autoregressive likelihood

**When NOT to use this tool:**
- For DNA sequences: Use Evo2
- For embeddings/variant scoring: Use ESM2/ESM3
- For structure-conditioned design: Use ProteinMPNN
- For structure prediction: Use ESMFold/Boltz2/Chai1

## Model Variants

**Available checkpoints (see `standalone/inference.py`):**
- `progen2-small` (151M parameters)
- `progen2-medium` (754M parameters)
- `progen2-oas` (754M parameters, antibody-specific)
- `progen2-large` (2B parameters, default)
- `progen2-BFD90` (2B parameters, trained on BFD90)
- `progen2-xlarge` (6B parameters)

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `progen2-sample` | Autoregressive protein generation | Generated sequences, optional logits |
| `progen2-score` | Autoregressive sequence scoring | Per-sequence metrics, optional logits |

## Package Layout

Modules in `bio_programming_tools.tools.causal_models.progen2`:
- `progen2_sample.py`: sampling/generation tool
- `progen2_score.py`: scoring tool
- `standalone/inference.py`: model wrapper and low-level inference
- `examples/example.ipynb`: usage walkthrough

## Execution Modes

- **Local execution** runs ProGen2 in an isolated venv via `EnvManager` (see `standalone/setup.sh`).
- **the cloud runtime execution** is used automatically when configured.

## Environment Setup

Local execution uses a dedicated venv defined by:
- `standalone/requirements.txt`
- `standalone/setup.sh`

If you need to refresh the environment, re-run the setup script.

## Inputs

### Sampling (`ProGen2SampleInput`)
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompts` | `str` or `List[str]` | Prompt protein sequence(s) for generation |

### Scoring (`ProGen2ScoringInput`)
| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `str` or `List[str]` | Protein sequences to score |

## Configurations

### Sampling (`ProGen2SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `progen2-large` | Model checkpoint to use |
| `local_path` | `Optional[str]` | `None` | Local weights path (if not using HF) |
| `max_length` | `int` | `256` | Max total length (prompt + generated) |
| `temperature` | `float` | `0.2` | Sampling temperature |
| `top_p` | `float` | `0.95` | Nucleus sampling threshold |
| `top_k` | `int` | `0` | Top-k sampling limit (0 disables) |
| `truncate_at_stop` | `bool` | `True` | Truncate at stop tokens (`1` or `2`) |
| `strip_special_tokens` | `bool` | `True` | Remove `1`/`2` tokens from output |
| `prepend_prompt` | `bool` | `True` | Include prompt in output |
| `batch_size` | `Optional[int]` | `None` | Prompts per batch (all if None) |
| `verbose` | `bool` | `False` | Verbose logging |
| `return_logits` | `bool` | `False` | Include per-token logits |

### Scoring (`ProGen2ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `progen2-large` | Model checkpoint to use |
| `local_path` | `Optional[str]` | `None` | Local weights path (if not using HF) |
| `batch_size` | `Optional[int]` | `None` | Sequences per batch |
| `device` | `str` | `cuda` | Device to run on |
| `verbose` | `bool` | `False` | Verbose logging |
| `return_logits` | `bool` | `False` | Include per-position logits |

## Outputs

### `ProGen2SampleOutput`

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Generated protein sequences |
| `logits` | `Optional[List[List[List[float]]]]` | Per-position logits (if requested) |

**Supported export formats:** `fasta`, `txt`, `json`

### `ProGen2ScoringOutput`

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[SequenceScores]` | Per-sequence metrics and optional logits |

Each `SequenceScores` entry includes:
- `metrics`: `log_likelihood`, `avg_log_likelihood`, `perplexity`
- `logits`: per-position logits if `return_logits=True`
- `vocab`: 30-token ProGen2 vocabulary if `return_logits=True`

## Best Practices and Gotchas

**Prompt format:**
- ProGen2 uses `1` as the start token and `2` as the end token.
- Raw amino acid prompts (e.g., `MKTL`) are automatically prepended with `1`.
- Use explicit `1` if you want full control over special tokens.

**Generation quality:**
- Lower temperatures (0.2–0.5) are usually best for proteins.
- `max_length` includes the prompt; set it accordingly for long proteins.
- `truncate_at_stop=True` may produce shorter sequences than `max_length`.

**Model selection:**
- Quick tests: `progen2-small`
- Production: `progen2-large` or `progen2-xlarge`
- Antibodies: `progen2-oas`

## Quick Start Examples

**Example 1: Basic protein generation**
```python
from bio_programming_tools.tools.causal_models.progen2 import (
    run_progen2_sample, ProGen2SampleInput, ProGen2SampleConfig
)

inputs = ProGen2SampleInput(prompts=["MKTAYIAKQRQISF"])
config = ProGen2SampleConfig(
    model_checkpoint="progen2-large",
    max_length=120,
    temperature=0.2,
)

result = run_progen2_sample(inputs, config)
print(f"Generated: {result.sequences[0]}")
```

**Example 2: Score sequences**
```python
from bio_programming_tools.tools.causal_models.progen2 import (
    run_progen2_score, ProGen2ScoringInput, ProGen2ScoringConfig
)

inputs = ProGen2ScoringInput(sequences=["MVLSPADKTN", "MKTLLILAVVAA"])
config = ProGen2ScoringConfig(batch_size=2)

result = run_progen2_score(inputs, config)
print(f"Perplexity: {result.scores[0].metrics['perplexity']:.3f}")
```

## References

- Nijkamp et al. (2023). "ProGen2: Exploring the boundaries of protein language models". Cell Systems. DOI: 10.1016/j.cels.2023.10.002
- Hugging Face: https://huggingface.co/hugohrban/
- GitHub (fine-tuning): https://github.com/hugohrban/ProGen2-finetuning
- Original GitHub: https://github.com/enijkamp/progen2
