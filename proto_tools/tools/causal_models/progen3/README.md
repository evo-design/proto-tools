<a href="https://bio-pro.mintlify.app/tools/causal-models/progen3"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ProGen3

## Overview

ProGen3 is a Mixture-of-Experts (MoE) protein language model from Profluent AI, based on the Mixtral/Mistral architecture. It supports autoregressive protein sequence generation in both forward (N→C) and reverse (C→N) directions, as well as bidirectional sequence scoring that averages forward and reverse log-likelihoods for more robust evaluation.

ProGen3 is available in six sizes from 112M to 3B parameters, trained on large-scale protein sequence databases. It uses Flash Attention 2 and MegaBlocks for efficient sparse MoE computation.

## When to Use This Tool

**Use ProGen3 for:**
- Generating novel protein sequences from N-terminal or C-terminal prompts
- Scoring protein sequences for naturalness/fitness (bidirectional likelihood)
- Comparing protein variants by perplexity
- Unconditional protein sequence generation

**Consider alternatives when:**
- You need DNA/genomic sequence generation → use Evo1/Evo2
- You need antibody-specific generation → use ProGen2 with `progen2-oas` checkpoint
- You need masked/infilling protein generation → use ESM2 or ESM3
- You need structure-conditioned sequence design → use ProteinMPNN or LigandMPNN

## Biological Background

Protein language models learn the statistical patterns of natural protein sequences, capturing evolutionary constraints on amino acid usage. ProGen3's bidirectional scoring (averaging N→C and C→N likelihoods) provides a more robust assessment of sequence naturalness than unidirectional models, since protein function depends on the complete 3D structure rather than just the N-to-C reading frame.

## Tool Catalog

| Key | Operation | Input | Output | Use Case |
|-----|-----------|-------|--------|----------|
| `progen3-sample` | Generate | Prompt sequences | Generated proteins | Design novel proteins from N/C-terminal seeds |
| `progen3-score` | Score | Protein sequences | Perplexity metrics | Evaluate sequence naturalness/fitness |

## How It Works

### Sampling (progen3-sample)

Generates protein sequences autoregressively from a prompt. The `direction` config controls generation direction:
- `direction="forward"` (default): N→C generation, extending from the N-terminus
- `direction="reverse"`: C→N generation, extending from the C-terminus
- Empty string prompt (`""`) for unconditional generation in either direction

### Scoring (progen3-score)

Evaluates protein sequences by computing bidirectional autoregressive likelihood. For each sequence, computes:
1. Forward log-likelihood: $P(x_t | x_{<t})$ from N→C
2. Reverse log-likelihood: $P(x_t | x_{>t})$ from C→N
3. Averages the two for a robust score

Returns `log_likelihood`, `avg_log_likelihood`, and `perplexity` per sequence.

## Quick Start Examples

```python
# Forward generation (N→C)
from proto_tools.tools.causal_models.progen3 import (
    ProGen3SampleInput, ProGen3SampleConfig, run_progen3_sample,
)

inputs = ProGen3SampleInput(prompts=["MKTL"])
config = ProGen3SampleConfig(max_new_tokens=100, temperature=0.2)
result = run_progen3_sample(inputs, config)
print(result.sequences[0])
```

```python
# Score protein sequences
from proto_tools.tools.causal_models.progen3 import (
    ProGen3ScoringInput, ProGen3ScoringConfig, run_progen3_score,
)

inputs = ProGen3ScoringInput(sequences=["MVLSPADKTNVKAAW", "MKTLLILAVVAA"])
config = ProGen3ScoringConfig(model_checkpoint="progen3-762m")
result = run_progen3_score(inputs, config)
for i, score in enumerate(result.scores):
    print(f"Seq {i}: perplexity={score.metrics['perplexity']:.2f}")
```

## Configuration

### Sampling (`ProGen3SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `progen3-762m` | Model checkpoint to use |
| `local_path` | `str \| None` | `None` | Local weights path (bypasses HF download) |
| `direction` | `str` | `forward` | Generation direction: `forward` (N→C) or `reverse` (C→N) |
| `temperature` | `float` | `0.2` | Sampling temperature |
| `top_p` | `float` | `0.95` | Nucleus sampling threshold |
| `max_new_tokens` | `int` | `256` | Max new tokens to generate (excludes prompt) |
| `min_new_tokens` | `int` | `1` | Min new tokens before stopping |
| `num_sequences` | `int` | `1` | Sequences to generate per prompt |
| `include_prompt_in_output` | `bool` | `True` | Include prompt residues in output sequence |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |

### Scoring (`ProGen3ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `progen3-762m` | Model checkpoint to use |
| `local_path` | `str \| None` | `None` | Local weights path |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `reduction` | `str` | `mean` | How to aggregate per-token log-likelihoods (`mean` or `sum`) |

## Best Practices & Gotchas

- **Set `direction`** to `"forward"` (N→C) or `"reverse"` (C→N) for sampling
- **Use `progen3-112m`** for testing/development, larger models for production
- **Temperature 0.2** is the recommended default for protein generation
- **Bidirectional scoring** is automatic — no need to manually reverse sequences
- **bfloat16 required** — will not work on GPUs without bf16 support (A100/H100 recommended)
- **Non-commercial weights** — model weights are CC BY-NC-SA 4.0

## References

- **Paper**: Roney et al. "ProGen3: A Large-Scale Protein Language Model" (2025). bioRxiv doi:10.1101/2025.05.16.654471
- **GitHub**: https://github.com/Profluent-AI/progen3
- **HuggingFace**: https://huggingface.co/Profluent-Bio

## Related Tools

| Tool | Relationship |
|------|-------------|
| ProGen2 | Previous generation protein LM (smaller, faster) |
| Evo1/Evo2 | DNA language models (genomic sequences) |
| ESM2 | Masked protein LM (infilling, embeddings) |
| ProteinMPNN | Structure-conditioned inverse folding |
