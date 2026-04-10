<a href="https://bio-pro.mintlify.app/tools/causal-models/progen3"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ProGen3

## Overview

ProGen3 is a Mixture-of-Experts (MoE) protein language model from Profluent, based on the Mixtral/Mistral architecture. It supports autoregressive protein sequence generation in both forward (N→C) and reverse (C→N) directions, as well as bidirectional sequence scoring that averages forward and reverse log-likelihoods for more robust evaluation.

ProGen3 is available in six sizes from 112M to 3B parameters, trained on large-scale protein sequence databases. It uses [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) and [MegaBlocks](https://github.com/databricks/megablocks) for efficient sparse MoE computation.

## Background

Protein language models learn the statistical patterns of natural protein sequences, capturing evolutionary constraints on amino acid usage. ProGen3's bidirectional scoring (averaging N→C and C→N likelihoods) provides a more robust assessment of sequence naturalness than unidirectional models, since protein function depends on the complete [3D structure](https://en.wikipedia.org/wiki/Protein_structure) rather than just the N-to-C reading frame.

The [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) architecture activates only a subset of parameters per token, allowing larger model capacity without proportionally increasing compute cost.

## Tool Catalog

| Key | Operation | Input | Output | Use Case |
|-----|-----------|-------|--------|----------|
| `progen3-sample` | Generate | Prompt sequences | Generated proteins | Design novel proteins from N/C-terminal seeds |
| `progen3-score` | Score | Protein sequences | Perplexity metrics | Evaluate sequence naturalness/fitness |

## Model Variants

| Checkpoint | Parameters | Use Case |
|------------|-----------|----------|
| `progen3-112m` | 112M | Fast prototyping and testing |
| `progen3-219m` | 219M | Light production workloads |
| `progen3-339m` | 339M | Balanced speed/quality |
| `progen3-762m` | 762M | **Default.** Good quality, reasonable speed |
| `progen3-1b` | 1B | Higher quality generation and scoring |
| `progen3-3b` | 3B | Best quality, highest GPU memory requirement |

All checkpoints are available on HuggingFace under the [Profluent-Bio](https://huggingface.co/Profluent-Bio) organization. Weights are licensed CC BY-NC-SA 4.0 (non-commercial).

## Execution Modes

ProGen3 runs in an isolated standalone environment with its own Python 3.12 installation, Flash Attention 2, and MegaBlocks. Requires a GPU with bfloat16 support (A100/H100 recommended). The environment is built automatically on first use.

## How It Works

### Sampling (progen3-sample)

Generates protein sequences autoregressively from a prompt:

1. A direction token is prepended internally (`"1"` for forward, `"2"` for reverse)
2. The model generates tokens autoregressively until a stop token or `max_new_tokens` is reached
3. Special tokens and direction markers are stripped from the output
4. If `prepend_prompt=False`, prompt residues are removed

The `direction` config controls generation direction:
- `direction="forward"` (default): N→C generation, extending from the N-terminus
- `direction="reverse"`: C→N generation, extending from the C-terminus
- Empty string prompt (`""`) for unconditional generation in either direction

### Scoring (progen3-score)

Evaluates protein sequences using bidirectional autoregressive likelihood:

1. **Forward pass**: computes per-token log-likelihood $P(x_t | x_{<t})$ from N→C
2. **Reverse pass**: computes per-token log-likelihood $P(x_t | x_{>t})$ from C→N
3. **Averaging**: per-position likelihoods are averaged where both directions contribute; aggregate metrics average the two directional scores

Returns `log_likelihood`, `avg_log_likelihood`, and `perplexity` per sequence, plus optional per-position metrics for each direction.

## Input Parameters

### Sampling

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompts` | `list[str]` | Amino acid prompt sequences. Pass `""` for unconditional generation |

### Scoring

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Protein sequences to score (standard amino acids, N-to-C direction) |

## Configuration

### Sampling (`ProGen3SampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `progen3-762m` | Model checkpoint to use |
| `direction` | `str` | `forward` | Generation direction: `forward` (N→C) or `reverse` (C→N) |
| `temperature` | `float` | `0.2` | Sampling temperature. Lower = more deterministic |
| `top_p` | `float` | `0.95` | Nucleus sampling threshold |
| `max_new_tokens` | `int` | `256` | Max new tokens to generate (excludes prompt) |
| `min_new_tokens` | `int` | `1` | Min new tokens before stopping |
| `num_sequences` | `int` | `1` | Sequences to generate per prompt |
| `prepend_prompt` | `bool` | `True` | Include prompt residues in output sequence |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |

### Scoring (`ProGen3ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_checkpoint` | `str` | `progen3-762m` | Model checkpoint to use |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `reduction` | `str` | `mean` | How to aggregate per-token log-likelihoods: `mean` (per-token average) or `sum` (total) |

### Parameter Guides

**Temperature** controls sequence diversity:

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.1 | Very conservative, near-greedy | Extending a known motif faithfully |
| 0.2 | Low diversity (default) | General protein design |
| 0.5 | Moderate diversity | Exploring sequence variants |
| 0.8-1.0 | High diversity | Library generation, creative exploration |

### Sweep Priorities

1. **`temperature`**: Largest effect on generation diversity and quality
2. **`model_checkpoint`**: Larger models produce more natural sequences
3. **`top_p`**: Fine-tunes the sampling distribution tail

## Output Specification

### Sampling

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `list[str]` | Generated protein sequences (amino acid characters) |

Export formats: `fasta` (default), `txt`, `json`

### Scoring

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `list[SequenceScores]` | One entry per input sequence |
| `scores[i].metrics` | `dict` | `log_likelihood`, `avg_log_likelihood`, `perplexity` |
| `scores[i].per_position_metrics` | `dict \| None` | `forward_log_likelihood`, `reverse_log_likelihood`, `log_likelihood` (bidirectional average) per position |

Export formats: `csv` (default), `json`

## Interpreting Results

### Perplexity

Perplexity measures how "surprised" the model is by a sequence. Lower perplexity indicates the sequence is more consistent with natural proteins the model has seen.

| Perplexity | Confidence | Interpretation |
|------------|-----------|----------------|
| 1.0-3.0 | Very high | Highly natural; resembles well-represented protein families |
| 3.0-6.0 | High | Natural; reasonable for most design applications |
| 6.0-10.0 | Moderate | Somewhat unusual; may contain non-standard motifs |
| 10.0-15.0 | Low | Unusual sequence; review for biological plausibility |
| 15.0+ | Very low | Likely unnatural or outside training distribution |

### Log-likelihood

Use `avg_log_likelihood` (per-token, length-normalized) when comparing sequences of different lengths. Use `log_likelihood` with `reduction="sum"` only when comparing sequences of the same length.

### Bidirectional scoring

ProGen3 always scores in both directions automatically. The bidirectional average is more robust than either direction alone because:
- Forward-only scoring can miss C-terminal context
- Reverse-only scoring can miss N-terminal context
- The average captures dependencies in both directions

Per-position metrics include separate `forward_log_likelihood` and `reverse_log_likelihood` lists, plus a bidirectional `log_likelihood` average. Positions where only one direction contributes (first/last residue) use the single available value.

## Quick Start Examples

```python
# Example 1: Basic forward generation (N→C)
from proto_tools.tools.causal_models.progen3 import (
    ProGen3SampleInput, ProGen3SampleConfig, run_progen3_sample,
)

inputs = ProGen3SampleInput(prompts=["MKTL"])
config = ProGen3SampleConfig(max_new_tokens=100, temperature=0.2)
result = run_progen3_sample(inputs, config)
print(result.sequences[0])
```

```python
# Example 2: Reverse generation (C→N)
inputs = ProGen3SampleInput(prompts=["RYTEAFLK"])
config = ProGen3SampleConfig(direction="reverse", max_new_tokens=100)
result = run_progen3_sample(inputs, config)
print(result.sequences[0])
```

```python
# Example 3: Batch generation with diversity
inputs = ProGen3SampleInput(prompts=["MKTL", "MVLS", ""])
config = ProGen3SampleConfig(
    num_sequences=3,
    temperature=0.5,
    max_new_tokens=150,
)
result = run_progen3_sample(inputs, config)
for i, seq in enumerate(result.sequences):
    print(f"Sequence {i}: {seq[:60]}... ({len(seq)} aa)")
```

```python
# Example 4: Score and rank protein sequences
from proto_tools.tools.causal_models.progen3 import (
    ProGen3ScoringInput, ProGen3ScoringConfig, run_progen3_score,
)

inputs = ProGen3ScoringInput(sequences=[
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF",  # Hemoglobin alpha
    "MKTLLILAVVAAALA",  # Signal peptide
    "AAAAAAAAAAAAAAAAA",  # Polyalanine (low complexity)
])
result = run_progen3_score(inputs)
for i, score in enumerate(result.scores):
    print(f"Seq {i}: perplexity={score.metrics['perplexity']:.2f}")
```

```python
# Example 5: Per-position scoring analysis
inputs = ProGen3ScoringInput(sequences=["MVLSPADKTNVKAAW"])
result = run_progen3_score(inputs)

score = result.scores[0]
per_pos = score.per_position_metrics
if per_pos:
    fwd = per_pos["forward_log_likelihood"]
    rev = per_pos["reverse_log_likelihood"]
    avg = per_pos["log_likelihood"]
    for pos, (f, r, a) in enumerate(zip(fwd, rev, avg)):
        f_str = f"{f:.3f}" if f is not None else "N/A"
        r_str = f"{r:.3f}" if r is not None else "N/A"
        a_str = f"{a:.3f}" if a is not None else "N/A"
        print(f"Position {pos}: fwd={f_str}  rev={r_str}  avg={a_str}")
```

## Best Practices & Gotchas

- **bfloat16 required**: ProGen3 will not work on GPUs without bf16 support. A100/H100 recommended
- **Non-commercial weights**: model weights are CC BY-NC-SA 4.0
- **Use `progen3-112m` for testing**: much faster iteration; switch to larger checkpoints for production
- **Temperature 0.2 is the recommended default** for protein generation. Higher temperatures increase diversity but reduce naturalness
- **Bidirectional scoring is automatic**: no need to manually reverse sequences or run separate passes
- **`max_new_tokens` excludes the prompt**: unlike ProGen2's `max_length`, ProGen3's `max_new_tokens` counts only newly generated tokens
- **Empty prompt for unconditional generation**: pass `""` as a prompt, not `None`
- **Scoring reduction matters**: use `mean` (default) for comparing sequences of different lengths; `sum` only for fixed-length comparisons

## Seed Reproducibility

ProGen3's MoE forward pass is non-deterministic: the `grouped-gemm` CUDA
kernels used by MegaBlocks to batch GEMMs across experts accumulate
reductions in non-deterministic order, so outputs drift by ~1e-3 in
log-likelihood across independent invocations with the same seed
(amplified to completely different sequences under autoregressive
sampling). For bit-exact repeat calls, keep the model in a single
persistent worker. Acknowledged upstream:

- [Profluent-AI/progen3#6 — Reproducibility issue in computing model logits](https://github.com/Profluent-AI/progen3/issues/6)
- [databricks/megablocks#83 — ParallelDroplessMLP initialises self.mlp twice](https://github.com/databricks/megablocks/issues/83)

## References

- **Paper**: Roney et al. "ProGen3: A Large-Scale Protein Language Model" (2025). [bioRxiv doi:10.1101/2025.05.16.654471](https://doi.org/10.1101/2025.05.16.654471)
- **GitHub**: https://github.com/Profluent-AI/progen3
- **HuggingFace**: https://huggingface.co/Profluent-Bio

## Related Tools

| Tool | Relationship |
|------|-------------|
| ProGen2 | Previous generation protein LM from Salesforce (smaller, faster, no bidirectional scoring) |
| Evo1/Evo2 | DNA language models (genomic sequences, not protein) |
| ESM2 | Masked protein LM (infilling and embeddings, not autoregressive) |
| ESM3 | Generative protein model (joint sequence/structure, masked approach) |
| ProteinMPNN | Structure-conditioned inverse folding (requires a backbone structure) |
