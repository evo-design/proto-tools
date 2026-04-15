<a href="https://bio-pro.mintlify.app/tools/masked-models/ablang"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AbLang

## Overview
AbLang is an antibody-specific language model family from the Oxford Protein Informatics Group (OPIG), trained on antibody sequences from the Observed Antibody Space (OAS). Built on a BERT/MLM architecture, it provides antibody sequence embeddings, pseudo-log-likelihood scoring, and masked residue restoration. The tool wraps three model variants — ablang1-heavy, ablang1-light, and ablang2-paired — with automatic model routing based on which chains are provided on the `Antibody` input.

This package also includes `ablang-gradient`, a relaxed-sequence gradient tool that computes a shifted cross-entropy objective over relaxed antibody logits (via `AntibodyLogits`). It supports all three model variants and uses an internal mapping from proto-language's canonical protein order `ACDEFGHIKLMNPQRSTVWY` into AbLang's token vocabulary order `ARNDCQEGHILKMFPSTWYV`.

## Background

**Why an antibody-specific language model?**
General protein language models like ESM2 are trained on diverse protein families. Antibodies have a unique architecture (variable/constant domains, CDR loops, framework regions) and evolve through somatic hypermutation rather than standard evolutionary selection. AbLang captures antibody-specific patterns by training exclusively on antibody sequences, including:
- **CDR variability**: Complementarity-determining regions that bind antigens
- **Framework conservation**: Structural scaffold residues
- **Chain pairing**: Co-evolutionary patterns between heavy and light chains
- **Species-specific signatures**: Human, mouse, and other germline patterns

**Why paired models matter:**
Heavy and light chains co-evolve to form functional antibodies. The paired model (`ablang2-paired`) processes both chains together, capturing inter-chain dependencies that single-chain models miss.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `ablang-embedding` | Extract antibody embeddings | Embeddings, attention masks |
| `ablang-gradient` | AbLang relaxed-sequence gradient | Gradient, loss, metrics, vocab |
| `ablang-sample` | Restore masked (`_`) positions | Completed sequences |
| `ablang-score` | Score sequences via pseudo-log-likelihood | Per-sequence metrics |

## Model Variants

All AbLang tools accept `Antibody` objects as input (from `proto_tools.entities.antibody`). The model variant is selected automatically based on which chains are provided:

| Input | Model Selected |
|-------|---------------|
| `Antibody(heavy_chain="EVQL...")` | `ablang1-heavy` |
| `Antibody(light_chain="DIQM...")` | `ablang1-light` |
| `Antibody(heavy_chain="EVQL...", light_chain="DIQM...")` | `ablang2-paired` |

At least one chain must be provided. For the gradient tool, use `AntibodyLogits` which accepts logit distributions instead of sequence strings.

| Checkpoint | Chain Type | Embedding Dim | Use Case |
|------------|-----------|---------------|----------|
| `ablang1-heavy` | Heavy chain only | 768 | When only VH sequences are available |
| `ablang1-light` | Light chain only | 768 | When only VL/VK sequences are available |
| `ablang2-paired` | Heavy + light | 480 | When both chains are available (recommended) |

## Execution Modes

- **Local GPU/CPU**: Loads the model on-demand. Use `device="cuda"`, `"cpu"`, or `"mps"`.

## How It Works

AbLang is a masked language model (BERT architecture) trained on antibody sequences from OAS. It learns to predict masked amino acids from surrounding context, capturing antibody-specific evolutionary and structural patterns.

- **Embeddings**: A forward pass produces per-position hidden states. Mean-pooling across positions (ignoring padding) yields a fixed-length sequence descriptor (768-dim for ablang1, 480-dim for ablang2-paired).
- **Sampling**: Masked positions (indicated by `_`) are filled in by the model's predicted distribution using greedy decoding (argmax).
- **Scoring**: Each position is masked one at a time, and the model's log-probability of the true amino acid is recorded. Aggregated scores give pseudo-log-likelihood (higher = more "natural" antibody sequence).

## Input Parameters

### Embeddings, Sampling, and Scoring Tools

| Parameter | Type | Description |
|-----------|------|-------------|
| `antibodies` | `list[Antibody]` | Antibody objects with `heavy_chain` and/or `light_chain` sequences |

For the sampling tool, chain sequences should contain `_` at positions to restore.

### Gradient Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `antibody` | `AntibodyLogits` | Antibody with `heavy_chain` and/or `light_chain` as distribution or logit matrices with shape `(L, 20)` in canonical protein order |
| `temperature` | `float \| null` | Optional softmax temperature. When set, applies `softmax(input / temperature)` before gradient computation. When `null` (default), input is used as-is — callers provide a pre-computed distribution |

## Configuration

### Embeddings Tool (`AbLangEmbeddingsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

### Sampling Tool (`AbLangSampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `1` | Sequences per forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

### Scoring Tool (`AbLangScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scoring_mode` | `str` | `"pseudo_log_likelihood"` | Scoring method: `"pseudo_log_likelihood"` or `"confidence"` |
| `batch_size` | `int` | `1` | Sequences per forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

### Gradient Tool (`AbLangGradientConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ste` | `bool` | `false` | When true, uses a Straight-Through Estimator: hard one-hot tokens in the forward pass with gradients flowing through soft probabilities. When false, uses soft blended embeddings |
| `seed` | `int \| null` | `null` | Optional PyTorch random seed for reproducibility |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

## Output Specification

### AbLangEmbeddingsOutput

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[SequenceEmbedding]` | Per-sequence embedding results |

**`SequenceEmbedding` fields:**
| Field | Type | Description |
|-------|------|-------------|
| `mean_embedding` | `list[float]` | Mean-pooled embedding vector (768-dim for ablang1, 480-dim for ablang2-paired) |
| `attention_mask` | `list[int]` | Binary mask: 1 = valid position, 0 = padding |

### AbLangSampleOutput

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `list[str]` | Sequences with masked positions restored |

### AbLangScoringOutput

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `list[MaskedModelScoringMetrics]` | One score object per input sequence |

**`MaskedModelScoringMetrics` fields:**
| Field | Type | Description |
|-------|------|-------------|
| `pseudo_log_likelihood`, `confidence` | `float` | Scoring metrics (attribute or mapping access: `score.pseudo_log_likelihood` or `score["pseudo_log_likelihood"]`) |

### AbLangGradientOutput

| Field | Type | Description |
|-------|------|-------------|
| `gradient` | `list[list[float]]` | Gradient matrix with the same shape as the input logits |
| `loss` | `float` | Shifted cross-entropy loss |
| `metrics` | `dict[str, Any]` | Auxiliary metrics: `log_likelihood`, `sequence_length`, `model_choice`, and `objective` |
| `vocab` | `list[str]` | Amino-acid column order for both the input logits and the returned gradient; always canonical protein order `ACDEFGHIKLMNPQRSTVWY` |

## Interpreting Results

These thresholds are heuristics. Use them comparatively and validate for your task.

**For sequence similarity (cosine similarity of embeddings):**
- **Highly similar**: > 0.9 (same antibody lineage or near-identical CDRs)
- **Related**: 0.7 - 0.9 (same germline family or similar binding properties)
- **Distant**: 0.5 - 0.7 (different germline families)
- **Unrelated**: < 0.5

**For scoring (pseudo-log-likelihood):**
- Higher pseudo-log-likelihood indicates a more "natural" antibody sequence
- Compare scores across variants rather than interpreting absolute values
- Useful for ranking humanization candidates or assessing developability

## Quick Start Examples

**Example 1: Heavy chain embeddings**
```python
from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsInput, run_ablang_embeddings,
)
import numpy as np

antibodies = [
    Antibody(heavy_chain="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPG"),
    Antibody(heavy_chain="QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPG"),
]

result = run_ablang_embeddings(AbLangEmbeddingsInput(antibodies=antibodies))

emb1 = np.array(result.results[0].mean_embedding)
emb2 = np.array(result.results[1].mean_embedding)
cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Model used: {result.metadata['model_choice']}")  # ablang1-heavy
print(f"Cosine similarity: {cosine_sim:.3f}")
```

**Example 2: Paired heavy+light chain embeddings**
```python
from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsInput, run_ablang_embeddings,
)

antibodies = [
    Antibody(heavy_chain="EVQLVESGGGLVQPGG", light_chain="DIQMTQSPSSLSASVG"),
    Antibody(heavy_chain="QVQLVESGGGVVQPGR", light_chain="EIVLTQSPATLSLSPG"),
]

result = run_ablang_embeddings(AbLangEmbeddingsInput(antibodies=antibodies))
print(f"Model used: {result.metadata['model_choice']}")  # ablang2-paired
print(f"Embedding dim: {len(result.results[0].mean_embedding)}")  # 480
```

**Example 3: Score antibody sequences**
```python
from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang import (
    AbLangScoringInput, AbLangScoringConfig, run_ablang_score,
)

inputs = AbLangScoringInput(antibodies=[
    Antibody(heavy_chain="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPG"),
    Antibody(heavy_chain="QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPG"),
])
config = AbLangScoringConfig(scoring_mode="pseudo_log_likelihood")

result = run_ablang_score(inputs, config)
for i, score in enumerate(result.scores):
    print(f"Sequence {i}: {score.metrics}")
```

**Example 4: Restore masked CDR positions**
```python
from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang import (
    AbLangSampleInput, run_ablang_sample,
)

inputs = AbLangSampleInput(antibodies=[
    Antibody(heavy_chain="EVQLVESGGGLVQPGGSLRLSCAASGFTFS___MSWVRQAPG"),
])

result = run_ablang_sample(inputs)
print(f"Restored: {result.sequences[0]}")
```

**Example 5: Light chain embeddings**
```python
from proto_tools.entities.antibody import Antibody
from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsInput, run_ablang_embeddings,
)

inputs = AbLangEmbeddingsInput(antibodies=[Antibody(light_chain="DIQMTQSPSSLSASVGDRVTITC")])

result = run_ablang_embeddings(inputs)
print(f"Model used: {result.metadata['model_choice']}")  # ablang1-light
print(f"Embedding dim: {len(result.results[0].mean_embedding)}")  # 768
```

## Best Practices & Gotchas

**Auto-routing:**

1. **No `model_choice` needed**: The model variant is selected automatically from which chains are provided on the `Antibody` input. Provide `heavy_chain` for ablang1-heavy, `light_chain` for ablang1-light, or both for ablang2-paired.

**Gradient tool:**

1. **Vocab order**: The gradient tool accepts and returns logits in canonical protein order `ACDEFGHIKLMNPQRSTVWY`. Internally it maps to AbLang's token vocabulary for the forward pass.

2. **Paired sequences**: Provide both `heavy_chain` and `light_chain` on `AntibodyLogits` — the tool concatenates them and inserts the chain separator automatically.

**Chain type matching:**

1. **Match chains to purpose**: Use `heavy_chain` for heavy chains, `light_chain` for light chains. The model variant is auto-selected to match.

**Batch processing:**

1. **Variable lengths**: Sequences are padded to batch max length — group similar-length sequences for efficiency.

2. **OOM errors**: Reduce `batch_size`. Antibody sequences are typically short (~120 residues per chain), so larger batches are feasible.

3. **Attention masks**: Always use masks to ignore padding positions in downstream analysis.

**Common mistakes:**

1. **Wrong mask character**: Use `_` (underscore) for masked positions in the sampling tool, not `*`, `<mask>`, or `X`.

2. **Mixing checkpoints**: Embeddings from different model variants (heavy vs. light vs. paired) are NOT comparable.

## References

**Primary publication:**
- Olsen, T.H. et al. (2024). "AbLang2: Addressing the antibody language model." *Bioinformatics Advances*, 4(1), vbae040. DOI: [10.1093/bioadv/vbae040](https://doi.org/10.1093/bioadv/vbae040)

**Implementation:**
- GitHub: [https://github.com/oxpig/AbLang2](https://github.com/oxpig/AbLang2)

## Related Tools

**Tools often used together:**
- `structure_prediction/esmfold`: Predict antibody structure from sequence
- `inverse_folding/proteinmpnn`: Structure-conditioned antibody sequence design
- `mmseqs-clustering`: Cluster antibody sequences before/after embedding analysis

**Alternative tools:**
- `esm2`: General protein language model (broader training data, but not antibody-specific)
