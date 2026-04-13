<a href="https://bio-pro.mintlify.app/tools/masked-models/ablang"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AbLang

## Overview
AbLang is an antibody-specific language model family from the Oxford Protein Informatics Group (OPIG), trained on antibody sequences from the Observed Antibody Space (OAS). Built on a BERT/MLM architecture, it provides antibody sequence embeddings, pseudo-log-likelihood scoring, and masked residue restoration. The tool wraps three model variants — ablang1-heavy, ablang1-light, and ablang2-paired — with automatic model routing based on input format.

This package also includes `ablang-germinal-gradient`, a Germinal-specific relaxed-sequence gradient backend that mirrors Germinal's existing `CustomAbLang` adapter. It is not a generic AbLang naturalness objective: it uses Germinal's shifted cross-entropy objective, Germinal's scFv chain handling, and an internal mapping from proto-language's canonical protein order `ACDEFGHIKLMNPQRSTVWY` into Germinal's AbLang order `ARNDCQEGHILKMFPSTWYV`.

## Background

**Why an antibody-specific language model?**
General protein language models like ESM2 are trained on diverse protein families. Antibodies have a unique architecture (variable/constant domains, CDR loops, framework regions) and evolve through somatic hypermutation rather than standard evolutionary selection. AbLang captures antibody-specific patterns by training exclusively on antibody sequences, including:
- **CDR variability**: Complementarity-determining regions that bind antigens
- **Framework conservation**: Structural scaffold residues
- **Chain pairing**: Co-evolutionary patterns between heavy and light chains
- **Species-specific signatures**: Human, mouse, and other germline patterns

**Why paired models matter:**
Heavy and light chains co-evolve to form functional antibodies. The paired model (`ablang2-paired`) processes both chains together using a pipe (`|`) separator, capturing inter-chain dependencies that single-chain models miss.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `ablang-embedding` | Extract antibody embeddings | Embeddings, attention masks |
| `ablang-germinal-gradient` | Germinal-specific AbLang relaxed-sequence gradient | Gradient, loss, metrics, vocab |
| `ablang-sample` | Restore masked (`_`) positions | Completed sequences |
| `ablang-score` | Score sequences via pseudo-log-likelihood | Per-sequence metrics |

## Model Variants

| Checkpoint | Chain Type | Embedding Dim | Use Case |
|------------|-----------|---------------|----------|
| `ablang1-heavy` | Heavy chain only | 768 | When only VH sequences are available |
| `ablang1-light` | Light chain only | 768 | When only VL/VK sequences are available |
| `ablang2-paired` | Heavy + light (pipe-separated) | 480 | When both chains are available (recommended) |

**Automatic model routing (`model_choice="auto"`, the default):**

When `model_choice` is set to `"auto"` (the default), the tool automatically selects the best model based on your input:
1. **Paired sequences** (containing `|`): Routes to `ablang2-paired`
2. **Single-chain sequences**: Routes to `ablang1-heavy`

To use `ablang1-light`, set `model_choice="ablang1-light"` explicitly.

**Manual model selection guidance:**
1. **Default choice**: Leave `model_choice="auto"` — it handles most cases correctly.
2. **Light chains**: Explicitly set `model_choice="ablang1-light"` when working with VL/VK sequences.
3. **Paired input format**: For the paired model, join heavy and light chains with a pipe separator: `"EVQLVES...|DIQMTQ..."`.

## Execution Modes

- **Local GPU/CPU**: Loads the model on-demand. Use `device="cuda"`, `"cpu"`, or `"mps"`.

## How It Works

AbLang is a masked language model (BERT architecture) trained on antibody sequences from OAS. It learns to predict masked amino acids from surrounding context, capturing antibody-specific evolutionary and structural patterns.

- **Embeddings**: A forward pass produces per-position hidden states. Mean-pooling across positions (ignoring padding) yields a fixed-length sequence descriptor (768-dim for ablang1, 480-dim for ablang2-paired).
- **Sampling**: Masked positions (indicated by `_`) are filled in by the model's predicted distribution using greedy decoding (argmax).
- **Scoring**: Each position is masked one at a time, and the model's log-probability of the true amino acid is recorded. Aggregated scores give pseudo-log-likelihood (higher = more "natural" antibody sequence).

## Input Parameters

### Embeddings Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Antibody sequences (amino acid strings) |

### Sampling Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Antibody sequences with `_` at positions to restore |

### Scoring Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Antibody sequences to score |

### Germinal Gradient Tool

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `list[list[float]]` | Relaxed sequence logits with shape `(L, 20)` in canonical protein order `ACDEFGHIKLMNPQRSTVWY` |
| `temperature` | `float` | Softmax temperature used to relax logits into probabilities |

## Configuration

### Embeddings Tool (`AbLangEmbeddingsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_choice` | `str` | `"auto"` | Model checkpoint: `"auto"`, `"ablang1-heavy"`, `"ablang1-light"`, `"ablang2-paired"` |
| `batch_size` | `int` | `1` | Sequences per GPU forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

### Sampling Tool (`AbLangSampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_choice` | `str` | `"auto"` | Model variant: `"auto"`, `"ablang1-heavy"`, `"ablang1-light"`, `"ablang2-paired"` |
| `batch_size` | `int` | `1` | Sequences per forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

### Scoring Tool (`AbLangScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_choice` | `str` | `"auto"` | Model checkpoint: `"auto"`, `"ablang1-heavy"`, `"ablang1-light"`, `"ablang2-paired"` |
| `scoring_mode` | `str` | `"pseudo_log_likelihood"` | Scoring method: `"pseudo_log_likelihood"` or `"confidence"` |
| `batch_size` | `int` | `1` | Sequences per forward pass |
| `device` | `str` | `"cuda"` | Device: `"cuda"`, `"cpu"`, `"mps"` |

### Germinal Gradient Tool (`AbLangGerminalGradientConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_single_chain_variable_fragment` | `bool` | `False` | Use Germinal's paired single-chain variable fragment path instead of the single-domain heavy-chain antibody path |
| `heavy_chain_first` | `bool` | `True` | Whether the paired layout is heavy chain, then linker, then light chain |
| `heavy_chain_length` | `int \| null` | `null` | Number of heavy-chain residues in the full relaxed sequence; required when paired mode is enabled |
| `light_chain_length` | `int \| null` | `null` | Number of light-chain residues in the full relaxed sequence; required when paired mode is enabled |
| `seed` | `int \| null` | `0` | Optional PyTorch random seed used to mirror Germinal's adapter initialization |
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

### AbLangGerminalGradientOutput

| Field | Type | Description |
|-------|------|-------------|
| `gradient` | `list[list[float]]` | Gradient matrix with the same shape as the input logits |
| `loss` | `float` | Germinal's shifted cross-entropy loss |
| `metrics` | `dict[str, Any]` | Auxiliary metrics such as `log_likelihood`, `linker_length`, `model_choice`, and `objective` |
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

**Example 1: Auto-routed embeddings (recommended)**
```python
from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsInput, run_ablang_embeddings,
)
import numpy as np

# Heavy chain sequences — auto-routes to ablang1-heavy
sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPG",
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPG",
]

result = run_ablang_embeddings(AbLangEmbeddingsInput(sequences=sequences))

# Compare embeddings
emb1 = np.array(result.results[0].mean_embedding)
emb2 = np.array(result.results[1].mean_embedding)
cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Model used: {result.metadata['model_choice']}")  # ablang1-heavy
print(f"Cosine similarity: {cosine_sim:.3f}")
```

**Example 2: Paired heavy+light chain embeddings (auto-routed)**
```python
from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsInput, run_ablang_embeddings,
)

# Paired sequences with pipe separator — auto-routes to ablang2-paired
sequences = [
    "EVQLVESGGGLVQPGG|DIQMTQSPSSLSASVG",
    "QVQLVESGGGVVQPGR|EIVLTQSPATLSLSPG",
]

result = run_ablang_embeddings(AbLangEmbeddingsInput(sequences=sequences))
print(f"Model used: {result.metadata['model_choice']}")  # ablang2-paired
print(f"Embedding dim: {len(result.results[0].mean_embedding)}")  # 480
```

**Example 3: Score antibody sequences**
```python
from proto_tools.tools.masked_models.ablang import (
    AbLangScoringInput, AbLangScoringConfig, run_ablang_score,
)

inputs = AbLangScoringInput(sequences=[
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPG",
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPG",
])
config = AbLangScoringConfig(scoring_mode="pseudo_log_likelihood")

result = run_ablang_score(inputs, config)
for i, score in enumerate(result.scores):
    print(f"Sequence {i}: {score.metrics}")
```

**Example 4: Restore masked CDR positions**
```python
from proto_tools.tools.masked_models.ablang import (
    AbLangSampleInput, run_ablang_sample,
)

# Mask CDR3 positions with _
inputs = AbLangSampleInput(sequences=[
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFS___MSWVRQAPG",
])

result = run_ablang_sample(inputs)
print(f"Original: {inputs.sequences[0]}")
print(f"Restored: {result.sequences[0]}")
```

**Example 5: Explicit light chain model selection**
```python
from proto_tools.tools.masked_models.ablang import (
    AbLangEmbeddingsInput, AbLangEmbeddingsConfig, run_ablang_embeddings,
)

# Light chain — must specify model_choice explicitly (auto defaults to heavy)
inputs = AbLangEmbeddingsInput(sequences=["DIQMTQSPSSLSASVGDRVTITC"])
config = AbLangEmbeddingsConfig(model_choice="ablang1-light")

result = run_ablang_embeddings(inputs, config)
print(f"Model used: {result.metadata['model_choice']}")  # ablang1-light
print(f"Embedding dim: {len(result.results[0].mean_embedding)}")  # 768
```

## Best Practices & Gotchas

**Auto-routing:**

1. **Default behavior**: With `model_choice="auto"` (the default), paired sequences (containing `|`) route to `ablang2-paired`, and single-chain sequences route to `ablang1-heavy`.

2. **Light chains require explicit selection**: Auto-routing cannot distinguish heavy from light chains, so it defaults to heavy. Set `model_choice="ablang1-light"` when working with VL/VK sequences.

**Germinal gradient backend:**

1. **Germinal-only semantics**: `ablang-germinal-gradient` mirrors Germinal's upstream `CustomAbLang` adapter rather than a generic AbLang objective.

2. **Vocab order**: The gradient tool accepts and returns logits in canonical protein order `ACDEFGHIKLMNPQRSTVWY`. Internally it maps to AbLang's token vocabulary for the forward pass.

3. **Single-chain variable fragment linker behavior**: When `use_single_chain_variable_fragment=True`, the gradient excludes linker residues from the AbLang objective and returns exact zero rows for linker positions.

**Chain type matching:**

1. **Match checkpoint to input**: Use `ablang1-heavy` for heavy chains, `ablang1-light` for light chains, and `ablang2-paired` for paired sequences. Mismatching will produce poor results.

2. **Paired format**: For `ablang2-paired`, join heavy and light chains with a pipe separator (`|`). Do not include spaces around the pipe.

**Batch processing:**

1. **Variable lengths**: Sequences are padded to batch max length — group similar-length sequences for efficiency.

2. **OOM errors**: Reduce `batch_size`. Antibody sequences are typically short (~120 residues per chain), so larger batches are feasible.

3. **Attention masks**: Always use masks to ignore padding positions in downstream analysis.

**Common mistakes:**

1. **Wrong mask character**: Use `_` (underscore) for masked positions in the sampling tool, not `*`, `<mask>`, or `X`.

2. **Wrong checkpoint for chain type**: Using `ablang1-heavy` with light chain sequences will produce meaningless results.

3. **Missing pipe in paired mode**: The paired model expects `heavy|light` format. Passing unpaired sequences to `ablang2-paired` will produce incorrect embeddings.

4. **Mixing checkpoints**: Embeddings from different checkpoints (heavy vs. light vs. paired) are NOT comparable.

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
