<a href="https://bio-pro.mintlify.app/tools/rna-splicing/splice-transformer"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# SpliceTransformer

## Overview
SpliceTransformer is a deep learning model for predicting [splice sites](https://en.wikipedia.org/wiki/RNA_splicing) with tissue-specific resolution. It identifies splice donors (5' splice sites) and acceptors (3' splice sites) across 15 human tissues, enabling analysis of [alternative splicing](https://en.wikipedia.org/wiki/Alternative_splicing) patterns and tissue-specific isoform usage. The model uses [transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) architecture with long-range sequence context for accurate splice site prediction.

## When to Use This Tool

**Primary use cases:**
- Predicting splice sites in human sequences
- Identifying tissue-specific splicing patterns
- Analyzing variant effects on splicing
- Designing synthetic introns and exons
- Understanding alternative splicing regulation

**When NOT to use this tool:**
- For non-human species: Model trained on human data only
- For gene expression (not splicing): Use Enformer or Borzoi
- For protein-level analysis: Use protein language models (ESM2, ESM3)
- For short sequences (<1kb): Need sufficient context for accurate predictions

## Biological Background

**What is splicing?**
[Pre-mRNA splicing](https://en.wikipedia.org/wiki/RNA_splicing) removes [introns](https://en.wikipedia.org/wiki/Intron) and joins [exons](https://en.wikipedia.org/wiki/Exon):
- **Donor site (5' splice site)**: End of exon, beginning of intron (GT dinucleotide)
- **Acceptor site (3' splice site)**: End of intron, beginning of exon (AG dinucleotide)
- **Branch point**: Internal intron sequence for lariat formation

**Tissue-specific splicing:**
Different tissues express different splice isoforms:
- Tissue-specific [splicing factors](https://en.wikipedia.org/wiki/Splicing_factor)
- Alternative exon inclusion/exclusion
- Alternative 5'/3' splice site usage

**The 15 tissues:**
| Index | Tissue |
|-------|--------|
| 0 | Adipose tissue |
| 1 | Blood |
| 2 | Blood vessel |
| 3 | Brain |
| 4 | Colon |
| 5 | Heart |
| 6 | Kidney |
| 7 | Liver |
| 8 | Lung |
| 9 | Muscle |
| 10 | Nerve |
| 11 | Small intestine |
| 12 | Skin |
| 13 | Spleen |
| 14 | Stomach |

## Execution Modes

SpliceTransformer runs on GPU (recommended) or CPU:
- **GPU**: Recommended for practical use. Long sequences require significant GPU memory.
- **CPU**: Possible but very slow. Not recommended for batch processing.

## How It Works

**Architecture:**
1. **Input**: Target sequence (1kb) + left context (4kb) + right context (4kb) = 9kb total
2. **Transformer**: Processes sequence to capture long-range dependencies
3. **Output**: Per-position predictions across 18 channels

**Output channels (18 total):**

| Channels | Description |
|----------|-------------|
| 0 | Neither donor nor acceptor |
| 1 | Acceptor splice site probability |
| 2 | Donor splice site probability |
| 3-17 | Tissue-specific splice site usage (15 tissues) |

**Resolution:**
- Single nucleotide resolution over target sequence
- Context provides information for accurate prediction

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_seqs` | `List[str]` | Target sequences for prediction (typically 1000bp) |
| `left_contexts` | `List[str]` | Left flanking sequences (must match `context_length`) |
| `right_contexts` | `List[str]` | Right flanking sequences (must match `context_length`) |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_length` | `int` | `4000` | Length of left/right context sequences |
| `device` | `str` | `cuda` | Device: `cuda`, `cpu` |
| `verbose` | `bool` | `False` | Print progress messages |

## Output Specification

### SpliceTransformerOutput

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `prediction` | `list[list[list[float]]]` | `[batch, target_len, 18]` | Per-position predictions (convert to numpy with `np.array(...)` for slicing) |

**Prediction channels:**
| Index | Content |
|-------|---------|
| 0 | P(neither) |
| 1 | P(acceptor) |
| 2 | P(donor) |
| 3 | Adipose tissue |
| 4 | Blood |
| 5 | Blood vessel |
| 6 | Brain |
| 7 | Colon |
| 8 | Heart |
| 9 | Kidney |
| 10 | Liver |
| 11 | Lung |
| 12 | Muscle |
| 13 | Nerve |
| 14 | Small intestine |
| 15 | Skin |
| 16 | Spleen |
| 17 | Stomach |

## Interpreting Results

**Splice site detection:**
| Probability | Interpretation |
|-------------|----------------|
| > 0.8 | Strong splice site |
| 0.5 - 0.8 | Moderate confidence |
| 0.2 - 0.5 | Weak/alternative site |
| < 0.2 | Not a splice site |

**Tissue-specific interpretation:**
- Compare tissue channels to identify tissue-specific sites
- Higher values indicate stronger usage in that tissue
- Differential analysis reveals alternative splicing

**Channel relationships:**
- Channels 0, 1, 2 roughly sum to 1 (softmax over splice types)
- Tissue channels (3-17) represent tissue-specific usage of identified splice sites
- A position can be a strong splice site overall but only active in specific tissues

## Quick Start Examples

**Example 1: Basic splice site prediction**
```python
from proto_tools.tools.rna_splicing.splice_transformer import (
    run_splice_transformer, SpliceTransformerInput, SpliceTransformerConfig
)
import numpy as np

# 1kb target with 4kb context on each side
target = "ATGC" * 250  # 1000bp
left_ctx = "CGTA" * 1000  # 4000bp
right_ctx = "TACG" * 1000  # 4000bp

inputs = SpliceTransformerInput(
    target_seqs=[target],
    left_contexts=[left_ctx],
    right_contexts=[right_ctx]
)
config = SpliceTransformerConfig(context_length=4000, verbose=True)

result = run_splice_transformer(inputs, config)
pred = np.array(result.prediction)
print(f"Prediction shape: {pred.shape}")  # (1, 1000, 18)
```

**Example 2: Find donor and acceptor sites**
```python
from proto_tools.tools.rna_splicing.splice_transformer import (
    run_splice_transformer, SpliceTransformerInput, SpliceTransformerConfig
)
import numpy as np

# Your genomic sequence
inputs = SpliceTransformerInput(
    target_seqs=[target_seq],
    left_contexts=[left_ctx],
    right_contexts=[right_ctx]
)
config = SpliceTransformerConfig()

result = run_splice_transformer(inputs, config)

# Convert to numpy for slicing
pred = np.array(result.prediction[0])  # First sequence
acceptor_probs = pred[:, 1]  # Channel 1
donor_probs = pred[:, 2]     # Channel 2

# Find high-confidence sites
acceptor_sites = np.where(acceptor_probs > 0.5)[0]
donor_sites = np.where(donor_probs > 0.5)[0]

print(f"Acceptor sites: {acceptor_sites}")
print(f"Donor sites: {donor_sites}")
```

**Example 3: Tissue-specific analysis**
```python
from proto_tools.tools.rna_splicing.splice_transformer import (
    run_splice_transformer, SpliceTransformerInput, SpliceTransformerConfig,
    SPLICE_TISSUE_CHANNEL_INDEX
)
import numpy as np

inputs = SpliceTransformerInput(
    target_seqs=[target_seq],
    left_contexts=[left_ctx],
    right_contexts=[right_ctx]
)
config = SpliceTransformerConfig()

result = run_splice_transformer(inputs, config)
pred = np.array(result.prediction[0])

# Get brain-specific splicing
brain_idx = SPLICE_TISSUE_CHANNEL_INDEX["BRAIN"]
brain_splicing = pred[:, brain_idx]

# Get liver-specific splicing
liver_idx = SPLICE_TISSUE_CHANNEL_INDEX["LIVER"]
liver_splicing = pred[:, liver_idx]

# Find positions with tissue-specific difference
diff = brain_splicing - liver_splicing
brain_specific = np.where(diff > 0.3)[0]
print(f"Brain-specific splice sites: {brain_specific}")
```

**Example 4: Batch processing**
```python
from proto_tools.tools.rna_splicing.splice_transformer import (
    run_splice_transformer, SpliceTransformerInput, SpliceTransformerConfig
)

# Multiple sequences
inputs = SpliceTransformerInput(
    target_seqs=[seq1, seq2, seq3],
    left_contexts=[ctx1_l, ctx2_l, ctx3_l],
    right_contexts=[ctx1_r, ctx2_r, ctx3_r]
)
config = SpliceTransformerConfig()

result = run_splice_transformer(inputs, config)
print(f"Processed {len(result.prediction)} sequences")
```

## Best Practices & Gotchas

**Sequence preparation:**

1. **Consistent lengths**: All `left_contexts` must be same length (`context_length`), all `right_contexts` same length.

2. **Same batch size**: Number of `target_seqs`, `left_contexts`, and `right_contexts` must match.

3. **Sufficient context**: Use 4000bp context for best results (default).

**Interpretation:**

1. **[GT-AG rule](https://en.wikipedia.org/wiki/RNA_splicing#Introns)**: Canonical splice sites follow GT...AG pattern -- check predictions at these positions.

2. **Multi-channel analysis**: Don't just look at donor/acceptor -- check tissue channels for specificity.

3. **Sum to 1**: Channels 0, 1, 2 roughly sum to 1 (softmax over splice types).

**Common mistakes:**

1. **Wrong context length**: Left/right context must exactly match `context_length` parameter.

2. **Misaligned sequences**: Target and context must be from same genomic locus.

3. **Ignoring tissue context**: Model predicts tissue-specific splicing -- use the right tissue channel for your analysis.

4. **Non-human sequences**: Model is trained on human data only -- predictions for other species may be unreliable.

**Computational considerations:**

1. **GPU recommended**: CPU inference is very slow.

2. **Memory**: Long sequences require significant GPU memory.

3. **Batching**: Process sequences in batches for efficiency.

## References

**Primary publication:**
- Chen, J. et al. (2024). "Predicting RNA splicing from DNA sequence using Pangolin." *Nature Communications*. DOI: [10.1038/s41467-024-53088-6](https://doi.org/10.1038/s41467-024-53088-6)

**Implementation:**
- GitHub: [https://github.com/ShenLab-Genomics/SpliceTransformer](https://github.com/ShenLab-Genomics/SpliceTransformer)

## Related Tools

**Tools often used together:**
- `borzoi`: Gene expression and splicing prediction (broader scope, can validate findings)
- `enformer`: Gene expression prediction (complementary to splicing analysis)

**Alternative tools:**
- `alphagenome`: Multi-output genomic predictions including splicing
- SpliceAI: Alternative splice site prediction model (not wrapped in this toolkit)
