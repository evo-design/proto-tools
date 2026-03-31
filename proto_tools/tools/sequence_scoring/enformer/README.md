<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/enformer"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Enformer

## Overview

Enformer is a transformer-based deep learning model that predicts gene expression and chromatin accessibility directly from a 196,608 bp DNA sequence. It uses [self-attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)) to capture long-range regulatory interactions up to ~100 kb away, enabling accurate prediction of how distal enhancers, silencers, and other regulatory elements influence gene expression.

- **Tool key**: `enformer-prediction`
- **Model context**: 196,608 bp (fixed length, ~98 kb in each direction from center)
- **Output resolution**: 896 bins x 128 bp per bin
- **Species heads**: `human` (5,313 tracks), `mouse` (1,643 tracks)
- **GPU required**: Yes

## When to Use This Tool

**Use Enformer when you need to:**
- Predict gene expression levels from DNA sequence alone
- Assess variant effects on regulatory activity (compare reference vs. alternate allele predictions)
- Evaluate chromatin accessibility (DNase, ATAC) or histone modification signals at a locus
- Score synthetic promoter or enhancer designs for predicted expression output
- Understand how sequence changes affect transcription factor binding profiles

**When NOT to use Enformer:**
- **Longer context needed** (>100 kb regulatory interactions): Use `borzoi-prediction` (524 kb context)
- **RNA-seq coverage prediction**: Use `borzoi-prediction` (trained directly on RNA-seq)
- **Splice site effects**: Use `splice-transformer` for splice-specific predictions
- **Protein structure**: Use `protenix-prediction` or `esmfold-prediction`
- **Sequence generation**: Use generative models like `evo2-sample`

## Biological Background

Gene expression is controlled by a complex interplay of promoters, [enhancers](https://en.wikipedia.org/wiki/Enhancer_(genetics)), [silencers](https://en.wikipedia.org/wiki/Silencer_(genetics)), [insulators](https://en.wikipedia.org/wiki/Insulator_(genetics)), and chromatin state. These regulatory elements can act over distances of tens to hundreds of kilobases. Traditional motif-based models capture local sequence features but miss long-range interactions.

Enformer addresses this by processing 196,608 bp of genomic context through a transformer architecture. The model was trained on 5,313 human and 1,643 mouse experimental tracks from [ENCODE](https://en.wikipedia.org/wiki/ENCODE) and Roadmap Epigenomics, covering:
- **Gene expression**: [CAGE](https://en.wikipedia.org/wiki/Cap_analysis_of_gene_expression) (promoter-level transcription initiation)
- **Chromatin accessibility**: [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq), [ATAC-seq](https://en.wikipedia.org/wiki/ATAC-seq)
- **Histone modifications**: H3K4me3, H3K27ac, H3K36me3, H3K27me3, and others
- **Transcription factor binding**: [ChIP-seq](https://en.wikipedia.org/wiki/ChIP_sequencing) for hundreds of TFs

Each output track corresponds to a specific assay in a specific cell type or tissue. The 896 output bins (128 bp each) tile the central ~114 kb of the input window, providing spatial resolution of predicted regulatory activity.

## Execution Modes

Enformer requires GPU acceleration for inference.

| Mode | Backend | Setup |
|------|---------|-------|
| **Local venv** | Local CUDA GPU | Requires NVIDIA GPU with CUDA |

## How It Works

1. **Input**: A fixed-length 196,608 bp DNA sequence centered on the region of interest
2. **One-hot encoding**: The sequence is converted to a 4-channel (A, C, G, T) binary representation; N bases are encoded as all zeros
3. **Convolutional stem**: Initial layers downsample the sequence by 2x, producing a 98,304-length feature map
4. **Transformer tower**: 11 transformer blocks with attention capture long-range dependencies across the sequence
5. **Pointwise heads**: Species-specific output heads project features to track predictions
6. **Output**: A [896, num_tracks] matrix of predicted signal values for the requested tracks

The central 114,688 bp of the input maps to the 896 output bins. The flanking sequence on each side (~41 kb) provides context but does not have direct output bins.

## Input Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sequence` | `str` | Yes | DNA sequence, must be exactly 196,608 bp. Only characters A, T, C, G, N allowed. |

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_tracks` | `List[int]` | **required** | Track indices to extract from the model output |
| `species` | `"human"` or `"mouse"` | `"human"` | Species output head to use |
| `device` | `str` | `"cuda"` | Inference device (`"cuda"` or `"cpu"`) |
| `verbose` | `bool` | `False` | Whether to log status messages during execution |

### Track Selection Guide

Enformer has thousands of output tracks. You must specify which tracks to extract via `output_tracks`. Common track categories for the human head:

| Track Range | Assay Type | Example Use Case |
|-------------|-----------|-----------------|
| 0-674 | CAGE (gene expression) | Promoter activity across tissues |
| 675-4674 | DNase / ATAC (accessibility) | Open chromatin scoring |
| 4675-5312 | Histone ChIP-seq | Enhancer marks (H3K27ac), repressive marks (H3K27me3) |

Consult the [Enformer track table](https://github.com/google-deepmind/deepmind-research/tree/master/enformer) for the full mapping of track index to assay and cell type.

### Sweep Priorities

When using Enformer in optimization loops, prioritize sweeping:

1. **`output_tracks`**: The most important parameter. Choose tracks matching your biological objective (e.g., CAGE tracks for expression, DNase tracks for accessibility).
2. **`species`**: Use `"human"` for human regulatory elements, `"mouse"` for mouse. Cross-species predictions are not supported.

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `str` | The input DNA sequence that was scored |
| `sequence_length` | `int` | Always 196,608 |
| `prediction` | `List[List[float]]` | Predicted signal matrix with shape `[896, num_tracks]` |
| `output_tracks` | `List[int]` | Track indices that were extracted |
| `species` | `str` | Species head used (`"human"` or `"mouse"`) |

The `prediction` matrix is indexed as `prediction[bin][track]`, where:
- `bin` ranges from 0 to 895 (896 spatial bins, each covering 128 bp)
- `track` ranges from 0 to `len(output_tracks) - 1`

## Interpreting Results

Enformer outputs are in **log(1 + x) transformed counts** space, matching the training targets. Higher values indicate stronger predicted signal.

**For variant effect analysis**, compare predictions between reference and alternate sequences:
- Compute the log2 fold change: `log2(alt_pred / ref_pred)` per bin and track
- Focus on bins near the variant position (center of the window)
- A fold change > 0.5 or < -0.5 in relevant tracks suggests a functional effect

**For promoter/enhancer design**, maximize the predicted signal at the central bins for expression-related tracks (CAGE) or accessibility tracks (DNase/ATAC).

**Spatial interpretation**: Bin index maps to genomic position as:
- Genomic offset from start = `bin_index * 128`
- The center of the input (position 98,304) corresponds approximately to bin 448

## Quick Start Examples

**Basic prediction with CAGE tracks:**
```python
from proto_tools.tools.sequence_scoring.enformer import (
    EnformerInput,
    EnformerConfig,
    run_enformer,
)

# 196,608 bp sequence (use real genomic sequence in practice)
sequence = "ATCG" * 49152  # 196,608 bp

inputs = EnformerInput(sequence=sequence)
config = EnformerConfig(
    output_tracks=[0, 1, 2],  # First 3 CAGE tracks
    species="human",
)

result = run_enformer(inputs, config)
print(f"Output shape: {len(result.prediction)} x {len(result.prediction[0])}")
# Output shape: 896 x 3
```

**Variant effect prediction (reference vs. alternate):**
```python
import copy

# Assume ref_seq is a 196,608 bp reference sequence
ref_seq = "A" * 98304 + "C" + "T" * 98303  # Simplified example

# Create alternate sequence with a single nucleotide change at center
alt_seq = ref_seq[:98304] + "G" + ref_seq[98305:]

config = EnformerConfig(output_tracks=[0, 1, 2], species="human")

ref_result = run_enformer(EnformerInput(sequence=ref_seq), config)
alt_result = run_enformer(EnformerInput(sequence=alt_seq), config)

# Compare central bins (around position 98,304 -> bin ~448)
import math
center_bin = 448
for track_idx in range(3):
    ref_val = ref_result.prediction[center_bin][track_idx]
    alt_val = alt_result.prediction[center_bin][track_idx]
    if ref_val > 0 and alt_val > 0:
        lfc = math.log2(alt_val / ref_val)
        print(f"Track {track_idx}: log2FC = {lfc:.3f}")
```

**Mouse species head:**
```python
config = EnformerConfig(
    output_tracks=[0, 1, 2, 3, 4],
    species="mouse",
)

result = run_enformer(EnformerInput(sequence=sequence), config)
print(f"Mouse prediction shape: {len(result.prediction)} x {len(result.prediction[0])}")
```

**Export results:**
```python
result = run_enformer(inputs, config)
result.export("enformer_output", file_format="json")
```

## Best Practices & Gotchas

- **Exact length required**: The input sequence must be exactly 196,608 bp. Shorter or longer sequences will be rejected. Pad with N characters if necessary, but prefer extracting the correct genomic window.
- **Center your region of interest**: Enformer's predictions are most accurate near the center of the window. Place your gene, variant, or regulatory element at position ~98,304.
- **N characters reduce accuracy**: While N is accepted, regions with many Ns (e.g., assembly gaps) will have unreliable predictions. Minimize N content.
- **Track indices are 0-based**: `output_tracks=[0]` extracts the first track, not a track labeled "0" in external databases.
- **Predictions are not absolute expression values**: Outputs are in model-specific units. Use them for relative comparisons (e.g., variant vs. reference, design A vs. design B), not as direct RPM or TPM estimates.
- **GPU memory**: A single inference uses ~4-6 GB of GPU memory. Batch processing is not natively supported; run one sequence at a time.

## References

- Avsec, Z., Agarwal, V., Visentin, D. et al. "Effective gene expression prediction from sequence by integrating long-range interactions." *Nature Methods* 18, 1196-1203 (2021). DOI: [10.1038/s41592-021-01252-x](https://doi.org/10.1038/s41592-021-01252-x)
- GitHub: [google-deepmind/deepmind-research/enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer)

## Related Tools

**Used together:**
- `borzoi-prediction` — Compare Enformer and Borzoi predictions for cross-validation of regulatory effects
- `evo2-sample` — Generate candidate sequences, then score them with Enformer

**Alternatives:**
- `borzoi-prediction` — Longer context (524 kb), higher resolution, trained on RNA-seq. Preferred for most new analyses.
- `alphagenome` — Google DeepMind's next-generation genomics model
- `splice-transformer` — Specialized for splice site prediction rather than general regulatory activity
