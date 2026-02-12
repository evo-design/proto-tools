# Borzoi Prediction

## Overview

Borzoi is a deep learning model that predicts gene expression and regulatory activity from a 524,288 bp DNA sequence. As the successor to Enformer, Borzoi uses dilated residual convolutional blocks combined with attention to achieve a 2.7x longer context window (524 kb vs. 196 kb), higher output resolution (6,144 bins), and improved accuracy, particularly for RNA-seq coverage prediction.

- **Tool keys**: `borzoi-prediction` (single replicate), `borzoi-ensemble` (all 4 replicates)
- **Model context**: 524,288 bp (fixed length, ~262 kb in each direction from center)
- **Output resolution**: 6,144 bins x 128 bp per bin (~786 kb output span)
- **Species heads**: `human`, `mouse`
- **Replicates**: 4 independently trained models
- **GPU required**: Yes

## When to Use This Tool

**Use Borzoi when you need to:**
- Predict RNA-seq coverage profiles from DNA sequence (Borzoi's primary training objective)
- Model long-range regulatory interactions beyond 100 kb (e.g., distal enhancers, TAD boundaries)
- Score variant effects on gene expression with higher resolution than Enformer
- Evaluate synthetic regulatory element designs in a broad genomic context
- Obtain ensemble predictions for uncertainty estimation across 4 replicates

**When NOT to use Borzoi:**
- **Shorter context is sufficient** (<100 kb interactions): `enformer-prediction` is faster and may suffice
- **Splice site effects**: Use `splice-transformer` for splice-specific predictions
- **Protein structure**: Use `protenix-prediction` or `esmfold-prediction`
- **Sequence generation**: Use generative models like `evo2-sample`

## Biological Background

Gene regulation involves interactions across a wide range of genomic distances. While most promoter-proximal elements act within a few kilobases, enhancers can regulate genes from distances exceeding 100 kb, and topologically associating domains (TADs) organize chromatin contacts across megabase scales. Capturing these long-range interactions requires models with sufficient context windows.

Borzoi was trained to predict RNA-seq coverage directly from DNA sequence, rather than the processed experimental tracks used by Enformer. This training objective enables:
- **Quantitative RNA-seq profiles**: Direct prediction of read coverage across gene bodies, capturing splicing patterns, alternative TSS usage, and transcript isoform ratios
- **Broader regulatory context**: The 524 kb window captures most enhancer-promoter interactions and some TAD-level organization
- **Higher output resolution**: 6,144 bins at 128 bp resolution (vs. 896 bins in Enformer) provide finer spatial detail

The model also predicts CAGE, DNase-seq, ATAC-seq, and histone modification tracks, making it a general-purpose regulatory genomics predictor with improved long-range accuracy.

## Tool Catalog

| Tool Key | Function | Description |
|----------|----------|-------------|
| `borzoi-prediction` | `run_borzoi` | Single replicate prediction. Fast, suitable for screening and optimization loops. |
| `borzoi-ensemble` | `run_borzoi_ensemble` | All 4 replicates. Returns per-replicate predictions for uncertainty quantification. |

Use the single-replicate tool for iterative design and optimization. Use the ensemble tool when you need confidence estimates or are making final assessments.

## Execution Modes

Borzoi requires GPU acceleration for inference.

| Mode | Backend | Setup |
|------|---------|-------|
| **Local venv** | Local CUDA GPU | Requires NVIDIA GPU with CUDA |

For ensemble mode, all 4 replicates run sequentially.

## How It Works

1. **Input**: A fixed-length 524,288 bp DNA sequence centered on the region of interest
2. **One-hot encoding**: The sequence is converted to a 4-channel (A, C, G, T) representation; N bases are encoded as all zeros
3. **Convolutional stem + dilated residual blocks**: Initial layers downsample and process the sequence using dilated convolutions for efficient long-range feature extraction
4. **Attention layers**: Transformer-style attention captures dependencies across the full context window
5. **Species-specific output heads**: Separate prediction heads for human and mouse tracks
6. **Output**: A prediction matrix of shape `[num_tracks, 6144]` for single replicate, or `[4, num_tracks, 6144]` for ensemble

## Input Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sequence` | `str` | Yes | DNA sequence, must be exactly 524,288 bp. Only characters A, T, C, G, N allowed. |

## Configuration

### Single Replicate (`BorzoiConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_tracks` | `List[int]` | **required** | Track indices to extract from model output |
| `species` | `"human"` or `"mouse"` | `"human"` | Species model to use |
| `replicate` | `"0"`, `"1"`, `"2"`, or `"3"` | `"0"` | Which replicate model to run |
| `avg_output_tracks` | `bool` | `True` | Whether to average selected tracks into one output |
| `use_flash_attn` | `bool` | `True` | Use FlashAttention models (faster, human only) |
| `device` | `str` | `"cuda"` | Inference device |
| `verbose` | `bool` | `False` | Log status messages |

### Ensemble (`BorzoiEnsembleConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_tracks` | `List[int]` | **required** | Track indices to extract from model output |
| `species` | `"human"` or `"mouse"` | `"human"` | Species model to use |
| `avg_output_tracks` | `bool` | `True` | Whether to average selected tracks into one output |
| `use_flash_attn` | `bool` | `True` | Use FlashAttention models (faster, human only) |
| `device` | `str` | `"cuda"` | Inference device |
| `verbose` | `bool` | `False` | Log status messages |

### Parameter Guides

**`avg_output_tracks`**: When `True` (default), all requested tracks are averaged into a single output track per bin. This is useful for combining related tracks (e.g., multiple CAGE tracks) into a composite signal. Set to `False` to get individual predictions per track.

**`use_flash_attn`**: FlashAttention provides significant speedup for attention computation. It is enabled by default for human models. Mouse models do NOT support FlashAttention -- setting `use_flash_attn=True` with `species="mouse"` raises a validation error.

**`replicate`**: Borzoi has 4 independently trained replicates (0-3). Each produces slightly different predictions. Replicate "0" is the default. For production analysis, use the ensemble tool to get all 4.

### Sweep Priorities

When using Borzoi in optimization loops:

1. **`output_tracks`**: Most critical. Select tracks matching your biological objective.
2. **`species`**: Must match the organism of your target sequence.
3. **`avg_output_tracks`**: Set to `True` for a single optimization signal, `False` for multi-objective optimization across tracks.

## Output Specification

### `BorzoiOutput` (single replicate)

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `str` | The input DNA sequence |
| `sequence_length` | `int` | Always 524,288 |
| `prediction` | `List[List[float]]` | Shape `[num_tracks, 6144]` (or `[1, 6144]` if `avg_output_tracks=True`) |
| `output_tracks` | `List[int]` | Track indices used |
| `species` | `str` | Species head used |
| `replicate` | `str` | Replicate ID used |
| `avg_output_tracks` | `bool` | Whether averaging was applied |

### `BorzoiEnsembleOutput` (ensemble)

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `str` | The input DNA sequence |
| `sequence_length` | `int` | Always 524,288 |
| `predictions` | `List[List[List[float]]]` | Shape `[4, num_tracks, 6144]` |
| `output_tracks` | `List[int]` | Track indices used |
| `species` | `str` | Species head used |
| `avg_output_tracks` | `bool` | Whether averaging was applied |
| `num_replicates` | `int` | Always 4 |

## Interpreting Results

Borzoi outputs are in **log(1 + x) transformed counts** space. Higher values indicate stronger predicted signal.

**For variant effect analysis:**
- Compare predictions between reference and alternate allele sequences
- Compute log2 fold changes at bins overlapping the variant and its regulatory neighborhood
- Use ensemble predictions to assess whether the effect is consistent across replicates

**For RNA-seq profile prediction:**
- The output captures read coverage patterns including exon/intron structure
- Peaks in CAGE tracks indicate predicted transcription start sites
- Track averaging (`avg_output_tracks=True`) gives a composite signal across related tracks

**Ensemble uncertainty:**
- Compute standard deviation across 4 replicate predictions per bin
- High variance across replicates indicates lower model confidence at that position
- Consistent predictions across all replicates suggest robust signal

## Quick Start Examples

**Single replicate prediction:**
```python
from bio_programming_tools.tools.sequence_scoring.borzoi import (
    BorzoiInput,
    BorzoiConfig,
    run_borzoi,
)

# 524,288 bp sequence (use real genomic sequence in practice)
sequence = "ATCG" * 131072  # 524,288 bp

inputs = BorzoiInput(sequence=sequence)
config = BorzoiConfig(
    output_tracks=[0, 1, 2],
    species="human",
    replicate="0",
    avg_output_tracks=True,
)

result = run_borzoi(inputs, config)
print(f"Output shape: {len(result.prediction)} x {len(result.prediction[0])}")
# Output shape: 1 x 6144 (averaged tracks)
```

**Ensemble prediction for uncertainty estimation:**
```python
from bio_programming_tools.tools.sequence_scoring.borzoi import (
    BorzoiInput,
    BorzoiEnsembleConfig,
    run_borzoi_ensemble,
)

inputs = BorzoiInput(sequence=sequence)
config = BorzoiEnsembleConfig(
    output_tracks=[0, 1, 2],
    species="human",
    avg_output_tracks=False,
)

result = run_borzoi_ensemble(inputs, config)
print(f"Ensemble shape: {len(result.predictions)} x "
      f"{len(result.predictions[0])} x {len(result.predictions[0][0])}")
# Ensemble shape: 4 x 3 x 6144

# Compute per-bin variance across replicates
import statistics
bin_idx = 3072  # Center bin
for track in range(3):
    vals = [result.predictions[rep][track][bin_idx] for rep in range(4)]
    print(f"Track {track}, bin {bin_idx}: "
          f"mean={statistics.mean(vals):.3f}, std={statistics.stdev(vals):.3f}")
```

**Mouse prediction (FlashAttention disabled):**
```python
config = BorzoiConfig(
    output_tracks=[0, 1, 2],
    species="mouse",
    use_flash_attn=False,  # Required for mouse models
)

result = run_borzoi(BorzoiInput(sequence=sequence), config)
```

**Export results:**
```python
result = run_borzoi(inputs, config)
result.export("borzoi_output", file_format="json")
```

## Best Practices & Gotchas

- **Exact length required**: Input must be exactly 524,288 bp. Shorter or longer sequences are rejected.
- **Center your region of interest**: Like Enformer, predictions are most informative near the center of the input window. Place your target gene or variant at position ~262,144.
- **Mouse requires `use_flash_attn=False`**: Mouse checkpoints were not trained with FlashAttention. Setting `species="mouse"` with `use_flash_attn=True` raises a `ValueError`.
- **Ensemble is 4x slower**: Each replicate runs a full forward pass. Use single replicate for iteration, ensemble for final analysis.
- **Track averaging collapses dimensions**: With `avg_output_tracks=True`, `prediction` shape is `[1, 6144]` regardless of how many tracks you requested. Set to `False` if you need per-track resolution.
- **GPU memory**: A single replicate inference uses ~6-8 GB of GPU memory due to the longer context window. Ensemble runs replicates sequentially, so peak memory is the same as single replicate.
- **N characters**: Accepted but degrade prediction quality. Minimize N content in your input.

## References

- Linder, J., Srivastava, D., Yuan, H. et al. "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation." *Nature Genetics* 57, 587-597 (2025). DOI: [10.1038/s41588-024-02053-6](https://doi.org/10.1038/s41588-024-02053-6)
- GitHub: [calico/borzoi](https://github.com/calico/borzoi)
- Model weights: [HuggingFace — johahi/borzoi-models](https://huggingface.co/collections/johahi/borzoi-models)

## Related Tools

**Used together:**
- `enformer-prediction` — Cross-validate regulatory predictions between Enformer and Borzoi
- `evo2-sample` — Generate candidate sequences, then score them with Borzoi

**Alternatives:**
- `enformer-prediction` — Shorter context (196 kb), faster inference. Use when long-range context is not needed.
- `alphagenome` — Google DeepMind's next-generation genomics model
- `splice-transformer` — Specialized for splice site prediction rather than general regulatory activity
