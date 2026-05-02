<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/borzoi"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Borzoi

## Overview

Borzoi is a deep learning model that predicts gene expression and regulatory activity from a 524,288 bp DNA sequence. As the successor to Enformer, Borzoi uses dilated residual convolutional blocks combined with attention to achieve a 2.7x longer context window (524 kb vs. 196 kb), higher output resolution (6,144 bins), and improved accuracy, particularly for RNA-seq coverage prediction.

- **Tool keys**: `borzoi-prediction` (single replicate), `borzoi-ensemble` (all 4 replicates)
- **Model context**: 524,288 bp (fixed length, ~262 kb in each direction from center)
- **Output resolution**: 6,144 bins x 32 bp per bin (~197 kb output span)
- **Species heads**: `human`, `mouse`
- **Replicates**: 4 independently trained models
- **GPU required**: Yes

## Background

Gene regulation involves interactions across a wide range of genomic distances. While most promoter-proximal elements act within a few kilobases, enhancers can regulate genes from distances exceeding 100 kb, and [topologically associating domains](https://en.wikipedia.org/wiki/Topologically_associating_domain) (TADs) organize chromatin contacts across megabase scales. Capturing these long-range interactions requires models with sufficient context windows.

Borzoi was trained to predict RNA-seq coverage directly from DNA sequence, rather than the processed experimental tracks used by Enformer. This training objective enables:
- **Quantitative RNA-seq profiles**: Direct prediction of read coverage across gene bodies, capturing splicing patterns, alternative [TSS](https://en.wikipedia.org/wiki/Transcription_start_site) usage, and transcript isoform ratios
- **Broader regulatory context**: The 524 kb window captures most enhancer-promoter interactions and some TAD-level organization
- **Higher output resolution**: 6,144 bins at 32 bp resolution provide finer spatial detail than Enformer

The model also predicts [CAGE](https://en.wikipedia.org/wiki/Cap_analysis_of_gene_expression), [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq), [ATAC-seq](https://en.wikipedia.org/wiki/ATAC-seq), and [histone modification](https://en.wikipedia.org/wiki/Histone_modification) tracks, making it a general-purpose regulatory genomics predictor with improved long-range accuracy.

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

1. **Input**: One or more 524,288 bp model-context sequences, or longer source sequences with `target_ranges`
2. **One-hot encoding**: The sequence is converted to a 4-channel (A, C, G, T) representation; N bases are encoded as all zeros
3. **Convolutional stem + dilated residual blocks**: Initial layers downsample and process the sequence using dilated convolutions for efficient long-range feature extraction
4. **Attention layers**: Transformer-style attention captures dependencies across the full context window
5. **Species-specific output heads**: Separate prediction heads for human and mouse tracks
6. **Output**: A prediction matrix of shape `[num_tracks, 6144]` for single replicate, or `[4, num_tracks, 6144]` for ensemble

## Input Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sequences` | `List[str]` | Yes | DNA source sequence(s). Without `target_ranges`, each sequence must be exactly 524,288 bp. With `target_ranges`, sequences may be longer and must contain enough context for a full Borzoi input window. Only A, T, C, G, N are allowed. |
| `target_ranges` | `List[SequenceTargetRange]` | No | Sequence-relative target range(s) to keep inside the model output window. Each range has `start` (0-based inclusive) and `end` (0-based exclusive). A single range is auto-wrapped into a list. |

If `target_ranges` is omitted, the provided sequence is the exact model input. If `target_ranges` is provided, the tool extracts a 524,288 bp model input window from each source sequence. The returned result reports where that model input window and the model output window landed in source-sequence coordinates.

Target-range extraction is start-aligned, not midpoint-centered: when possible, bin 0 starts at `target_ranges[i].start`. Near the right edge, the context window shifts left so the full target range still fits.

## Configuration

### Single Replicate (`BorzoiConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_tracks` | `List[int]` | **required** | Track indices to extract from model output |
| `species` | `"human"` or `"mouse"` | `"human"` | Species model to use |
| `replicate` | `"0"`, `"1"`, `"2"`, or `"3"` | `"0"` | Which replicate model to run |
| `avg_output_tracks` | `bool` | `True` | Whether to average selected tracks into one output |
| `batch_size` | `int` | `1` | Number of sequences to process simultaneously on GPU |
| `device` | `str` | `"cuda"` | Inference device |
| `verbose` | `bool` | `False` | Log status messages |

### Ensemble (`BorzoiEnsembleConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_tracks` | `List[int]` | **required** | Track indices to extract from model output |
| `species` | `"human"` or `"mouse"` | `"human"` | Species model to use |
| `avg_output_tracks` | `bool` | `True` | Whether to average selected tracks into one output |
| `batch_size` | `int` | `1` | Number of sequences to process simultaneously on GPU |
| `device` | `str` | `"cuda"` | Inference device |
| `verbose` | `bool` | `False` | Log status messages |

### Parameter Guides

**`avg_output_tracks`**: When `True` (default), all requested tracks are averaged into a single output track per bin. This is useful for combining related tracks (e.g., multiple CAGE tracks) into a composite signal. Set to `False` to get individual predictions per track.

Human Borzoi runs use FlashAttention-backed checkpoints automatically. Mouse Borzoi runs use the non-FlashAttention checkpoints because those are the available mouse checkpoints.

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
| `results` | `List[BorzoiPredictionResult]` | Per-sequence prediction results |
| `output_tracks` | `List[int]` | Track indices used |
| `species` | `str` | Species head used |
| `replicate` | `str` | Replicate ID used |
| `avg_output_tracks` | `bool` | Whether averaging was applied |

Each `BorzoiPredictionResult` contains `sequence`, `sequence_length`, coordinate metadata, and `prediction` with shape `[num_tracks, 6144]` (or `[1, 6144]` if `avg_output_tracks=True`).

Coordinate metadata is relative to the input `sequence`:

| Field | Meaning |
|-------|---------|
| `context_start`, `context_end` | The 524,288 bp model input window that was sent to Borzoi |
| `output_start`, `output_end` | The source-sequence span covered by Borzoi output bins |
| `output_resolution` | Base pairs per output bin (`32`) |
| `target_start`, `target_end` | The requested `target_ranges` entry, if one was provided |

### `BorzoiEnsembleOutput` (ensemble)

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[BorzoiEnsemblePredictionResult]` | Per-sequence ensemble prediction results |
| `output_tracks` | `List[int]` | Track indices used |
| `species` | `str` | Species head used |
| `avg_output_tracks` | `bool` | Whether averaging was applied |
| `num_replicates` | `int` | Always 4 |

Each `BorzoiEnsemblePredictionResult` contains the same coordinate metadata as `BorzoiPredictionResult`, plus `predictions` with shape `[4, num_tracks, 6144]`.

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

**Spatial interpretation**: Bin index maps to source-sequence coordinates as:
- Source coordinate = `output_start + bin_index * output_resolution`
- With exact-window inputs, `output_start` is 163,840 and the center of the input (position 262,144) corresponds to bin 3,072

**Ensemble uncertainty:**
- Compute standard deviation across 4 replicate predictions per bin
- High variance across replicates indicates lower model confidence at that position
- Consistent predictions across all replicates suggest robust signal

## Quick Start Examples

**Single replicate prediction:**
```python
from proto_tools.tools.sequence_scoring.borzoi import (
    BorzoiInput,
    BorzoiConfig,
    run_borzoi,
)

# 524,288 bp sequence (use real genomic sequence in practice)
sequence = "ATCG" * 131072  # 524,288 bp

inputs = BorzoiInput(sequences=[sequence])
config = BorzoiConfig(
    output_tracks=[0, 1, 2],
    species="human",
    replicate="0",
    avg_output_tracks=True,
)

result = run_borzoi(inputs, config)
prediction = result.results[0].prediction
print(f"Output shape: {len(prediction)} x {len(prediction[0])}")
# Output shape: 1 x 6144 (averaged tracks)
```

**Ensemble prediction for uncertainty estimation:**
```python
from proto_tools.tools.sequence_scoring.borzoi import (
    BorzoiInput,
    BorzoiEnsembleConfig,
    run_borzoi_ensemble,
)

inputs = BorzoiInput(sequences=[sequence])
config = BorzoiEnsembleConfig(
    output_tracks=[0, 1, 2],
    species="human",
    avg_output_tracks=False,
)

result = run_borzoi_ensemble(inputs, config)
predictions = result.results[0].predictions
print(f"Ensemble shape: {len(predictions)} x "
      f"{len(predictions[0])} x {len(predictions[0][0])}")
# Ensemble shape: 4 x 3 x 6144

# Compute per-bin variance across replicates
import statistics
bin_idx = 3072  # Center bin
for track in range(3):
    vals = [predictions[rep][track][bin_idx] for rep in range(4)]
    print(f"Track {track}, bin {bin_idx}: "
          f"mean={statistics.mean(vals):.3f}, std={statistics.stdev(vals):.3f}")
```

**Mouse prediction:**
```python
config = BorzoiConfig(
    output_tracks=[0, 1, 2],
    species="mouse",
)

result = run_borzoi(BorzoiInput(sequences=[sequence]), config)
```

**Export results:**
```python
result = run_borzoi(inputs, config)
result.export("borzoi_output", file_format="json")
```

## Best Practices & Gotchas

- **Exact-window inputs must be 524,288 bp**: If you do not pass `target_ranges`, each input sequence is treated as the exact Borzoi model input and must be exactly 524,288 bp.
- **Use `target_ranges` for longer source sequences**: When your sequence includes extra flanking context, provide one sequence-relative target range per sequence. Borzoi will extract the fixed model input window and report where the context and output windows landed.
- **Center your region of interest**: Like Enformer, predictions are most informative near the center of the input window. Place your target gene or variant at position ~262,144.
- **Species selects the checkpoint family**: Human runs use FlashAttention-backed checkpoints automatically; mouse runs use the available non-FlashAttention checkpoints.
- **Ensemble is 4x slower**: Each replicate runs a full forward pass. Use single replicate for iteration, ensemble for final analysis.
- **Track averaging collapses dimensions**: With `avg_output_tracks=True`, each result's `prediction` shape is `[1, 6144]` regardless of how many tracks you requested. Set to `False` if you need per-track resolution.
- **GPU memory**: A single replicate inference uses ~6-8 GB of GPU memory due to the longer context window. Ensemble runs replicates sequentially, so peak memory is the same as single replicate.
- **N characters**: Accepted but degrade prediction quality. Minimize N content in your input.

## References

- Linder, J., Srivastava, D., Yuan, H. et al. "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation." *Nature Genetics* 57, 587-597 (2025). DOI: [10.1038/s41588-024-02053-6](https://doi.org/10.1038/s41588-024-02053-6)
- GitHub: [calico/borzoi](https://github.com/calico/borzoi)
- Model weights: [HuggingFace: johahi/borzoi-models](https://huggingface.co/collections/johahi/borzoi-models)

## Related Tools

**Used together:**
- `enformer-prediction`: Cross-validate regulatory predictions between Enformer and Borzoi
- `evo2-sample`: Generate candidate sequences, then score them with Borzoi

**Alternatives:**
- `enformer-prediction`: Shorter context (196 kb), faster inference. Use when long-range context is not needed.
- `alphagenome`: Google DeepMind's next-generation genomics model
- `splice-transformer`: Specialized for splice site prediction rather than general regulatory activity
