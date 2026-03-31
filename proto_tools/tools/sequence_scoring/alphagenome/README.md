<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/alphagenome"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AlphaGenome

## Overview

AlphaGenome is Google DeepMind's multi-task genomic foundation model that predicts diverse regulatory signals from DNA sequence. It takes long genomic context windows (up to 1 Mb) and outputs spatial prediction tracks for [RNA-seq](https://en.wikipedia.org/wiki/RNA-Seq), [ATAC-seq](https://en.wikipedia.org/wiki/ATAC-seq), [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq), [CAGE](https://en.wikipedia.org/wiki/Cap_analysis_of_gene_expression), [ChIP-seq](https://en.wikipedia.org/wiki/ChIP_sequencing), splice sites, and 3D contact maps. This wrapper provides six batched tools covering interval prediction, variant-effect prediction, raw sequence prediction, and three scoring modes (variant, interval, ISM).

This implementation uses **local GPU inference** with Hugging Face weights through an isolated standalone venv runtime (`ToolInstance("alphagenome")`).

## When to Use This Tool

**Primary use cases:**
- Predicting regulatory activity (chromatin accessibility, gene expression, TF binding) across a genomic region
- Assessing the functional impact of SNVs and indels on regulatory signals
- In-silico mutagenesis scanning to identify regulatory-sensitive positions
- Comparing reference vs. alternate allele predictions for variant prioritization
- Predicting splice site usage and junction effects of genomic variants

**When NOT to use this tool:**
- For protein sequence tasks: use ESM2, ESM3, or ProGen2
- For DNA sequence generation: use Evo2
- For lightweight expression-only prediction: use Enformer (faster, simpler output)
- For mammalian expression prediction at 32 bp resolution with RNA-seq coverage: use Borzoi
- For splice-only analysis: use SpliceTransformer

## Biological Background

Gene regulation is controlled by a complex interplay of [cis-regulatory elements](https://en.wikipedia.org/wiki/Cis-regulatory_element) (promoters, enhancers, silencers) and [chromatin](https://en.wikipedia.org/wiki/Chromatin) state. Different experimental assays measure different aspects of this regulatory landscape:

- **RNA-seq** measures transcript abundance, reflecting gene expression levels
- **ATAC-seq / DNase-seq** measure chromatin accessibility, indicating active regulatory regions
- **CAGE** captures transcription start sites at single-nucleotide resolution
- **ChIP-seq** maps protein-DNA interactions (histone modifications, transcription factor binding)
- **Splice site assays** quantify alternative splicing patterns
- **Contact maps ([Hi-C](https://en.wikipedia.org/wiki/Hi-C_(genomic_analysis_technique)))** capture 3D chromatin organization

AlphaGenome jointly predicts all of these signals from DNA sequence alone, leveraging long-range context (up to 1 Mb) to capture distal regulatory effects. This multi-task architecture allows it to model the relationships between regulatory layers -- for example, how a variant that disrupts a [CTCF](https://en.wikipedia.org/wiki/CTCF) binding site might alter both chromatin accessibility and 3D contact structure.

## Tool Catalog

| Tool Key | Description | Input | Output |
|----------|-------------|-------|--------|
| `alphagenome-predict-intervals` | Predict regulatory signals for batched genomic regions | List of intervals (auto-wraps single) | List of prediction matrices |
| `alphagenome-predict-variants` | Predict signals for batched variant effects | List of variants (auto-wraps single) | List of prediction matrices + variant metadata |
| `alphagenome-predict-sequences` | Predict signals from raw DNA sequence(s) | List of DNA sequences (auto-wraps single string) | List of prediction matrices |
| `alphagenome-score-variants` | Score batched variant effects using variant scorers | List of variants (auto-wraps single) | List of tidy score records |
| `alphagenome-score-intervals` | Score batched genomic intervals with interval scorers | List of intervals (auto-wraps single) | List of tidy score records |
| `alphagenome-score-ism-variants-batch` | Batched in-silico mutagenesis | List of ISM requests (auto-wraps single) | List of tidy score records |

All tools accept either a single input item (auto-wrapped into a list) or an explicit list for batch processing. All tools use `cacheable=True` on `@tool()` for per-item caching.

**Prediction tools** return raw spatial tracks (matrices) suitable for visualization and custom downstream analysis. **Scoring tools** return tidy tabular records (one row per scorer-track-gene combination) suitable for ranking and filtering.

## How It Works

1. **Input preparation**: The tool accepts either genomic coordinates (chromosome + interval) or raw DNA sequence(s). If the interval length does not match one of AlphaGenome's supported context lengths (16 KB, 100 KB, 500 KB, 1 MB), it is automatically resized by the inference layer. Raw sequence inputs must already match one of these lengths.
2. **Model inference**: The DNA sequence is passed through AlphaGenome's multi-task transformer architecture, which produces predictions for all requested output types in a single forward pass. Inference runs in an isolated venv on GPU.
3. **Output formatting**: Prediction tools return the raw output matrices keyed by output type. Scoring tools apply recommended scorer functions that summarize the predictions into interpretable per-variant or per-interval scores.
4. **Variant scoring**: For variant-effect tools, the model runs inference on both the reference and alternate sequences, and scorers compute the difference (e.g., log-fold-change in predicted signal).

Coordinates use **0-based indexing with exclusive end** ([BED](https://en.wikipedia.org/wiki/BED_(file_format))-style).

## Input Parameters

### Shared Inner Models

**`AlphaGenomeInterval`** — Base input for interval-based tools.

| Field | Type | Description |
|-------|------|-------------|
| `chromosome` | `str` | Chromosome identifier, e.g. `"chr1"` |
| `interval_start` | `int` | Interval start (0-based, inclusive) |
| `interval_end` | `int` | Interval end (0-based, exclusive) |

**`AlphaGenomeVariant`** — Extends `AlphaGenomeInterval` with variant alleles.

| Field | Type | Description |
|-------|------|-------------|
| `chromosome` | `str` | Chromosome identifier, e.g. `"chr1"` |
| `interval_start` | `int` | Interval start (0-based, inclusive) |
| `interval_end` | `int` | Interval end (0-based, exclusive) |
| `variant_position` | `int` | Variant genomic position (0-based, must be within interval) |
| `reference_bases` | `str` | Reference allele (A/C/G/T/N string) |
| `alternate_bases` | `str` | Alternate allele (A/C/G/T/N string) |

**`AlphaGenomeISM`** — Extends `AlphaGenomeInterval` with ISM sub-interval and optional variant context.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chromosome` | `str` | *(required)* | Chromosome identifier, e.g. `"chr1"` |
| `interval_start` | `int` | *(required)* | Interval start (0-based, inclusive) |
| `interval_end` | `int` | *(required)* | Interval end (0-based, exclusive) |
| `ism_interval_start` | `int` | *(required)* | ISM sub-interval start (0-based, inclusive, must be within interval) |
| `ism_interval_end` | `int` | *(required)* | ISM sub-interval end (0-based, exclusive, must be within interval) |
| `variant_position` | `Optional[int]` | `None` | Optional existing variant position for ISM context (0-based) |
| `reference_bases` | `Optional[str]` | `None` | Optional existing variant reference allele |
| `alternate_bases` | `Optional[str]` | `None` | Optional existing variant alternate allele |

### Interval Tools (`alphagenome-predict-intervals`, `alphagenome-score-intervals`)

| Field | Type | Description |
|-------|------|-------------|
| `intervals` | `List[AlphaGenomeInterval]` | Genomic intervals (single item auto-wrapped) |

### Variant Tools (`alphagenome-predict-variants`, `alphagenome-score-variants`)

| Field | Type | Description |
|-------|------|-------------|
| `variants` | `List[AlphaGenomeVariant]` | Variants with interval context (single item auto-wrapped) |

### Sequences Tool (`alphagenome-predict-sequences`)

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Raw DNA sequence strings (single string auto-wrapped) |

### ISM Tool (`alphagenome-score-ism-variants-batch`)

| Field | Type | Description |
|-------|------|-------------|
| `requests` | `List[AlphaGenomeISM]` | ISM requests (single item auto-wrapped) |

## Configuration

### Prediction Configs (`AlphaGenomePredictIntervalsConfig` / `PredictVariantsConfig` / `PredictSequencesConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_version` | `str` | `"all_folds"` | HF model version |
| `requested_outputs` | `List[str]` | required | Output types to predict (see table below) |
| `organism` | `"human" \| "mouse"` | `"human"` | Organism setting |
| `ontology_terms` | `Optional[List[str]]` | `None` | Optional ontology term filters |
| `device` | `str` | `"cuda"` | Inference device |

**Available output types:** `ATAC`, `CAGE`, `DNASE`, `RNA_SEQ`, `CHIP_HISTONE`, `CHIP_TF`, `SPLICE_SITES`, `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`, `CONTACT_MAPS`, `PROCAP`

### Variant Scoring Config (`AlphaGenomeScoreVariantsConfig` / `AlphaGenomeScoreISMConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_version` | `str` | `"all_folds"` | HF model version |
| `variant_scorers` | `Optional[List[str]]` | `None` | Scorer names (`None` = all recommended) |
| `organism` | `"human" \| "mouse"` | `"human"` | Organism setting |
| `device` | `str` | `"cuda"` | Inference device |

**Available variant scorers:** `ATAC`, `CONTACT_MAPS`, `DNASE`, `CHIP_TF`, `CHIP_HISTONE`, `CAGE`, `PROCAP`, `RNA_SEQ`, `RNA_SEQ_ACTIVE`, `SPLICE_SITES`, `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`, `POLYADENYLATION`, `ATAC_ACTIVE`, `DNASE_ACTIVE`, `CHIP_TF_ACTIVE`, `CHIP_HISTONE_ACTIVE`, `CAGE_ACTIVE`, `PROCAP_ACTIVE`

### Interval Scoring Config (`AlphaGenomeScoreIntervalsConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_version` | `str` | `"all_folds"` | HF model version |
| `interval_scorers` | `Optional[List[str]]` | `None` | Scorer names (`None` = all recommended) |
| `organism` | `"human" \| "mouse"` | `"human"` | Organism setting |
| `device` | `str` | `"cuda"` | Inference device |

**Available interval scorers:** `RNA_SEQ`

## Output Specification

### Prediction Outputs (`AlphaGenomePredictOutput`)

| Field | Type | Description |
|-------|------|-------------|
| `chromosome` | `str` | Chromosome identifier |
| `interval_start` | `int` | Interval start (0-based) |
| `interval_end` | `int` | Interval end (0-based, exclusive) |
| `requested_outputs` | `List[str]` | Output types that were requested |
| `result` | `Dict[str, Any]` | Serialized prediction payload (spatial tracks) |
| `variant` | `Optional[Dict]` | Variant metadata (variant prediction only) |

Export formats: `json`, `npy`

### Scoring Outputs (`AlphaGenomeScoreOutput`)

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[Dict[str, Any]]` | Tidy score records (one per scorer-track-gene combination) |

Each score record contains keys such as `variant_id`, `scored_interval`, `gene_id`, `gene_name`, `output_type`, `variant_scorer` or `interval_scorer`, `track_name`, `raw_score`.

Export formats: `json`, `csv`

### Batched Output Wrappers

All batched tools return an output with a `results` field (`List` of per-item outputs) that supports `__len__`, `__getitem__`, and `__iter__` for convenient iteration:

```python
result = run_alphagenome_predict_intervals(inputs, config)
len(result)      # number of items
result[0]        # first item
for item in result:
    print(item.chromosome)
```

## Interpreting Results

**Prediction outputs** are spatial tracks (matrices) where each row corresponds to a genomic position and each column to a specific assay track. Higher values indicate stronger predicted signal (e.g., higher chromatin accessibility, higher expression). These tracks can be visualized in genome browsers or used for custom quantitative analysis.

**Scoring outputs** provide pre-computed summary statistics:
- **Variant scores** quantify the predicted effect of a variant on each regulatory signal. Larger absolute scores indicate stronger predicted effects. Positive/negative direction depends on the scorer type.
- **ISM scores** provide a per-position, per-mutation breakdown of predicted regulatory effects. Useful for identifying the most functionally sensitive positions within a region.
- **Interval scores** summarize predicted activity levels across a region.

**Scorers with `_ACTIVE` suffix** (e.g., `RNA_SEQ_ACTIVE`, `ATAC_ACTIVE`) restrict scoring to tracks/genes that are predicted to be active, reducing noise from inactive loci.

## Quick Start Examples

**Example 1: Predict RNA-seq and ATAC for a genomic interval**
```python
from proto_tools import (
    AlphaGenomeInterval,
    AlphaGenomePredictIntervalsConfig,
    AlphaGenomePredictIntervalsInput,
    run_alphagenome_predict_intervals,
)

inputs = AlphaGenomePredictIntervalsInput(
    intervals=AlphaGenomeInterval(
        chromosome="chr19",
        interval_start=10_587_331,
        interval_end=11_635_907,
    ),
)
config = AlphaGenomePredictIntervalsConfig(
    requested_outputs=["RNA_SEQ", "ATAC"],
    organism="human",
)

result = run_alphagenome_predict_intervals(inputs, config)
print(result[0].requested_outputs)  # ['RNA_SEQ', 'ATAC']
```

**Example 2: Score a variant's regulatory impact**
```python
from proto_tools import (
    AlphaGenomeScoreVariantsConfig,
    AlphaGenomeScoreVariantsInput,
    AlphaGenomeVariant,
    run_alphagenome_score_variants,
)

inputs = AlphaGenomeScoreVariantsInput(
    variants=AlphaGenomeVariant(
        chromosome="chr19",
        interval_start=10_587_331,
        interval_end=11_635_907,
        variant_position=11_000_000,
        reference_bases="A",
        alternate_bases="G",
    ),
)
config = AlphaGenomeScoreVariantsConfig(
    variant_scorers=["RNA_SEQ", "ATAC", "CAGE"],
)

result = run_alphagenome_score_variants(inputs, config)
for score in result[0].scores[:3]:
    print(f"{score['variant_scorer']} | {score['track_name']}: {score['raw_score']:.4f}")
```

**Example 3: Predict from raw DNA sequence(s)**
```python
from proto_tools import (
    AlphaGenomePredictSequencesConfig,
    AlphaGenomePredictSequencesInput,
    run_alphagenome_predict_sequences,
)

inputs = AlphaGenomePredictSequencesInput(
    sequences=["ACGT" * 4096],  # 16,384 bp
)
config = AlphaGenomePredictSequencesConfig(
    requested_outputs=["DNASE", "CHIP_TF"],
)

result = run_alphagenome_predict_sequences(inputs, config)
first_output = result[0]
```

**Example 4: In-silico mutagenesis over a promoter region**
```python
from proto_tools import (
    AlphaGenomeISM,
    AlphaGenomeScoreISMConfig,
    AlphaGenomeScoreISMInput,
    run_alphagenome_score_ism_variants_batch,
)

inputs = AlphaGenomeScoreISMInput(
    requests=AlphaGenomeISM(
        chromosome="chr19",
        interval_start=10_587_331,
        interval_end=11_635_907,
        ism_interval_start=11_000_000,
        ism_interval_end=11_000_050,
    ),
)
config = AlphaGenomeScoreISMConfig(variant_scorers=["CAGE"])

result = run_alphagenome_score_ism_variants_batch(inputs, config)
print(f"ISM scores: {len(result[0].scores)} records")
```

## Best Practices & Gotchas

- **Context length matters**: AlphaGenome supports 4 context lengths (16 KB to 1 MB). The 1 MB window captures the most distal regulatory effects. Shorter windows are faster but miss long-range interactions. Non-matching interval lengths are automatically resized.
- **First run is slow**: Model weights are downloaded from Hugging Face on first use. Subsequent runs load from the local cache.
- **GPU required**: AlphaGenome requires a CUDA GPU.
- **Auto-wrapping**: All batched tools accept a single item (dict, model instance, or string) which is automatically wrapped into a one-element list. This avoids the need for explicit list syntax for single-input use cases.
- **Per-item caching**: All tools use `cacheable=True` on `@tool()` which caches results per individual input item. Duplicate items in a batch are computed only once.
- **Variant position validation**: The variant position must fall within `[interval_start, interval_end)`. The ISM sub-interval must be fully contained within the main interval.
- **Scorer selection**: Passing `None` for `variant_scorers` or `interval_scorers` uses all recommended scorers, which provides comprehensive coverage but is slower. Specify individual scorers when you only need specific assay types.
- **ISM sub-intervals**: Keep ISM windows small (tens to low hundreds of bp). Each position generates 3 alternate allele predictions, so computation scales linearly with window size.
- **Ontology filtering**: Use `ontology_terms` to restrict predictions to specific cell types or tissues when you know the biological context, reducing output size and noise.
- **Export**: Prediction outputs support `json` and `npy` formats. Scoring outputs support `json` and `csv`. Use `result.export("path", format="csv")` for downstream analysis in pandas.

## References

**Primary publication:**
- Google DeepMind (2025). "AlphaGenome: multi-task foundation model for the genome." Preprint.
- Summary: Multi-task genomic model predicting RNA-seq, ATAC-seq, DNase-seq, CAGE, ChIP-seq, splice sites, and 3D contact maps from DNA sequence with up to 1 Mb context.

**Implementation:**
- GitHub: [https://github.com/google-deepmind/alphagenome_research](https://github.com/google-deepmind/alphagenome_research)
- Hugging Face: [https://huggingface.co/google/alphagenome](https://huggingface.co/google/alphagenome)

## Related Tools

**Used together:**
- `enformer`, `borzoi` — compare expression predictions across models
- `splice_transformer` — detailed splice-specific analysis alongside AlphaGenome's splice predictions
- `evo2` — generate candidate DNA sequences, then score with AlphaGenome

**Alternatives:**
- `enformer` — lighter-weight expression prediction (200 KB context, Basenji-style architecture)
- `borzoi` — mammalian expression prediction at 32 bp resolution with RNA-seq coverage tracks
- `splice_transformer` — dedicated splice site prediction
