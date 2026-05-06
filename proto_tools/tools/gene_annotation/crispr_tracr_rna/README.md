<a href="https://bio-pro.mintlify.app/tools/gene-annotation/crispr-tracr-rna"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# CRISPRtracrRNA

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

[CRISPRtracrRNA](https://github.com/BackofenLab/CRISPRtracrRNA) is a multi-evidence pipeline from the Backofen Lab that predicts [tracrRNA](https://en.wikipedia.org/wiki/Trans-activating_crRNA) sequences in nucleotide [CRISPR](https://en.wikipedia.org/wiki/CRISPR) loci. It combines covariance-model search, CRISPR array detection, cas-effector cassette detection, anti-repeat similarity, RNA-RNA interaction prediction, and terminator detection into a weighted multi-evidence ranking score.

## Background

**What does this tool measure/predict?**
Per-locus tracrRNA candidates with full evidence: candidate position and sequence, CRISPR array context, anti-repeat similarity to the array's repeat consensus, predicted RNA-RNA interaction with the CRISPR repeat, transcription terminator location, distance to the nearest cas-effector cassette, and a single weighted multi-evidence ranking score.

**Why is this important?**
For a [CRISPR-Cas9](https://en.wikipedia.org/wiki/CRISPR#Class_2) system to function, three components must be present: the [Cas9](https://en.wikipedia.org/wiki/Cas9) effector, the [crRNA](https://en.wikipedia.org/wiki/CRISPR#crRNA) (from the array), and the tracrRNA. Detecting tracrRNA is essential for confirming a complete Type II locus, designing [single-guide RNAs](https://en.wikipedia.org/wiki/Guide_RNA) by fusing crRNA + tracrRNA, validating computationally generated CRISPR systems, and discovering novel Cas9/Cas12 systems in metagenomic data.

**Scientific foundation:**
Mitrofanov et al. (2022) — multi-evidence integration is more sensitive than covariance-model search alone, especially for divergent tracrRNA families and novel CRISPR-Cas systems.

## Tools

### CRISPRtracrRNA Prediction (`crispr-tracr-rna`)

Predict tracrRNA sequences from nucleotide CRISPR loci.

Uses the CRISPRtracrRNA tool from the Backofen Lab to predict tracrRNA
sequences associated with CRISPR loci. This is used as a Stage 3 filter
in the Cas9 filtering pipeline to confirm that candidate sequences
contain functional tracrRNA binding sites.

## How It Works

The wrapper has two modes:

### `run_type="complete_run"` (default) — multi-evidence pipeline

| Stage | Tool | Contribution |
|-------|------|--------------|
| 1. CRISPR array detection | [CRISPRidentify](https://github.com/BackofenLab/CRISPRidentify) (ML) | Locates CRISPR arrays + repeat consensus |
| 2. Cas-effector cassette detection | [CRISPRcasIdentifier](https://github.com/BackofenLab/CRISPRcasIdentifier) (HMM + ML) | Finds Cas9 / Cas12 cassettes |
| 3. tracrRNA candidate detection | [Infernal](http://eddylab.org/infernal/) `cmsearch` against curated covariance models | Initial tracrRNA hits |
| 4. Anti-repeat similarity | [fasta36](https://github.com/wrpearson/fasta36), vmatch, clustalo, blast | Aligns candidates to array repeat |
| 5. RNA-RNA interaction | [IntaRNA](https://github.com/BackofenLab/IntaRNA) | Predicts anti-repeat ↔ repeat duplex |
| 6. Terminator detection | erpin (bundled in upstream) | Locates poly-U transcription terminator |
| 7. Multi-evidence ranking | upstream `candidate_ranking.py` | Weighted sum across all evidence |

The ten ranking weights (`weight_*` config fields) control how each piece of evidence contributes to the final ranking score. Defaults are upstream's documented values; sweep them only when you have a specific reason.

### `run_type="model_run"` — fast covariance-model scan only

Runs only stage 3 (Infernal `cmsearch`). Skips array detection, cas detection, anti-repeat similarity, IntaRNA interaction, terminator detection, and multi-evidence ranking. Produces tracrRNA candidates with covariance-model E-values but none of the validation evidence. Use this for high-throughput pre-filtering when you'll do downstream validation yourself.

**Key assumptions:**
- Input is double-stranded DNA in the `A/C/G/T` alphabet (the wrapper strips other characters).
- Sequences include enough flanking context around the CRISPR array (~5 kb either side) for the multi-evidence pipeline to find adjacent cas cassettes and terminators.
- For `model_type="II"`, the input is expected to contain Type II Cas9 systems. Use `model_type="all"` (and optionally `perform_type_v_anti_repeat_analysis=True`) to also screen for Type V (Cas12).

**Limitations:**
- `complete_run` requires the CRISPRcasIdentifier ML/HMM models, which `setup.sh` downloads from Google Drive on first install. The download is rate-limited by Google; on `setup.sh` failure, retry or download manually per upstream's README.
- Covariance-model sensitivity falls off for tracrRNAs highly divergent from known families.
- Type I and Type III CRISPR systems do not use tracrRNA — `complete_run` will return None-heavy candidates for those loci even when arrays are detected.

**Computational requirements:**
- **Hardware:** CPU only; no GPU.
- **Runtime:** seconds per sequence in `model_run`; tens of seconds to a minute per sequence in `complete_run`, dominated by Infernal `cmsearch` and IntaRNA.
- **Parallelism:** the `num_workers` config splits sequences into independent batches (each in its own working directory to avoid file contention). Defaults to 1 or `$SLURM_CPUS_PER_TASK` if set.

## Important Parameters

**Input:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `list[str]` | *required* | Nucleotide sequence(s); each should contain a CRISPR locus with flanking context |
| `sequence_ids` | `list[str] \| None` | `None` | Optional sequence IDs (defaults to `seq_0`, `seq_1`, ...) |

**Configuration (mirrors upstream's argparse):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `Literal["II", "all"]` | `"II"` | Type II only (faster) vs all-types (broader, slower) |
| `run_type` | `Literal["complete_run", "model_run"]` | `"complete_run"` | Full pipeline vs fast covariance-model scan only |
| `num_workers` | `int \| None` | `None` | Parallel batches; `None` → 1 or `$SLURM_CPUS_PER_TASK` |
| `anti_repeat_similarity_threshold` | `float` | `0.7` | Minimum anti-repeat ↔ repeat similarity (0-1) |
| `anti_repeat_coverage_threshold` | `float` | `0.6` | Minimum anti-repeat alignment coverage (0-1) |
| `perform_type_v_anti_repeat_analysis` | `bool` | `False` | Type V (Cas12) anti-repeat search; combine with `model_type="all"` |

**Ranking weights (advanced — only effective in `complete_run`):**

| Parameter | Default | What it weighs |
|-----------|---------|----------------|
| `weight_crispr_array_score` | `0.5` | CRISPRidentify array confidence |
| `weight_anti_repeat_sim` | `0.5` | Anti-repeat sequence similarity |
| `weight_anti_repeat_coverage` | `0.5` | Anti-repeat alignment coverage |
| `weight_anti_sim_coverage` | `0.5` | similarity x coverage product |
| `weight_interaction_score` | `0.6` | IntaRNA interaction energy |
| `weight_model_hit_score` | `0.9` | Covariance-model tail hit |
| `weight_terminator_hit_score` | `0.9` | erpin terminator presence/score |
| `weight_consistency_orientation` | `0.1` | Repeat / anti-repeat orientation consistency |
| `weight_consistency_anti_repeat_tail` | `0.1` | Anti-repeat ↔ tail positional consistency |
| `weight_consistency_tail_terminator` | `0.1` | Tail ↔ terminator positional consistency |

---

**Output specification:**

`CrisprTracrRNAOutput.results` is one entry per input sequence; each entry is a `CrisprTracrRNASequenceResult` carrying every candidate hit upstream produced for that accession, sorted by `score` descending (`candidates[0]` is the top-ranked hit).

```python
# Return type: CrisprTracrRNAOutput
{
    "results": [
        CrisprTracrRNASequenceResult(
            sequence_id="seq1",
            candidates=[
                CrisprTracrRNAPrediction(
                    sequence_id="seq1",

            # Identity (set in complete_run)
            accession_number=...,
            crispr_array_index=..., crispr_array_category=...,

            # CRISPR array — from CRISPRidentify
            crispr_array_score=..., crispr_array_start=..., crispr_array_end=...,
            crispr_array_repeat_consensus=..., crispr_array_orientation=...,
            crispr_orientation_flag=...,

            # Anti-repeat — from fasta36/vmatch/clustalo/blast
            anti_repeat_sequence=...,
            anti_repeat_start=..., anti_repeat_end=..., anti_repeat_direction=...,
            anti_repeat_relative_location=..., anti_repeat_distance_from_crispr_array=...,
            anti_repeat_similarity=..., anti_repeat_coverage=...,
            anti_repeat_similarity_coverage_multiplication=...,
            anti_repeat_upstream=...,

            # tracrRNA — from covariance-model search
            tracr_rna_taken_flag=..., tracr_rna_tail_sequence=...,
            tracr_rna_global_window_sequence=..., tracr_rna_sequence=...,

            # Interaction — from IntaRNA
            intarna_anti_repeat_interaction_interval=...,
            intarna_anti_repeat_interaction=...,
            interaction_energy=...,
            poli_u_signal_coordinates=...,

            # Terminator — from erpin
            terminator_all_locations=..., terminator_all_scores=...,
            best_terminator_location=..., best_terminator_score=...,
            terminator_presence_flag=...,

            # Tail — covariance-model tail hit
            tail_model_hit_location=..., tail_model_hit_score=..., tail_presence_flag=...,

            # Cas — from CRISPRcasIdentifier
            closest_corresponding_cas_interval=..., distance_to_cas=...,

                    # Multi-evidence ranking
                    score=...,  # weighted sum across all evidence
                ),
                # Additional candidates for the same accession (lower-scoring), if any.
            ],
        ),
        ...  # one CrisprTracrRNASequenceResult per input sequence
    ]
}
```

All `CrisprTracrRNAPrediction` fields are `Optional`. In `model_run` mode only the cmsearch-only columns (`start`, `end`, `e_value` / `best_e_value`, `hit_sequence`) are populated; in `complete_run` the full evidence stack can populate.

**Convenience properties:**
- `CrisprTracrRNAPrediction.has_tracr` — True if `tracr_rna_sequence`, `anti_repeat_start`, or `hit_sequence` is set.
- `CrisprTracrRNASequenceResult.top_candidate` — first candidate (top-ranked) or `None`.
- `CrisprTracrRNASequenceResult.has_tracr` — True if any candidate has `has_tracr`.
- `CrisprTracrRNAOutput.num_with_tracr` — count of input sequences for which a tracrRNA was detected.

**Supported export formats:** `csv`, `json`.

**Interpreting key fields:**
- `score` — the headline number; higher is better. Compare across candidates within a run, not across runs with different weights.
- `interaction_energy` — IntaRNA energy in kcal/mol; more negative = stronger duplex. Below `-5.0` is strong, `-5.0` to `-2.0` is moderate, above `-2.0` is weak.
- `best_terminator_score` — erpin score; higher = stronger terminator match.
- `distance_to_cas` — bp distance from tracrRNA to the nearest cas cassette; near-zero values support a complete Type II/V locus.

## Best Practices & Gotchas

- **Run MinCED first** to confirm CRISPR array presence on a per-locus basis, then run CRISPRtracrRNA on the broader region (~5 kb flank).
- **`complete_run` requires the Google-Drive-hosted CRISPRcasIdentifier models.** `setup.sh` handles the download; if it fails, retry or follow the upstream README to install the two `tar.gz` archives manually into the CRISPRcasIdentifier directory.
- **Don't expect tracrRNA hits in Type I/III systems** — those don't use tracrRNA.
- **For high-throughput pre-filtering, use `model_run`** to triage candidates by covariance-model E-value, then re-run `complete_run` on the top candidates for full evidence.
- **Don't tune individual ranking weights without a target metric.** They interact; sweep coarsely with a representative held-out set or stick with upstream's defaults.
- **The `perform_type_v_anti_repeat_analysis` flag is bool but upstream uses `argparse(type=bool)`.** Our wrapper sidesteps the footgun by only emitting the flag when our config is True; setting it to False is equivalent to omitting it.

## Quick Start Examples

```python
from proto_tools.tools.gene_annotation.crispr_tracr_rna import (
    CrisprTracrRNAConfig,
    CrisprTracrRNAInput,
    run_crispr_tracr_rna,
)

# Full pipeline — multi-evidence ranking
inputs = CrisprTracrRNAInput(sequences=["ATCG..." * 1000], sequence_ids=["my_locus"])
result = run_crispr_tracr_rna(inputs, CrisprTracrRNAConfig(model_type="II"))
for seq_result in result.results:
    top = seq_result.top_candidate
    if top is not None and top.has_tracr:
        print(f"{seq_result.sequence_id}: score={top.score}, "
              f"tracrRNA={top.tracr_rna_sequence}, "
              f"interaction={top.interaction_energy} kcal/mol "
              f"({len(seq_result.candidates)} candidates)")
```

```python
# Fast scan — covariance-model search only
result = run_crispr_tracr_rna(
    inputs,
    CrisprTracrRNAConfig(model_type="II", run_type="model_run"),
)
```

```python
# All Cas types (Type II + V) with tuned anti-repeat thresholds
result = run_crispr_tracr_rna(
    inputs,
    CrisprTracrRNAConfig(
        model_type="all",
        anti_repeat_similarity_threshold=0.6,
        anti_repeat_coverage_threshold=0.5,
        perform_type_v_anti_repeat_analysis=True,
    ),
)
```

## Related Tools

- **`minced`** — detects CRISPR arrays in nucleotide sequence. Run before CRISPRtracrRNA on a per-locus basis to confirm array presence and bound the input region (~5 kb either side).
- **`pyhmmer-hmmsearch`** — search adjacent ORFs for Cas effector domains (Cas9, Cas12) when validating the multi-evidence ranking.
- **`prodigal`** — predict ORFs near the locus to locate cas operons.

## References

- Mitrofanov, A., Alkhnbashi, O.S., Shmakov, S.A., Makarova, K.S., Koonin, E.V. & Backofen, R. (2022). "CRISPRtracrRNA: robust approach for CRISPR tracrRNA detection." *Bioinformatics* 38(Supplement_2):ii42-ii48. [DOI: 10.1093/bioinformatics/btac466](https://doi.org/10.1093/bioinformatics/btac466)
- Upstream: [https://github.com/BackofenLab/CRISPRtracrRNA](https://github.com/BackofenLab/CRISPRtracrRNA)
- CRISPRidentify: [https://github.com/BackofenLab/CRISPRidentify](https://github.com/BackofenLab/CRISPRidentify)
- CRISPRcasIdentifier: [https://github.com/BackofenLab/CRISPRcasIdentifier](https://github.com/BackofenLab/CRISPRcasIdentifier)
- Infernal: [http://eddylab.org/infernal/](http://eddylab.org/infernal/)
- IntaRNA: [https://github.com/BackofenLab/IntaRNA](https://github.com/BackofenLab/IntaRNA)
