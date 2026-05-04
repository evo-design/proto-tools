<a href="https://bio-pro.mintlify.app/tools/gene-annotation/promoter-calculator"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Salis Lab Promoter Calculator

## Overview
The [Salis Lab Promoter Calculator](https://github.com/barricklab/promoter-calculator) is a 346-parameter biophysical + machine-learning model that predicts [sigma70 promoter](https://en.wikipedia.org/wiki/Sigma_factor) strength in *Escherichia coli*. It scans both strands of input DNA for canonical promoter elements -- the [-35 hexamer](https://en.wikipedia.org/wiki/Pribnow_box#%E2%88%9235_element), spacer, [-10 hexamer (Pribnow box)](https://en.wikipedia.org/wiki/Pribnow_box), [UP element](https://en.wikipedia.org/wiki/Promoter_(genetics)#UP_element), and discriminator -- and returns binding free energy (`dG_total`, kcal/mol) and transcription initiation rate (`Tx_rate`, arbitrary units) per candidate.

## Background

**What does this tool measure/predict?**
Per-TSS (transcription start site) [Gibbs free energy](https://en.wikipedia.org/wiki/Gibbs_free_energy) of RNA polymerase holoenzyme binding (`dG_total`) and a calibrated transcription initiation rate (`Tx_rate`).

**Why is this important?**
sigma70 is the housekeeping [sigma factor](https://en.wikipedia.org/wiki/Sigma_factor) of *E. coli*. Quantitative promoter strength prediction is foundational for designing tunable expression cassettes, avoiding cryptic promoters in engineered DNA, and understanding native gene regulation.

**Scientific foundation:**
The model combines a free-energy biophysical layer (per-element contributions: -10/-35 boxes, spacer length and composition, UP element, discriminator) with a regression learned on a [massively parallel reporter assay](https://en.wikipedia.org/wiki/Reporter_gene#Massively_parallel_reporter_assays) and validated across 22,132 bacterial promoters with diverse sequences.

## How It Works

**Method overview:**
Every candidate TSS on both strands is scored independently by the biophysical layer; the ML layer maps the resulting dG to an absolute Tx rate.

**Key assumptions:**
- Input is double-stranded DNA in the `A/C/G/T` alphabet
- Promoters use the *E. coli* sigma70 holoenzyme
- Sequences are linear unless `circular=True`

**Limitations:**
- Models *E. coli* sigma70 only -- not alternative sigma factors (sigmaS, sigma32, sigma54, ...) or other organisms' housekeeping sigmas
- Sequences shorter than the calculator's scan window return no predictions; pad with neutral context if needed
- Predicts in vitro transcription rate, not in vivo expression (which also depends on RBS strength, copy number, growth state)
- Does not model transcription factor repression, attenuation, or anti-sigma factors

**Computational requirements:**
- **Hardware:** CPU only
- **Runtime:** Seconds for short sequences (~100-1000 bp); scales with sequence length and candidate TSS count
- **Parallelism:** Single-threaded by default; `threads > 1` parallelises the internal TSS scan

## Important Parameters

**Input parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `list[str]` | *Required* | DNA sequences to scan for sigma70 promoters |
| `sequence_ids` | `list[str] \| None` | `None` | Optional sequence identifiers (defaults to `seq_0`, `seq_1`, ...) |

**Configuration parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threads` | `int` | `1` | CPU threads for the internal TSS scan (must be `>= 1`) |
| `verbosity` | `int` | `0` | Calculator verbosity (0 = quiet) |
| `circular` | `bool` | `False` | Treat input sequences as circular DNA (e.g. plasmids) |

**Parameters to prioritize:**
1. **`circular`**: Set to `True` for plasmids and bacterial chromosomes -- otherwise promoters spanning the origin are silently missed.
2. **`threads`**: No effect on results, only on wall-clock runtime. Raise for whole-plasmid scans or large batches.

---

**Output specification:**

```python
# Return type: PromoterCalculatorOutput
{
    "results": [
        {
            "sequence_id": str,
            "predictions": [
                {
                    "tss_name": str,           # e.g. "Fwd123" or "Rev456"
                    "tss": int,                # TSS position
                    "strand": str,             # "+" or "-"
                    "dG_total": float,         # Predicted binding free energy (kcal/mol)
                    "Tx_rate": float,          # Predicted transcription rate (a.u.)
                    "promoter_sequence": str,  # DNA spanning the predicted promoter
                    "length": int,             # Length of the promoter sequence
                    "UP_position": [int, int],     # UP element bounds
                    "hex35_position": [int, int],  # -35 hexamer bounds
                    "spacer_position": [int, int], # spacer bounds
                    "hex10_position": [int, int],  # -10 hexamer bounds
                    "disc_position": [int, int],   # discriminator bounds
                },
                ...
            ]
        },
        ...
    ]
}
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[PromoterCalculatorSequenceResult]` | Per-sequence predictions |
| `predictions` | `list[PromoterPrediction]` | All predicted promoters across both strands |
| `tss_name` / `strand` | `str` | `Fwd...`/`+` or `Rev...`/`-` |
| `dG_total` | `float` | Binding free energy in kcal/mol (more negative = stronger) |
| `Tx_rate` | `float` | Transcription initiation rate (higher = stronger) |

**Convenience properties:**
- `PromoterCalculatorOutput.num_sequences_with_promoter`
- `PromoterCalculatorSequenceResult.num_promoters`
- `PromoterCalculatorSequenceResult.has_promoter`

**Supported export formats:** `csv`, `json`

**Interpreting `dG_total` and `Tx_rate`:**
- `dG_total < -3.0 kcal/mol` -- strong promoter
- `dG_total` between `-3.0` and `-1.5` -- moderate
- `dG_total > -1.5` -- weak / unlikely
- `Tx_rate > 10000` -- strong; `3000-10000` -- moderate; `< 3000` -- weak

Results are not pre-filtered by strength -- apply your own threshold downstream.

## Best Practices & Gotchas

- **Plasmids**: always set `circular=True`; otherwise promoters spanning the linearised origin are silently missed.
- **Short sequences**: pad with neutral context (e.g. `A`-rich flanking) if the input is below the calculator's scan-window minimum.
- **Cross-organism use**: the model is trained on *E. coli*. Treat predictions in other organisms as relative rankings only.
- **`Tx_rate` vs expression**: `Tx_rate` is *transcription initiation*, not protein output -- combine with an RBS calculator and copy-number information for end-to-end expression prediction.
- **Threads**: bump only for long sequences or large batches; doesn't change results.

## Quick Start Examples

**Example 1: Predict promoters in the lacUV5 promoter**
```python
from proto_tools.tools.gene_annotation.promoter_calculator import (
    PromoterCalculatorConfig,
    PromoterCalculatorInput,
    run_promoter_calculator,
)

# lacUV5 promoter padded with 20 nt of neutral context. The calculator needs
# ~20 nt of flanking sequence on each side to score the promoter elements;
# shorter padding returns no predictions (boundary effect).
seq = "A" * 20 + "AAAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACC" + "A" * 20

result = run_promoter_calculator(
    PromoterCalculatorInput(sequences=[seq], sequence_ids=["lacUV5"]),
    PromoterCalculatorConfig(),
)

for seq_result in result.results:
    print(f"{seq_result.sequence_id}: {seq_result.num_promoters} promoter(s)")
    for pred in seq_result.predictions:
        print(
            f"  {pred.tss_name} strand={pred.strand} TSS={pred.tss} "
            f"dG={pred.dG_total:.2f} Tx_rate={pred.Tx_rate:.1f}"
        )
```

**Example 2: Rank an Anderson-collection library by predicted strength**
```python
# Three Anderson-collection synthetic E. coli sigma70 promoters, padded with
# 30 nt of A on each side. J23119 is the canonical "strong" reference.
variants = {
    "J23119_strong": "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC",
    "J23100_strong": "TTGACGGCTAGCTCAGTCCTAGGTACAGTGCTAGC",
    "J23113_weak":   "CTGATGGCTAGCTCAGTCCTAGGGATTATGCTAGC",
}

result = run_promoter_calculator(
    PromoterCalculatorInput(
        sequences=["A" * 30 + s + "A" * 30 for s in variants.values()],
        sequence_ids=list(variants.keys()),
    ),
    PromoterCalculatorConfig(threads=4),
)

for seq_result in result.results:
    fwd = [p for p in seq_result.predictions if p.strand == "+"]
    if not fwd:
        print(f"{seq_result.sequence_id}: no forward-strand promoter")
        continue
    best = max(fwd, key=lambda p: p.Tx_rate)
    print(f"{seq_result.sequence_id}: best Tx_rate={best.Tx_rate:.1f} dG={best.dG_total:.2f}")

result.export("promoter_library_predictions", file_format="csv")
```

**Example 3: Scan a circular plasmid**
```python
result = run_promoter_calculator(
    PromoterCalculatorInput(sequences=[plasmid_seq], sequence_ids=["pUC19_variant"]),
    PromoterCalculatorConfig(threads=8, circular=True),
)
print(f"{result.num_sequences_with_promoter} of {len(result.results)} have promoters")
```

## References

- LaFleur, T. L., Hossain, A., & Salis, H. M. (2022). "Automated Model-Predictive Design of Synthetic Promoters to Control Transcriptional Profiles in Bacteria." *Nature Communications* 13, 5159. [DOI: 10.1038/s41467-022-32829-5](https://doi.org/10.1038/s41467-022-32829-5)
- Source (Barrick Lab fork used here): [https://github.com/barricklab/promoter-calculator](https://github.com/barricklab/promoter-calculator)
- Original Salis Lab implementation: [https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator](https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator)

## Related Tools

- **`minced`**, **`blast-search`**, **`pyhmmer-hmmsearch`**: complementary annotation of the same DNA region
- **RBS Calculator** (separate tool): pair with promoter strength for full transcription + translation modelling
