<a href="https://bio-pro.mintlify.app/tools/database-retrieval/alphamissense"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AlphaMissense

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

`alphamissense-fetch` retrieves per-residue, per-substitution AlphaMissense pathogenicity scores for human proteins by UniProt accession. It returns the full set of single amino acid substitution predictions (typically ~7,000-20,000 per protein, one row per (position, alt_aa) pair), each with a pathogenicity score in [0, 1] and a discrete classification (`likely_benign` / `ambiguous` / `likely_pathogenic`). This is a CPU-only tool that issues a single HTTPS GET against the AlphaFold Protein Structure Database, which hosts the AlphaMissense CSVs.

## Background

**What does this tool measure/predict?**
[AlphaMissense](https://github.com/google-deepmind/alphamissense) is a deep-learning model from Google DeepMind that predicts the pathogenicity of every possible missense substitution across the human proteome. For each canonical UniProt sequence, AlphaMissense scores all 19 possible alternate amino acids at every position, yielding a dense missense saturation map. Predictions are pre-computed and distributed as CSV files via the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), keyed by UniProt accession.

**Why is this important?**
- Variant interpretation: triage missense variants of uncertain significance (VUS) reported in clinical sequencing
- Protein design: avoid substitutions predicted to disrupt fold or function during sequence optimization
- Disease research: prioritize candidate disease-causing variants from large case cohorts
- Constraint scoring: use as a per-residue penalty in directed evolution or generative-design loops
- Evolutionary analysis: contrast pathogenicity predictions with observed allele frequencies (e.g. gnomAD)

**Scientific foundation:**
AlphaMissense is an adaptation of AlphaFold fine-tuned on human and primate variant population frequency databases, treating variants common in healthy populations as benign and rare variants as putatively pathogenic. By combining the structural context inherited from AlphaFold's structure-prediction pretraining with evolutionary conservation signal, it produces a calibrated pathogenicity score per substitution. Class thresholds are calibrated against ClinVar. Per the DeepMind announcement of Cheng et al. (2023, *Science*), AlphaMissense classifies 89% of all 71 million possible human missense variants, with 32% labeled likely pathogenic and 57% labeled likely benign at the default thresholds.

## Tools

### AlphaMissense Fetch (`alphamissense-fetch`)

Fetch AlphaMissense pathogenicity scores for a UniProt accession.

AlphaMissense covers all reviewed human UniProt proteins. Non-human accessions
raise ValueError. Returns the full saturation grid (UniProt coords) or the
SNV-accessible subset (genomic coords); filter the output client-side as needed.

## How It Works

**Method overview:**
The tool issues a single HTTP GET to AlphaFold DB for one of three CSV file-name variants chosen by `coordinate_system`:
- `"uniprot"` (default) → `AF-{accession}-F1-aa-substitutions.csv` (3 cols: `protein_variant`, `am_pathogenicity`, `am_class`; full saturation grid)
- `"hg19"` → `AF-{accession}-F1-hg19.csv` (10 cols incl. `CHROM`, `POS`, `REF`, `ALT`, `transcript_id`; SNV-accessible only, GRCh37)
- `"hg38"` → `AF-{accession}-F1-hg38.csv` (10 cols, SNV-accessible only, GRCh38)

The CSV is parsed into a list of `AlphaMissensePrediction` records, each carrying the protein-level fields. In genomic mode the genomic-coordinate fields (`chrom`, `pos`, `ref`, `alt`, `transcript_id`) are also populated. Filtering is post-hoc and client-side — the wrapper exposes no filter knobs.

**Key assumptions:**
- The provided UniProt accession is a reviewed human protein covered by AlphaMissense
- Network access to alphafold.ebi.ac.uk is available
- The canonical UniProt sequence is the relevant isoform (AlphaMissense scores only the canonical sequence)

**Limitations:**
- Human proteome only: non-human accessions return 404 from the CSV endpoint and surface as `output.success=False` with a clear error message
- Missense substitutions only: does not predict effects of insertions, deletions, frameshifts, splice variants, or stop-gained / stop-lost
- Canonical isoform only: alternative isoforms, signal peptide cleavage products, and post-translational fragments are not separately scored
- Pre-computed scores: the wrapper does not run inference; if AlphaFold DB has not published a CSV for the accession, no result is available

**Computational requirements:**
- **Hardware:** CPU only, network access required
- **Runtime:** 1-5 seconds per query (one HTTP GET; a typical CSV is 1-10 MB)
- **Scalability:** Sequential queries; for batch retrieval, loop over accessions

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `uniprot_id` | `str` | UniProt accession for a reviewed human protein (e.g., `"P04637"`). Stripped and uppercased before use. |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coordinate_system` | `Literal["uniprot", "hg19", "hg38"]` | `"uniprot"` | Which AFDB CSV variant to fetch. UniProt mode returns the full saturation grid in protein coordinates; hg19/hg38 modes return only SNV-accessible substitutions mapped to genomic coordinates. |

**Filtering:** the wrapper exposes no filter knobs because the AFDB CSV has no server-side filtering API. Filter the returned `predictions` list client-side: `[p for p in output.predictions if p.position in {175, 248, 273}]` or `[p for p in output.predictions if p.pathogenicity_score >= 0.564]`.

**Common threshold shortcuts:** `pathogenicity_score >= 0.564` → likely-pathogenic only; `>= 0.34` → ambiguous + pathogenic.

## Output Specification

```python
# Return type: AlphaMissenseFetchOutput
AlphaMissenseFetchOutput(
    uniprot_accession: str,                              # UniProt accession looked up
    predictions: list[AlphaMissensePrediction],          # Per-substitution predictions
    num_predictions: int,                                # Number of predictions in the source CSV
    mean_pathogenicity: float | None,                    # Mean score across all predictions; None if empty
    source_url: str,                                     # URL of the AlphaMissense CSV fetched
)
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `uniprot_accession` | `str` | UniProt accession that was looked up (uppercased). |
| `predictions` | `list[AlphaMissensePrediction]` | Per-substitution pathogenicity predictions. |
| `num_predictions` | `int` | Number of predictions in the source CSV. For UniProt mode this is the full saturation grid (~7,500 rows for TP53); for hg19/hg38 modes it's the SNV-accessible subset (~2,500 for TP53). |
| `mean_pathogenicity` | `float \| None` | Mean pathogenicity score across all predictions; `None` when `predictions` is empty. |
| `source_url` | `str` | URL of the AlphaMissense CSV fetched (useful for provenance / debugging). |

**`AlphaMissensePrediction` fields:**

| Field | Type | Mode | Description |
|-------|------|------|-------------|
| `position` | `int` | both | 1-indexed residue position in the canonical UniProt sequence. |
| `wild_type_aa` | `str` | both | Wild-type amino acid at this position (single letter). |
| `alt_aa` | `str` | both | Alternate amino acid being scored (single letter). |
| `pathogenicity_score` | `float` | both | AlphaMissense pathogenicity score in `[0, 1]`. Higher = more likely to be pathogenic. |
| `classification` | `AlphaMissenseClass` | both | Class label: `"likely_benign"`, `"ambiguous"`, or `"likely_pathogenic"`. |
| `chrom` | `str \| None` | hg19/hg38 | Chromosome (e.g. `"chr17"`). `None` in UniProt mode. |
| `pos` | `int \| None` | hg19/hg38 | 1-indexed genomic position. `None` in UniProt mode. |
| `ref` | `str \| None` | hg19/hg38 | Reference allele. `None` in UniProt mode. |
| `alt` | `str \| None` | hg19/hg38 | Alternate allele. `None` in UniProt mode. |
| `transcript_id` | `str \| None` | hg19/hg38 | GENCODE transcript ID (e.g. `"ENST00000445888.6"`). `None` in UniProt mode. |

**Supported export formats:** `json`, `csv`

## Interpreting Results

**Pathogenicity score scale:**
The `pathogenicity_score` is a calibrated value in `[0, 1]`, where `0` indicates a substitution very likely to be tolerated and `1` indicates a substitution very likely to disrupt protein function. The score is on a continuous scale; the discrete `classification` is provided as a convenience that bins the score against the thresholds reported in the AlphaMissense paper.

**Classification thresholds (from Cheng et al. 2023):**
- **`likely_benign`**: `pathogenicity_score < 0.34`. The default cutoff used by AlphaMissense for benign calls. Substitutions in this band are predicted to be tolerated; consistent with common, neutral, or stabilizing variants.
- **`ambiguous`**: `0.34 <= pathogenicity_score <= 0.564`. The model has insufficient confidence to call benign or pathogenic. Treat with caution; corroborate with orthogonal evidence (e.g. ClinVar, gnomAD allele frequency, structural context).
- **`likely_pathogenic`**: `pathogenicity_score > 0.564`. The default cutoff used by AlphaMissense for pathogenic calls. Substitutions in this band are predicted to disrupt protein function; high prior for disease association.

**Important:** the `classification` field is pre-computed by the upstream AlphaMissense model and shipped in the CSV. This wrapper does not recompute the class from the score; the numeric thresholds above are documented in the paper and exposed here for reference and for downstream filtering logic.

**Interpreting edge cases:**
- A high `pathogenicity_score` does not guarantee a clinically reportable variant. AlphaMissense is calibrated on protein-level functional disruption; loss-of-function does not always produce disease in heterozygotes (recessive genes) or in genes under low selective constraint.
- A `likely_benign` call does not imply zero functional impact. Subtle, condition-dependent, or quantitative effects (e.g. reduced affinity, altered allostery) may still be present.
- `mean_pathogenicity` over a wide region (or the whole protein) is a coarse summary; for hotspot detection, group predictions by `position` and inspect distributions per residue.
- Empty `predictions`: only happens for non-human or uncovered accessions, which surface as `output.success=False`. The wrapper itself never filters the CSV — every row is returned as a `AlphaMissensePrediction`.

## Quick Start Examples

**Example 1: Fetch all AlphaMissense predictions for human TP53**
```python
from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

# Fetch all per-substitution predictions for human TP53 (P04637)
inputs = AlphaMissenseFetchInput(uniprot_id="P04637")
output = run_alphamissense_fetch(inputs, AlphaMissenseFetchConfig())

print(f"Accession: {output.uniprot_accession}")
print(f"Predictions in CSV: {output.num_predictions}")
print(f"Mean pathogenicity: {output.mean_pathogenicity:.3f}")
print(f"Source: {output.source_url}")

# Inspect the first few predictions
for p in output.predictions[:5]:
    print(f"  {p.wild_type_aa}{p.position}{p.alt_aa}: {p.pathogenicity_score:.3f} ({p.classification})")
```

**Example 2: Filter to high-confidence pathogenic substitutions for BRCA1**
```python
from collections import Counter

from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

# Pull the full saturation grid for human BRCA1 (P38398), then filter
# client-side. Useful for triaging missense VUS in a clinical sequencing pipeline.
output = run_alphamissense_fetch(
    AlphaMissenseFetchInput(uniprot_id="P38398"),
    AlphaMissenseFetchConfig(),
)

# 0.564 = likely_pathogenic threshold; 0.8 = high-confidence only
high_conf_pathogenic = [p for p in output.predictions if p.pathogenicity_score >= 0.8]
print(f"BRCA1: {len(high_conf_pathogenic)} of {output.num_predictions} substitutions are high-confidence pathogenic")

# Group by position to find hotspots
hotspots = Counter(p.position for p in high_conf_pathogenic).most_common(10)
print("Top hotspots (position, # pathogenic alts):")
for pos, count in hotspots:
    print(f"  {pos}: {count}/19")
```

**Example 3: Disease-relevant filter -- pathogenic vs benign at specific TP53 positions**
```python
from collections import Counter

from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

# TP53 R175, R248, R273 are well-known mutational hotspots in cancer.
output = run_alphamissense_fetch(
    AlphaMissenseFetchInput(uniprot_id="P04637"),
    AlphaMissenseFetchConfig(),
)

# Filter to the hotspot positions (client-side)
hotspots = {175, 248, 273}
hotspot_predictions = [p for p in output.predictions if p.position in hotspots]

# Sort by score; pathogenic first
ranked = sorted(hotspot_predictions, key=lambda p: -p.pathogenicity_score)
for p in ranked:
    print(f"{p.wild_type_aa}{p.position}{p.alt_aa}: {p.pathogenicity_score:.3f} ({p.classification})")

# Counts by class at these positions
class_counts = Counter(p.classification for p in hotspot_predictions)
print(f"\nClass distribution: {dict(class_counts)}")
```

**Example 4: Genomic mode -- map a VCF variant to its AlphaMissense score**
```python
from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

# Pull the GRCh38-coordinate AlphaMissense table for TP53.
# Each row carries the genomic coordinate fields needed to match a VCF entry.
output = run_alphamissense_fetch(
    AlphaMissenseFetchInput(uniprot_id="P04637"),
    AlphaMissenseFetchConfig(coordinate_system="hg38"),
)

# Look up a specific VCF variant: chr17:7669612 G->T
hits = [
    p for p in output.predictions
    if p.chrom == "chr17" and p.pos == 7669612 and p.ref == "G" and p.alt == "T"
]
for hit in hits:
    print(
        f"chr17:{hit.pos} {hit.ref}>{hit.alt} -> {hit.wild_type_aa}{hit.position}{hit.alt_aa} "
        f"on {hit.transcript_id}: {hit.classification} (score {hit.pathogenicity_score:.3f})"
    )
```

**Example 5: Chained workflow -- gene symbol -> UniProt -> AlphaMissense (variant-design constraint loop)**
```python
from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
    UniProtFetchConfig, UniProtFetchInput, run_uniprot_fetch,
)

# 1. UniProt: gene symbol -> canonical Swiss-Prot accession
#    `prefer_pdb_crossref=True` biases the ranker toward the reviewed entry.
uniprot = run_uniprot_fetch(
    UniProtFetchInput(target_name="KRAS", organism="Homo sapiens", prefer_pdb_crossref=True),
    UniProtFetchConfig(),
)
# uniprot.accession == "P01116", uniprot.length == 189

# 2. AlphaMissense: full saturation grid for the canonical accession
am = run_alphamissense_fetch(
    AlphaMissenseFetchInput(uniprot_id=uniprot.accession),
    AlphaMissenseFetchConfig(),
)
# am.num_predictions == 189 * 19 == 3591

# 3. Sanity check: UniProt sequence and AlphaMissense WT letters must agree.
#    A silent disagreement here would mean the constraint scores the wrong residues.
for prediction in am.predictions:
    assert prediction.wild_type_aa == uniprot.sequence[prediction.position - 1]
```

**Example 6: Chained workflow -- joining AFDB structure + AlphaMissense tolerance to rank "design-friendly" residues**
```python
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

accession = "P04637"  # human TP53

# Pull structure + per-residue pLDDT
afdb = run_alphafold_db_fetch(
    AlphaFoldDBFetchInput(uniprot_id=accession),
    AlphaFoldDBFetchConfig(),
)
plddt = afdb.structure.metrics["plddt_per_residue"]

# Pull saturation grid; aggregate by position
am = run_alphamissense_fetch(
    AlphaMissenseFetchInput(uniprot_id=accession),
    AlphaMissenseFetchConfig(),
)
classes_by_position: dict[int, list[str]] = {}
for prediction in am.predictions:
    classes_by_position.setdefault(prediction.position, []).append(prediction.classification)

# A residue is "design-friendly" if it sits in a confidently folded region (pLDDT > 80)
# AND a majority of substitutions are tolerated (likely_benign).
design_friendly = []
for position, classes in classes_by_position.items():
    pct_benign = sum(c == "likely_benign" for c in classes) / len(classes)
    if plddt[position - 1] > 80 and pct_benign > 0.5:
        design_friendly.append(position)

print(f"{len(design_friendly)} design-friendly TP53 residues")
# For TP53, the canonical cancer-driver positions (R175, R248, R273) appear in
# the complementary "intolerant" set instead (high pLDDT but high pathogenicity).
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Using a non-human UniProt accession:** AlphaMissense covers all reviewed human UniProt proteins only. Non-human accessions return 404 from the CSV endpoint and surface as `output.success=False` with a clear error message. Resolve the accession with `uniprot-fetch` first if you are unsure of the organism.
2. **Pulling all predictions in tight constraint loops:** A typical protein has 7,000-20,000 substitution rows. The wrapper always returns the full grid; cache the output once per accession and filter client-side as needed.
3. **Applying scores to non-missense variants:** The score is for missense substitutions only; it does not predict effects of insertions, deletions, frameshifts, or stop-gained / stop-lost. Do not extrapolate.
4. **Using AlphaMissense as the sole evidence for novel disease calls:** AlphaMissense is calibrated against ClinVar's known disease genes. For novel disease-related interpretations, cross-reference with ClinVar (`ncbi-efetch`), gnomAD allele frequencies, and orthogonal functional evidence.

**Tips for optimal results:**
- Use `[p for p in output.predictions if p.pathogenicity_score >= 0.564]` to keep only likely-pathogenic predictions; use `>= 0.8` for high-confidence only.
- For hotspot analysis on a single protein, inspect `predictions` grouped by `position` rather than relying on `mean_pathogenicity`.
- The full saturation grid for a typical protein is `19 × length` predictions (~1 MB for TP53). One fetch covers all client-side filters, so cache once per accession.

**Edge cases to watch for:**
- Recently added or recently retired UniProt accessions: AlphaFold DB CSVs are released as a fixed snapshot; very recent accessions may not have a CSV yet, returning 404.
- `mean_pathogenicity is None`: indicates `predictions` is empty (only happens when the upstream CSV is empty). Check `num_predictions` to confirm.
- Selenocysteine (U) and pyrrolysine (O): AlphaMissense scores the canonical 20-AA alphabet only; positions encoding non-standard residues may behave unexpectedly in downstream consumers.

## References

**Primary publication:**
- Cheng, J., Novati, G., Pan, J., Bycroft, C., Zemgulyte, A., Applebaum, T., Pritzel, A., Wong, L. H., Zielinski, M., Sargeant, T., Schneider, R. G., Senior, A. W., Jumper, J., Hassabis, D., Kohli, P., & Avsec, Z. (2023). "Accurate proteome-wide missense variant effect prediction with AlphaMissense." *Science*, 381(6664), eadg7492. [DOI: 10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492)
- Summary: Introduces AlphaMissense, an AlphaFold2-derived model fine-tuned with population-frequency and protein-language-modeling objectives, that classifies 89% of all 71M possible human missense variants (32% likely pathogenic, 57% likely benign) and outperforms prior methods on clinical and experimental benchmarks.

**Implementation:**
- AlphaMissense GitHub: [https://github.com/google-deepmind/alphamissense](https://github.com/google-deepmind/alphamissense)
- AlphaFold Protein Structure Database (hosts the AlphaMissense CSVs): [https://alphafold.ebi.ac.uk/](https://alphafold.ebi.ac.uk/)

## Related Tools

**Tools often used together:**
- **`uniprot-fetch`**: Resolve a gene name or organism to a canonical UniProt accession before calling `alphamissense-fetch`. Especially useful when the user provides a gene symbol rather than an accession, or to confirm an accession is a reviewed human protein.
- **`alphafold-db-fetch`**: Fetch the AlphaFold-predicted 3D structure for the same UniProt accession. AlphaMissense and AlphaFold DB share an accession-keyed workflow -- pulling both gives you per-residue pathogenicity scores aligned 1:1 with backbone coordinates for structural visualization. (`alphafold-db-fetch` lands in a separate PR; this tool ships AlphaMissense first.)

**Alternative tools (similar function):**
- **`ncbi-efetch`**: Pull ClinVar variant interpretations for the same gene. Use as orthogonal evidence; AlphaMissense provides a model-based score on every possible substitution, while ClinVar provides expert-curated calls on observed variants.
