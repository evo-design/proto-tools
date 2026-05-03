<a href="https://bio-pro.mintlify.app/tools/database-retrieval/alphamissense"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AlphaMissense

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

## How It Works

**Method overview:**
The tool issues a single HTTP GET against `https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-aa-substitutions.csv`, parses the CSV (one row per substitution with columns `protein_variant`, `am_pathogenicity`, `am_class`), and applies any optional filters specified in `Config` (positions, alt residues, score range, classification). It returns the filtered list of `AlphaMissensePrediction` records along with summary statistics (`num_total_predictions`, `num_returned`, `mean_pathogenicity`, `source_url`).

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
| `positions` | `list[int] \| None` | `None` | If set, return only predictions whose 1-indexed position is in this list. `None` returns all positions. |
| `alt_residues` | `list[str] \| None` | `None` | If set, return only predictions whose alternate amino acid (single letter) is in this list. `None` returns all alts. |
| `min_pathogenicity` | `float \| None` | `None` | If set, drop predictions with score below this threshold (range 0.0-1.0). |
| `max_pathogenicity` | `float \| None` | `None` | If set, drop predictions with score above this threshold (range 0.0-1.0). |
| `classification_filter` | `list[AlphaMissenseClass] \| None` | `None` | If set, return only predictions whose classification is in this list. `AlphaMissenseClass` is `Literal["likely_benign", "ambiguous", "likely_pathogenic"]`. |

All filter knobs are AND-combined: e.g. `positions=[100, 200]` together with `alt_residues=["L", "K"]` returns only the four `(pos, alt)` combinations that satisfy both constraints.

## Output Specification

```python
# Return type: AlphaMissenseFetchOutput
AlphaMissenseFetchOutput(
    uniprot_accession: str,                              # UniProt accession looked up
    predictions: list[AlphaMissensePrediction],          # Per-substitution predictions after filtering
    num_total_predictions: int,                          # Predictions in source CSV before filtering
    num_returned: int,                                   # Predictions in `predictions` after filtering
    mean_pathogenicity: float | None,                    # Mean score across returned predictions; None if empty
    source_url: str,                                     # URL of the AlphaMissense CSV fetched
)
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `uniprot_accession` | `str` | UniProt accession that was looked up (uppercased). |
| `predictions` | `list[AlphaMissensePrediction]` | Per-substitution pathogenicity predictions, after Config filters have been applied. |
| `num_total_predictions` | `int` | Number of predictions in the source CSV before any filters were applied. |
| `num_returned` | `int` | Number of predictions in `predictions` after filtering. Equal to `num_total_predictions` when no filters are set. |
| `mean_pathogenicity` | `float \| None` | Mean pathogenicity score across the returned predictions; `None` when `predictions` is empty. |
| `source_url` | `str` | URL of the AlphaMissense CSV fetched (useful for provenance / debugging). |

**`AlphaMissensePrediction` fields:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `position` | `int` | `>= 1` | 1-indexed residue position in the canonical UniProt sequence. |
| `wild_type_aa` | `str` | single letter | Wild-type amino acid at this position. |
| `alt_aa` | `str` | single letter | Alternate amino acid being scored. |
| `pathogenicity_score` | `float` | `0.0 - 1.0` | AlphaMissense pathogenicity score. Higher values indicate the variant is more likely to be pathogenic. |
| `classification` | `AlphaMissenseClass` | `Literal` | Pre-computed AlphaMissense class label: `"likely_benign"`, `"ambiguous"`, or `"likely_pathogenic"`. |

**Supported export formats:** `json`

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
- Empty `predictions` after aggressive filtering: re-check filter values; very high `min_pathogenicity` thresholds combined with a small protein can yield zero rows.

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
print(f"Total predictions in CSV: {output.num_total_predictions}")
print(f"Returned (after filters): {output.num_returned}")
print(f"Mean pathogenicity: {output.mean_pathogenicity:.3f}")
print(f"Source: {output.source_url}")

# Inspect the first few predictions
for p in output.predictions[:5]:
    print(f"  {p.wild_type_aa}{p.position}{p.alt_aa}: {p.pathogenicity_score:.3f} ({p.classification})")
```

**Example 2: Fetch only likely-pathogenic substitutions for BRCA1**
```python
from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

# Pull only high-confidence pathogenic predictions for human BRCA1 (P38398).
# Useful for triaging missense VUS in a clinical sequencing pipeline.
inputs = AlphaMissenseFetchInput(uniprot_id="P38398")
config = AlphaMissenseFetchConfig(
    classification_filter=["likely_pathogenic"],  # Drop ambiguous + benign
    min_pathogenicity=0.8,                         # Tighten beyond default 0.564 cutoff
)
output = run_alphamissense_fetch(inputs, config)

print(f"BRCA1: {output.num_returned} of {output.num_total_predictions} substitutions are high-confidence pathogenic")

# Group by position to find hotspots
from collections import Counter
hotspots = Counter(p.position for p in output.predictions).most_common(10)
print("Top hotspots (position, # pathogenic alts):")
for pos, count in hotspots:
    print(f"  {pos}: {count}/19")
```

**Example 3: Disease-relevant filter -- pathogenic vs benign at specific TP53 positions**
```python
from proto_tools.tools.database_retrieval import (
    AlphaMissenseFetchConfig, AlphaMissenseFetchInput, run_alphamissense_fetch,
)

# TP53 R175, R248, R273 are well-known mutational hotspots in cancer.
# Compare predicted pathogenicity for substitutions at these positions.
inputs = AlphaMissenseFetchInput(uniprot_id="P04637")
config = AlphaMissenseFetchConfig(
    positions=[175, 248, 273],  # Cancer hotspot residues
)
output = run_alphamissense_fetch(inputs, config)

# Sort by score; pathogenic first
ranked = sorted(output.predictions, key=lambda p: -p.pathogenicity_score)
for p in ranked:
    print(f"{p.wild_type_aa}{p.position}{p.alt_aa}: {p.pathogenicity_score:.3f} ({p.classification})")

# Counts by class at these positions
from collections import Counter
class_counts = Counter(p.classification for p in output.predictions)
print(f"\nClass distribution: {dict(class_counts)}")
```

**Example 4: Chained workflow -- gene symbol -> UniProt -> AlphaMissense (variant-design constraint loop)**
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
# am.num_total_predictions == 189 * 19 == 3591

# 3. Sanity check: UniProt sequence and AlphaMissense WT letters must agree.
#    A silent disagreement here would mean the constraint scores the wrong residues.
for prediction in am.predictions:
    assert prediction.wild_type_aa == uniprot.sequence[prediction.position - 1]
```

**Example 5: Chained workflow -- joining AFDB structure + AlphaMissense tolerance to rank "design-friendly" residues**
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
plddt = afdb.plddt_per_residue

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
2. **Pulling all predictions in tight constraint loops:** A typical protein has 7,000-20,000 substitution rows. When using AlphaMissense as a per-step constraint inside an optimization loop, set `min_pathogenicity` / `max_pathogenicity` (or `classification_filter`) to keep payloads small.
3. **Misreading combined filters:** Filter combinations are AND-ed. `positions=[100, 200]` with `alt_residues=["L", "K"]` returns only the four `(pos, alt)` combinations that satisfy both, not the union of position-100 rows and alt-L rows.
4. **Applying scores to non-missense variants:** The score is for missense substitutions only; it does not predict effects of insertions, deletions, frameshifts, or stop-gained / stop-lost. Do not extrapolate.
5. **Using AlphaMissense as the sole evidence for novel disease calls:** AlphaMissense is calibrated against ClinVar's known disease genes. For novel disease-related interpretations, cross-reference with ClinVar (`ncbi-efetch`), gnomAD allele frequencies, and orthogonal functional evidence.

**Tips for optimal results:**
- Use `classification_filter=["likely_pathogenic"]` together with a tighter `min_pathogenicity` (e.g. `0.8`) to rank only high-confidence variants.
- For hotspot analysis on a single protein, inspect `predictions` grouped by `position` rather than relying on `mean_pathogenicity`.
- Cache the `source_url` and the full unfiltered output once per accession when running many filtered queries against the same protein -- a single fetch already contains all 19x(L) substitutions.

**Edge cases to watch for:**
- Recently added or recently retired UniProt accessions: AlphaFold DB CSVs are released as a fixed snapshot; very recent accessions may not have a CSV yet, returning 404.
- `mean_pathogenicity is None`: indicates `predictions` is empty after filtering, not that the protein is uncovered. Check `num_total_predictions` to disambiguate.
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
