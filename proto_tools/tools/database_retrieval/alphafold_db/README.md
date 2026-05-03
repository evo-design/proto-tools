<a href="https://bio-pro.mintlify.app/tools/database-retrieval/alphafold-db"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AlphaFold DB

## Overview

`alphafold-db-fetch` retrieves an AlphaFold-predicted protein structure from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) by UniProt accession. It returns the structure file text (PDB or mmCIF), per-residue pLDDT confidence, an optional pAE (predicted aligned error) matrix, and rich metadata (entry ID, AFDB version, gene, organism, source URLs, full JSON record). This is a CPU-only tool that wraps the AlphaFold DB REST API.

## Background

**What does this tool measure/predict?**
The [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) (AFDB) is a public archive of protein 3D structures predicted by [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) (Jumper et al. 2021), built and maintained by [DeepMind](https://deepmind.google/) and [EMBL-EBI](https://www.ebi.ac.uk/). It hosts predictions for >214 million UniProt sequences (Varadi et al. 2024 NAR update) across the proteomes of nearly every catalogued organism, indexed by UniProt accession. Each entry includes the predicted atomic coordinates, per-residue pLDDT confidence (0-100), and a pairwise pAE matrix that reports the expected positional error in angstroms when residues `i` and `j` are aligned.

**Why is this important?**
- Structural biology: obtain a high-quality predicted structure when no experimental PDB entry exists
- Protein design: use AFDB structures as starting templates for inverse folding (ProteinMPNN), docking, or binder design
- Functional annotation: identify domains, active sites, and disorder from predicted structure + pLDDT
- Comparative analysis: align AFDB structures (TM-align, US-align) across orthologs and homologs
- Quality assessment: pLDDT and pAE quantify per-residue and pairwise confidence so you can flag low-confidence regions

**Scientific foundation:**
AlphaFold2 (Jumper et al. 2021) predicts protein 3D structure from sequence using an attention-based neural network operating jointly on a multiple sequence alignment (MSA) and an evolving pair representation. AlphaFold DB (Varadi et al. 2022) industrialises this method to produce a predicted structure for every reference sequence in UniProt, with periodic version refreshes when the underlying model or pipeline is updated. AFDB serves only AlphaFold2 monomer predictions; complexes are produced by other pipelines (e.g. AlphaFold-Multimer, AlphaFold3) and are not part of this database.

## How It Works

**Method overview:**
The tool wraps the AlphaFold DB prediction API in a single HTTP flow:
1. **Metadata lookup:** GET `https://alphafold.ebi.ac.uk/api/prediction/{accession}` returns a JSON list of prediction records. Each record carries the entry ID, sequence, version, mean pLDDT (`globalMetricValue`), and download URLs for the PDB, mmCIF, pLDDT, pAE, and PAE-image files.
2. **Record selection:** AFDB returns multiple records when the protein has annotated alternative isoforms (common for human proteins; e.g. P04637 / TP53 returns 9 records, one per isoform) or when the canonical sequence is split into overlapping fragments (rare; only proteins longer than ~2,700 residues). The wrapper picks the first record (the canonical isoform's first fragment) and logs a warning so the caller knows about additional records.
3. **Optional payload downloads:** The tool then issues follow-up GETs for the configured artefacts: structure file text in `structure_format` (PDB or mmCIF), per-residue pLDDT array (extracted from `confidenceScore` in the pLDDT JSON), and the pAE matrix (extracted from `predicted_aligned_error` in the pAE JSON).

**Key assumptions:**
- Network access to `alphafold.ebi.ac.uk` is available
- The supplied UniProt accession matches AFDB's indexing (AFDB indexes UniProt reference sequences; isoforms and obsolete accessions may not resolve)
- For proteins with alternative isoforms or additional fragments, the canonical (first) record is sufficient for the caller's downstream use

**Limitations:**
- Monomer predictions only: AFDB does not host complexes (use AlphaFold-Multimer or AlphaFold3 for those)
- Coverage is broad but not universal: not every UniProt accession has an AFDB entry
- Multiple records per accession: alternative isoforms are selectable via the `isoform` input parameter (`isoform=2`, etc.); the wrapper defaults to the canonical record (`AF-{accession}-F1`). Fragments for proteins >~2,700 residues are not exposed -- query AFDB directly for downstream fragments.
- Predictions reflect AlphaFold2's confidence; low-pLDDT regions may be genuinely disordered or simply hard to predict
- Rate-limited by the AFDB API (typically generous, but burst queries may be throttled)

**Computational requirements:**
- **Hardware:** CPU only, network access required
- **Runtime:** 1-5 seconds per query for metadata + structure; pAE downloads can add several seconds for long proteins (matrix size is `N x N`)
- **Scalability:** Sequential queries; for batch retrieval, loop over accessions and consider caching per-accession output

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uniprot_id` | `str` | (required) | UniProt accession to look up (e.g., "P04637" for human TP53) |
| `isoform` | `int \| None` | `None` | Optional isoform number to select from the AFDB records list. `None` returns the canonical entry (e.g., `AF-P04637-F1` for TP53). Set to `2` for `AF-P04637-2-F1`, etc. Typical isoform numbers are 2-9. |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure_format` | `Literal["pdb", "cif"]` | `"pdb"` | Structure file format to download |
| `include_structure` | `bool` | `True` | Download the structure file text. Set to `False` for metadata-only probes (saves ~100-500 KB per call) |
| `include_plddt` | `bool` | `True` | Download the per-residue pLDDT JSON (~3 KB per call) |
| `include_pae` | `bool` | `False` | Download the pAE matrix; off by default because pAE files are large for long proteins |
| `include_msa` | `bool` | `False` | Download the A3M MSA used as input to prediction; off by default because A3M files can be hundreds of KB to several MB for highly conserved proteins |

## Output Specification

```python
# Return type: AlphaFoldDBFetchOutput
AlphaFoldDBFetchOutput(
    uniprot_accession: str,                 # Primary UniProt accession (e.g., "P04637")
    entry_id: str,                          # AlphaFold entry ID (e.g., "AF-P04637-F1")
    gene: Optional[str],                    # Gene symbol from the AFDB record
    organism_scientific_name: Optional[str],# Source organism scientific name
    tax_id: Optional[int],                  # NCBI taxonomy ID
    sequence: str,                          # Amino-acid sequence covered by the prediction
    sequence_length: int,                   # Length of the predicted sequence
    sequence_start: int,                    # 1-indexed start residue of the prediction
    sequence_end: int,                      # 1-indexed inclusive end residue of the prediction
    latest_version: int,                    # Latest AFDB version of this prediction (the version served)
    model_created_date: Optional[str],      # ISO 8601 prediction timestamp
    mean_plddt: Optional[float],            # Mean per-residue pLDDT (0-100)
    pdb_url: str,                           # URL to PDB structure file
    cif_url: str,                           # URL to mmCIF structure file
    pae_doc_url: str,                       # URL to pAE JSON document
    plddt_doc_url: str,                     # URL to per-residue pLDDT JSON document
    pae_image_url: str,                     # URL to rendered pAE PNG
    msa_url: Optional[str],                 # URL to MSA A3M file, when present
    structure_format: str,                  # "pdb" or "cif"; "" when include_structure=False
    structure_text: Optional[str],          # Structure file contents; None when include_structure=False
    plddt_per_residue: Optional[List[float]],  # Per-residue pLDDT (0-100); None when include_plddt=False
    pae_matrix: Optional[List[List[float]]],      # N x N pAE matrix; None when include_pae=False
    msa_a3m: Optional[str],                 # A3M MSA contents; None when include_msa=False or no msaUrl
    source_url: str,                        # AFDB API URL used for the metadata lookup
    raw_entry: Dict[str, Any],              # Complete AFDB JSON record
)
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `uniprot_accession` | `str` | Primary UniProt accession returned by AFDB |
| `entry_id` | `str` | AlphaFold entry identifier. Canonical isoform looks like `"AF-P04637-F1"` (`F1` = fragment 1). Alternative isoforms include the isoform index, e.g. `"AF-P04637-2-F1"` for isoform 2. |
| `gene` | `Optional[str]` | Gene symbol from the AFDB record |
| `organism_scientific_name` | `Optional[str]` | Source organism scientific name |
| `sequence` | `str` | Amino-acid sequence covered by the prediction |
| `sequence_length` | `int` | Length of the predicted sequence in residues |
| `latest_version` | `int` | Latest AFDB version of this prediction (always the version served by the API) |
| `mean_plddt` | `Optional[float]` | Global mean pLDDT score (0-100); higher is more confident |
| `pdb_url` | `str` | URL to the PDB structure file on AFDB |
| `cif_url` | `str` | URL to the mmCIF structure file on AFDB |
| `plddt_doc_url` | `str` | URL to the per-residue pLDDT JSON document |
| `pae_doc_url` | `str` | URL to the pAE JSON document |
| `structure_format` | `str` | `"pdb"` or `"cif"`, matching the downloaded file |
| `structure_text` | `Optional[str]` | Structure file contents in `structure_format`; `None` when `include_structure=False` |
| `plddt_per_residue` | `Optional[List[float]]` | Per-residue pLDDT scores (0-100), one per residue; `None` when `include_plddt=False` |
| `pae_matrix` | `Optional[List[List[float]]]` | N x N predicted aligned error matrix in angstroms; `None` when `include_pae=False` |
| `msa_a3m` | `Optional[str]` | A3M-format MSA contents used as input to prediction; `None` when `include_msa=False` or when the entry has no associated `msaUrl` |
| `raw_entry` | `Dict[str, Any]` | Complete AFDB JSON record for advanced programmatic access |

**Supported export formats:** `json`

## Interpreting Results

**pLDDT confidence tiers (per-residue and global):**
- **Very high:** `pLDDT > 90`: Reliable backbone and side-chain orientation. Safe to use for docking, design, and structural reasoning.
- **Confident:** `70 < pLDDT <= 90`: Backbone is reliable; side chains may be less precise. Suitable for most downstream tasks.
- **Low:** `50 < pLDDT <= 70`: Treat with caution. Local conformation may be wrong, although general topology is often plausible.
- **Very low:** `pLDDT <= 50`: Likely disordered, flexible, or genuinely uncertain. Often biologically meaningful (linkers, IDRs); do not interpret as a folded structure.

**pAE (predicted aligned error):**
- pAE is a pairwise matrix in angstroms reporting AlphaFold's expected error in residue `i`'s position when the structure is aligned on residue `j`. **Lower is better.**
- Low-pAE blocks (typically `< 5 angstrom`) along the diagonal indicate confidently folded domains.
- Low-pAE off-diagonal blocks indicate confident relative placement of two regions (i.e., they are reliably packed together in 3D).
- High-pAE off-diagonal blocks indicate that the relative orientation between two regions is uncertain (common between flexibly linked domains).

**Interpreting edge cases:**
- A high `mean_plddt` can hide locally unreliable regions; always inspect `plddt_per_residue` before trusting a specific residue.
- `latest_version` advances when AFDB refreshes the prediction (e.g. with a newer AlphaFold model); refetch when the version moves past what you cached.
- For very long proteins split into multiple fragments, `sequence_start` / `sequence_end` describe the residue range covered by the canonical first fragment, not the full UniProt sequence.

## Quick Start Examples

**Example 1: Fetch a structure by UniProt accession**
```python
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
)

# Fetch the AlphaFold prediction for human TP53
inputs = AlphaFoldDBFetchInput(uniprot_id="P04637")
output = run_alphafold_db_fetch(inputs, AlphaFoldDBFetchConfig())

print(f"Entry: {output.entry_id} ({output.organism_scientific_name})")
print(f"Length: {output.sequence_length} aa")
print(f"Mean pLDDT: {output.mean_plddt:.1f}")
print(f"AFDB version: v{output.latest_version}")
print(f"Structure ({output.structure_format}, first 200 chars):\n{output.structure_text[:200]}")
```

**Example 2: Fetch as mmCIF and include the pAE matrix**
```python
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
)

# Fetch human MLH1 in mmCIF format with the pAE matrix for inter-domain analysis
inputs = AlphaFoldDBFetchInput(uniprot_id="Q9Y6K9")
config = AlphaFoldDBFetchConfig(
    structure_format="cif",  # mmCIF preserves chain/residue metadata more cleanly
    include_pae=True,        # Need pAE to assess inter-domain orientation confidence
)
output = run_alphafold_db_fetch(inputs, config)

# Identify low-confidence residues
low_conf = [i + 1 for i, score in enumerate(output.plddt_per_residue) if score < 70]
print(f"{len(low_conf)} residues with pLDDT < 70 (out of {output.sequence_length})")

# Mean pAE between residue blocks (rough inter-domain confidence indicator)
n = len(output.pae_matrix)
print(f"pAE matrix: {n} x {n}, mean = {sum(sum(r) for r in output.pae_matrix) / (n * n):.2f} angstrom")
```

**Example 3: Metadata-only probe (skip the heavy downloads)**
```python
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
)

# Cheap probe: confirm AFDB has the protein, get the URLs and mean pLDDT,
# but skip the structure file (~250 KB) and per-residue pLDDT (~3 KB).
# Useful for batch coverage checks before committing to large downloads.
output = run_alphafold_db_fetch(
    AlphaFoldDBFetchInput(uniprot_id="P00533"),  # EGFR
    AlphaFoldDBFetchConfig(include_structure=False, include_plddt=False),
)

print(f"AFDB has {output.entry_id}, mean pLDDT {output.mean_plddt:.1f}")
print(f"PDB URL:  {output.pdb_url}")
print(f"mmCIF URL: {output.cif_url}")
# output.structure_text is None; output.plddt_per_residue is None
```

**Example 4: Save the structure to disk for downstream tools**
```python
from pathlib import Path

from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
)

inputs = AlphaFoldDBFetchInput(uniprot_id="P04637")
output = run_alphafold_db_fetch(inputs, AlphaFoldDBFetchConfig())

pdb_path = Path(f"{output.entry_id}.pdb")
pdb_path.write_text(output.structure_text)
print(f"Wrote {pdb_path} ({pdb_path.stat().st_size:,} bytes)")
# Now usable as input to tmalign, usalign, ProteinMPNN, PyRosetta, etc.
```

**Example 5: Chained workflow -- gene symbol -> UniProt -> AFDB structure (template-fetching for variant design)**
```python
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
    UniProtFetchConfig, UniProtFetchInput, run_uniprot_fetch,
)

# 1. UniProt: resolve a gene symbol to its canonical Swiss-Prot accession.
#    `prefer_pdb_crossref=True` biases the ranker toward the reviewed entry
#    (without it the search can return an unreviewed TrEMBL hit).
uniprot = run_uniprot_fetch(
    UniProtFetchInput(target_name="KRAS", organism="Homo sapiens", prefer_pdb_crossref=True),
    UniProtFetchConfig(),
)
# uniprot.accession == "P01116", uniprot.length == 189

# 2. AFDB: fetch the predicted structure for the canonical accession.
afdb = run_alphafold_db_fetch(
    AlphaFoldDBFetchInput(uniprot_id=uniprot.accession),
    AlphaFoldDBFetchConfig(),
)
# afdb.entry_id == "AF-P01116-F1"

# 3. Sanity check: UniProt and AFDB must agree on the canonical sequence.
#    Disagreement here would silently corrupt downstream design tools.
assert afdb.sequence == uniprot.sequence
assert afdb.sequence_length == uniprot.length
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Assuming every UniProt accession has an AFDB entry:** AFDB has broad but not universal coverage. The wrapper raises `ValueError` when the API returns no prediction; catch this and fall back to predicting from sequence with `esmfold` or `alphafold3`.
2. **Downloading pAE for long proteins by default:** pAE matrices scale as `N x N` and can be tens of MB for proteins with thousands of residues. `include_pae` defaults to `False`; only enable when you actually need inter-residue confidence.
3. **Ignoring the warning when multiple records are returned:** AFDB returns extra records when the protein has alternative isoforms (very common for human proteins) or when a >2,700-residue sequence is split across fragments (rare). The wrapper defaults to the canonical record (`AF-{accession}-F1`) and logs a warning listing other available isoforms; pass `isoform=N` to select a specific alternative isoform. Downstream fragments require querying AFDB directly.
4. **Expecting complex structures:** AFDB hosts AlphaFold2 monomer predictions only. For complexes, use `alphafold3` or AlphaFold-Multimer; for an experimental complex, use `pdb-fetch-entry`.
5. **Using AFDB for designed (non-natural) sequences:** AFDB only contains predictions for natural UniProt sequences. For wild-type targets that are not in AFDB, or for any custom sequence, predict the structure de novo with `esmfold` or `alphafold3`.

**Tips for optimal results:**
- Use `mmCIF` (`structure_format="cif"`) when downstream tooling needs full chain/residue metadata; PDB is fine for most quick analyses.
- Inspect `plddt_per_residue` before any per-residue interpretation; the global `mean_plddt` can be misleading.
- Cache `latest_version` alongside any structure you persist; refetch when AFDB advances past your cached version.

**Edge cases to watch for:**
- 404 from AFDB -> the wrapper raises `ValueError("AlphaFold DB has no prediction for accession ...")`. Handle it and fall back to de novo prediction (`esmfold`, `alphafold3`).
- Multiple records (alternative isoforms or additional fragments) -> the canonical record (`AF-{accession}-F1`) is returned by default and a warning lists other available isoforms; pass `isoform=N` to select a specific alternative. Check `entry_id` and `sequence_start` / `sequence_end` to confirm which record was selected.
- Disordered or flexibly linked regions -> very-low pLDDT and high off-diagonal pAE are expected; treat them as biology, not as a model failure.

## References

**Primary publication:**
- Varadi, M., Anyango, S., Deshpande, M., Nair, S., Natassia, C., Yordanova, G., et al. (2022). "AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models." *Nucleic Acids Research*, 50(D1), D439-D444. [DOI: 10.1093/nar/gkab1061](https://doi.org/10.1093/nar/gkab1061)
- Summary: Describes AlphaFold DB, the public archive that hosts AlphaFold2 predicted structures for the UniProt reference proteomes, indexed by UniProt accession with per-residue pLDDT and pAE confidence.

**Underlying method:**
- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596, 583-589. [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- Summary: Introduces AlphaFold2, the attention-based neural network that produces all structures hosted in AlphaFold DB.

**Implementation:**
- AlphaFold DB website: [https://alphafold.ebi.ac.uk/](https://alphafold.ebi.ac.uk/)
- AlphaFold DB API: [https://alphafold.ebi.ac.uk/api-docs](https://alphafold.ebi.ac.uk/api-docs)

## Related Tools

**Tools often used together:**
- **`uniprot-fetch`**: Resolve a gene name or organism to a UniProt accession first, then pass that accession to `alphafold-db-fetch`.
- **`tmalign`** / **`usalign`**: Compare an AFDB structure against an experimental PDB structure or another predicted model to quantify structural similarity.
- **`pdb-fetch-entry`**: Pull experimental structures (with method, resolution, ligands) when AFDB has a prediction but you want the experimental counterpart.

**Alternative tools (similar function):**
- **`pdb-fetch-entry`**: Use when an experimental structure exists; AFDB predictions are a complement, not a replacement, for experimental data.
- **`esmfold`**: Predict structure from sequence directly when AFDB has no entry (e.g., designed sequences, missing accessions, custom variants).
- **`alphafold3`**: Predict structure from scratch when AFDB has no entry, when you need a complex (AFDB is monomer-only), or when you want a current AlphaFold3-quality prediction rather than the cached AlphaFold2 result.
