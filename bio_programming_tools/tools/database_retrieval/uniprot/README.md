<a href="https://bio-pro.mintlify.app/tools/database-retrieval/uniprot"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# UniProt

## Overview

`uniprot-fetch` retrieves protein entries from UniProt by accession ID or by searching with target name and organism. It returns the protein sequence, gene names, PDB cross-references, and the full UniProt JSON record. This is a CPU-only tool that wraps the UniProt REST API.

## When to Use This Tool

**Primary use cases:**
- Fetching a protein sequence by UniProt accession (e.g., P04637 for human TP53)
- Searching UniProt by gene name + organism to resolve accessions when you don't have an ID
- Extracting PDB cross-references from UniProt entries to find available experimental structures
- Getting curated protein metadata (gene names, review status, function annotations)

**When NOT to use this tool:**
- Multi-source orchestrated fetches across NCBI, UniProt, and PDB: use `sequence-fetch` instead, which automatically routes to the best database
- DNA or RNA sequence retrieval: use `ncbi-efetch` (UniProt is protein-only)
- PDB structure metadata or chain sequences: use `pdb-fetch-entry` / `pdb-fetch-fasta`
- Homology search or sequence similarity: use `blast-search` or `mmseqs-search-proteins`

**Comparison with alternatives:**
- **`uniprot-fetch` vs `sequence-fetch`:** `sequence-fetch` is a higher-level orchestrator that can resolve sequences from gene names by querying multiple databases. Use `uniprot-fetch` when you specifically need UniProt metadata (review status, gene names, PDB cross-refs) or when you already have a UniProt accession.
- **`uniprot-fetch` vs `ncbi-efetch`:** UniProt is protein-centric with curated annotations. NCBI covers all sequence types (protein, DNA, RNA) but with less curation. Use UniProt for protein-specific queries, NCBI for nucleotide sequences or when you have NCBI-style accessions.

## Biological Background

**What does this tool measure/predict?**
[UniProt](https://www.uniprot.org/) (Universal Protein Resource) is the most comprehensive curated protein sequence and functional annotation database. It combines data from [Swiss-Prot](https://en.wikipedia.org/wiki/Swiss-Prot) (manually reviewed, ~570K entries) and [TrEMBL](https://en.wikipedia.org/wiki/UniProt#UniProtKB/TrEMBL) (computationally annotated, ~250M entries).

**Why is this important?**
- Protein design: retrieve reference sequences for target proteins before optimization
- Structural biology: find PDB structures linked to a protein of interest
- Functional annotation: access curated function descriptions, active sites, domains
- Comparative analysis: retrieve well-characterized homologs for benchmarking designed sequences
- Quality assessment: reviewed (Swiss-Prot) entries provide experimentally validated reference sequences

**Scientific foundation:**
UniProt integrates sequence data from [EMBL-Bank](https://www.ebi.ac.uk/ena/browser/home), protein structures from [PDB](https://www.rcsb.org/), and curated annotations from expert biocurators. Swiss-Prot entries undergo manual review including experimental evidence attribution, cross-referencing to literature, and standardized functional annotation. The database is updated biweekly with new sequences and annotations from the scientific community.

## How It Works

**Method overview:**
The tool wraps the UniProt REST API with two modes:
1. **Direct lookup** (`uniprot_id` provided): Fetches the entry directly by accession from the UniProt API
2. **Name-based search** (`target_name` + `organism` provided): Searches UniProt using two query strategies (exact gene match, then broader gene+protein match), ranks results by review status and gene name match quality, and returns the best match

**Key assumptions:**
- Network access to api.uniprot.org is available
- UniProt accessions follow standard format (e.g., P04637, Q9Y6K9)
- For name-based search: at least one matching entry exists in UniProt

**Limitations:**
- Protein sequences only (no DNA, RNA, or small molecule data)
- Search quality depends on how well the gene/protein name matches UniProt nomenclature
- Rate-limited by UniProt API (typically generous, but burst queries may be throttled)

**Computational requirements:**
- **Hardware:** CPU only, network access required
- **Runtime:** 1-5 seconds per query (depends on network latency and UniProt API load)
- **Scalability:** Sequential queries; for batch retrieval, loop over accessions

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uniprot_id` | `Optional[str]` | `None` | UniProt accession for direct entry lookup (e.g., "P04637") |
| `target_name` | `Optional[str]` | `None` | Gene or protein name for search (e.g., "TP53", "tumor protein p53") |
| `organism` | `Optional[str]` | `None` | Organism for search disambiguation (e.g., "Homo sapiens", "E. coli") |
| `prefer_pdb_crossref` | `bool` | `False` | When searching by name, prioritize entries that have linked PDB structures |
| `max_candidates` | `int` | `5` | Maximum number of search results to evaluate when ranking (1-20) |

**Validation rules:** Provide either `uniprot_id` (direct lookup) OR both `target_name` and `organism` (search mode). The validator enforces this.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_timeout_seconds` | `int` | `15` | HTTP timeout in seconds per request |
| `http_retries` | `int` | `2` | Number of retries for failed HTTP requests |
| `backoff_seconds` | `float` | `1.0` | Initial wait between retries (doubles after each attempt) |
| `user_agent` | `str` | `"bio-programming-tools/uniprot-fetch-v1"` | Identifier string sent to UniProt API |

## Output Specification

```python
# Return type: UniProtFetchOutput
UniProtFetchOutput(
    accession: str,              # Primary UniProt accession (e.g., "P04637")
    sequence: Optional[str],     # Protein amino acid sequence
    length: Optional[int],       # Sequence length
    entry_type: Optional[str],   # Review status (Swiss-Prot vs TrEMBL)
    gene_names: List[str],       # Gene name symbols
    pdb_crossrefs: List[str],    # Linked PDB structure IDs
    source_url: str,             # UniProt entry URL
    raw_entry: Dict[str, Any],   # Complete UniProt JSON record
)
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `accession` | `str` | Primary UniProt accession ID |
| `sequence` | `Optional[str]` | Full protein amino acid sequence |
| `length` | `Optional[int]` | Sequence length in amino acids |
| `entry_type` | `Optional[str]` | Review status: `"UniProtKB reviewed (Swiss-Prot)"` for curated entries, `"UniProtKB unreviewed (TrEMBL)"` for computationally annotated |
| `gene_names` | `List[str]` | Gene name symbols (e.g., `["TP53", "P53"]`) |
| `pdb_crossrefs` | `List[str]` | PDB IDs linked to this protein (e.g., `["1TUP", "2XWR"]`) |
| `raw_entry` | `Dict[str, Any]` | Complete UniProt JSON for advanced access (function annotations, cross-references, feature tables, etc.) |

**Supported export formats:** `json`

## Interpreting Results

**Entry quality indicators:**
- **Swiss-Prot (reviewed):** `entry_type` contains "reviewed" — manually curated, experimentally validated. Highest quality; use as reference sequences.
- **TrEMBL (unreviewed):** `entry_type` contains "unreviewed" — computationally annotated, not manually verified. Useful but verify critical annotations independently.

**PDB cross-references:**
- Non-empty `pdb_crossrefs` means experimental 3D structures exist for this protein
- Multiple PDB IDs may represent different conditions, ligand complexes, or resolution improvements
- Use `pdb-fetch-entry` to get metadata (method, resolution) for specific PDB IDs

**Interpreting edge cases:**
- Empty `pdb_crossrefs` does not mean the protein structure is unknown — it may have been predicted by AlphaFold2 (check the [AlphaFold DB](https://alphafold.ebi.ac.uk/))
- Multiple gene names are common for well-studied proteins with historical nomenclature
- Very long `raw_entry` records contain rich functional annotation; explore `raw_entry["comments"]` for function, subcellular location, and disease associations

## Quick Start Examples

**Example 1: Fetch by accession**
```python
from bio_programming_tools.tools.database_retrieval import (
    UniProtFetchConfig, UniProtFetchInput, run_uniprot_fetch,
)

# Fetch human TP53 by accession
inputs = UniProtFetchInput(uniprot_id="P04637")
output = run_uniprot_fetch(inputs, UniProtFetchConfig())

print(f"Accession: {output.accession}")
print(f"Gene names: {output.gene_names}")
print(f"Length: {output.length} aa")
print(f"Review status: {output.entry_type}")
print(f"PDB structures: {output.pdb_crossrefs[:5]}")
print(f"Sequence: {output.sequence[:50]}...")
```

**Example 2: Search by gene name and organism**
```python
from bio_programming_tools.tools.database_retrieval import (
    UniProtFetchConfig, UniProtFetchInput, run_uniprot_fetch,
)

# Find Cas9 from S. pyogenes
inputs = UniProtFetchInput(
    target_name="Cas9",
    organism="Streptococcus pyogenes",
    prefer_pdb_crossref=True,  # Prioritize entries with linked PDB structures
)
output = run_uniprot_fetch(inputs, UniProtFetchConfig())

print(f"Found: {output.accession} ({output.gene_names})")
print(f"Length: {output.length} aa")
print(f"Available PDB structures: {output.pdb_crossrefs}")
```

**Example 3: Access advanced annotations from raw entry**
```python
from bio_programming_tools.tools.database_retrieval import (
    UniProtFetchConfig, UniProtFetchInput, run_uniprot_fetch,
)

inputs = UniProtFetchInput(uniprot_id="P04637")
output = run_uniprot_fetch(inputs, UniProtFetchConfig())

# Explore function annotations
for comment in output.raw_entry.get("comments", []):
    if comment.get("commentType") == "FUNCTION":
        for text in comment.get("texts", []):
            print(f"Function: {text.get('value', '')[:200]}...")

# List all cross-referenced databases
xrefs = output.raw_entry.get("uniProtKBCrossReferences", [])
db_types = set(xref.get("database") for xref in xrefs)
print(f"Cross-referenced databases: {sorted(db_types)}")
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Providing neither accession nor name+organism:** The validator requires either `uniprot_id` OR both `target_name` and `organism`. Providing only `target_name` without `organism` will fail.
2. **Using organism common names inconsistently:** UniProt accepts both scientific names ("Escherichia coli") and common names ("E. coli"), but scientific names give more precise results.
3. **Assuming all entries have PDB structures:** Many proteins lack experimental structures. Check `pdb_crossrefs` before attempting to fetch PDB data.
4. **Ignoring review status:** Swiss-Prot (reviewed) entries are manually curated and more reliable than TrEMBL (unreviewed) entries. Always check `entry_type` for critical applications.

**Tips for optimal results:**
- Set `prefer_pdb_crossref=True` when you need both sequence and structure information
- Use `raw_entry` for advanced use cases — it contains function annotations, disease associations, subcellular localization, active sites, and more
- For ambiguous gene names, increase `max_candidates` to evaluate more search results

**Edge cases to watch for:**
- Deprecated or merged accessions: UniProt may redirect to a new accession
- Isoforms: the primary sequence may not match all known isoforms; check `raw_entry` for isoform data
- Very common gene names (e.g., "kinase"): search may return unexpected results; be specific with organism

## References

**Primary publication:**
- The UniProt Consortium. (2025). "UniProt: the Universal Protein Knowledgebase in 2025." *Nucleic Acids Research*, 53(D1), D609-D617. [DOI: 10.1093/nar/gkae1010](https://doi.org/10.1093/nar/gkae1010)
- Summary: Describes the UniProt database combining Swiss-Prot (curated) and TrEMBL (computed) protein sequences and functional annotations, serving as the primary reference for protein information.

**Implementation:**
- UniProt REST API: [https://rest.uniprot.org](https://rest.uniprot.org)
- UniProt website: [https://www.uniprot.org](https://www.uniprot.org)

## Related Tools

**Tools often used together:**
- **`sequence-fetch`**: Multi-source orchestrator that resolves sequences from gene names across NCBI, UniProt, and PDB. Use when you want automatic database routing.
- **`pdb-fetch-entry`** / **`pdb-fetch-fasta`**: Fetch PDB metadata and chain sequences for PDB IDs found in `pdb_crossrefs`.
- **`blast-search`**: Search for homologs of a protein sequence retrieved from UniProt.

**Alternative tools (similar function):**
- **`ncbi-efetch`**: NCBI Entrez sequence retrieval. Broader scope (protein + nucleotide + gene) but less curated for proteins than UniProt.
- **`sequence-fetch`**: Higher-level orchestrator; use when you don't need UniProt-specific metadata.
