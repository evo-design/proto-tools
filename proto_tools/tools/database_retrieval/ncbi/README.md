<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ncbi"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# NCBI Entrez

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

Three tools wrapping NCBI Entrez E-utilities for searching and retrieving biological sequences (protein, nucleotide, gene):

- **`ncbi-esearch`**: Search for IDs by query term
- **`ncbi-esummary`**: Retrieve record metadata by ID
- **`ncbi-efetch`**: Fetch sequences/records by ID in FASTA or other formats

All are CPU-only tools that wrap the NCBI Entrez REST API.

## Tool Catalog

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `ncbi-esearch` | Database + search term | List of NCBI IDs | Find sequence IDs by gene name, organism, or keyword |
| `ncbi-esummary` | Database + ID | Record metadata dict | Get gene summaries, accession details, annotations |
| `ncbi-efetch` | Database + ID + format | FASTA records | Download sequences by accession in FASTA format |

## Background

**What do these tools access?**
[NCBI](https://www.ncbi.nlm.nih.gov/) (National Center for Biotechnology Information) maintains the world's largest collection of biological sequence databases. The [Entrez](https://www.ncbi.nlm.nih.gov/search/) system provides unified search and retrieval across multiple databases:

- **protein**: Protein sequences from [RefSeq](https://www.ncbi.nlm.nih.gov/refseq/), [GenBank](https://www.ncbi.nlm.nih.gov/genbank/), [Swiss-Prot](https://www.uniprot.org/), [PIR](https://proteininformationresource.org/), and other sources
- **nuccore**: Nucleotide sequences including genomic DNA, mRNA, and other nucleotide records
- **gene**: Gene records with summaries, genomic context, and cross-references

**Why is this important?**
- Gene engineering: retrieve wild-type sequences as starting points for design
- Codon optimization: fetch coding sequences (CDS) to analyze natural codon usage
- Comparative genomics: retrieve sequences from multiple organisms for alignment and analysis
- Regulatory element design: fetch genomic regions upstream/downstream of genes
- Literature-linked sequences: NCBI entries link to [PubMed](https://pubmed.ncbi.nlm.nih.gov/) references and functional annotations

**Scientific foundation:**
NCBI's Entrez [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) provide programmatic access to over 40 interconnected databases. The three tools here cover the most common workflow: search for identifiers (ESearch), get metadata (ESummary), and download sequences (EFetch). NCBI search syntax supports field-specific queries (e.g., `[Gene Name]`, `[Organism]`) and Boolean operators for precise retrieval.

## Tools

### NCBI Entrez EFetch (`ncbi-efetch`)

Fetch sequences from NCBI Entrez databases.

Retrieves FASTA records by accession or ID with optional subsequence
extraction.

### NCBI Entrez ESearch (`ncbi-esearch`)

Search NCBI Entrez databases by query term.

### NCBI Entrez ESummary (`ncbi-esummary`)

Retrieve record summary metadata from NCBI Entrez.

## How It Works

**Method overview:**
Each tool wraps one NCBI Entrez E-utility endpoint:
1. **ESearch**: Posts a search query to the specified database and returns matching NCBI IDs
2. **ESummary**: Fetches document summaries (metadata) for a given ID from the specified database
3. **EFetch**: Downloads full records (sequences in FASTA format) for a given ID

All tools include automatic retry with exponential backoff and optional NCBI API key support for higher rate limits.

**Key assumptions:**
- Network access to eutils.ncbi.nlm.nih.gov is available
- Search terms follow NCBI query syntax (field tags like `[Gene Name]`, `[Organism]`)
- Accessions/IDs are valid for the specified database

**Limitations:**
- Rate-limited: 3 requests/second without API key, 10 requests/second with API key
- Large batch retrieval may require pagination (use `max_results` in ESearch)
- EFetch returns FASTA format only (no GenBank, XML, or other formats currently)
- Gene database results differ structurally from protein/nuccore results

**Computational requirements:**
- **Hardware:** CPU only, network access required
- **Runtime:** 1-5 seconds per query (depends on network latency and NCBI load)
- **Scalability:** Sequential queries; for batch retrieval, chain ESearch → EFetch

## Input Parameters

### `NCBIEsearchInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `NCBIDatabase` | *required* | NCBI Entrez database (27 supported: `protein`, `nuccore`, `gene`, `pubmed`, `pmc`, `taxonomy`, `structure`, `snp`, `clinvar`, `genome`, `medgen`, `pccompound`, ...) |
| `search_term` | `str` | *required* | NCBI search query (e.g., `lacI[Gene Name] AND Escherichia coli[Organism]`) |
| `max_results` | `int` | `20` | Maximum number of IDs to return (1-10000; NCBI `retmax`). Default of 20 matches the NCBI API default. |
| `retstart` | `int` | `0` | Sequential index of the first hit to return (NCBI `retstart`, 0-indexed). Use with `max_results` to paginate through large result sets. |
| `sort` | `str \| None` | `None` | Db-specific sort key (e.g. `pub_date`, `relevance` on pubmed). |
| `field` | `str \| None` | `None` | Restrict the search term to a single index field (db-dependent — e.g. `title`, `author` on pubmed). |
| `datetype` | `Literal["mdat","pdat","edat"] \| None` | `None` | Date axis for `mindate`/`maxdate`/`reldate`: modification / publication / Entrez. |
| `mindate` | `str \| None` | `None` | Lower date bound (`YYYY/MM/DD`, `YYYY/MM`, or `YYYY`); requires `datetype`. |
| `maxdate` | `str \| None` | `None` | Upper date bound; requires `datetype`. |
| `reldate` | `int \| None` | `None` | Records dated within the last N days; requires `datetype`. |

### `NCBIEsummaryInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `NCBIDatabase` | *required* | NCBI Entrez database (27 supported: `protein`, `nuccore`, `gene`, `pubmed`, `pmc`, `taxonomy`, `structure`, `snp`, `clinvar`, `genome`, `medgen`, `pccompound`, ...) |
| `identifier` | `str` | *required* | Accession or NCBI ID (e.g., `NP_000537.3`, `7157`) |

### `NCBIEfetchInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `NCBISequenceDatabase` | *required* | Sequence database only: `protein`, `nuccore`, or `nucleotide` (other dbs accept the FASTA rettype but return empty bodies — silently). |
| `identifier` | `str` | *required* | Accession or NCBI ID |
| `return_format` | `Literal["fasta", "fasta_cds_na"]` | `"fasta"` | NCBI rettype: `fasta` for full sequences, `fasta_cds_na` for coding DNA sequences |
| `seq_start` | `Optional[int]` | `None` | Start position for subsequence extraction (1-indexed, inclusive) |
| `seq_stop` | `Optional[int]` | `None` | Stop position for subsequence extraction (1-indexed, inclusive) |
| `strand` | `Optional[Literal["+", "-"]]` | `None` | Strand for nucleotide retrieval (`"+"` for sense, `"-"` for antisense). Nucleotide-only. |

### Sweep Priorities

For sequence retrieval workflows:
1. **`return_format`**: Most impactful choice. Use `"fasta"` for full sequences, `"fasta_cds_na"` specifically for coding DNA sequences (required for codon optimization analysis).
2. **`max_results`** (ESearch): Control breadth of search. Default of 20 matches the NCBI API default; raise up to 10000 for comprehensive surveys.

## Configuration

All three tools share `NCBIFetchConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ncbi_api_key` | `Optional[str]` | `None` | NCBI API key for higher rate limits (10 req/sec vs 3 req/sec) |
| `ncbi_email` | `Optional[str]` | `None` | Contact email; pair with API key for IP-block recovery |

## Output Specification

### `NCBIEsearchOutput`

| Field | Type | Description |
|-------|------|-------------|
| `ids` | `List[str]` | List of NCBI IDs matching the search query |

### `NCBIEsummaryOutput`

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `Dict[str, Any]` | Record summary data (structure varies by database) |
| `source_url` | `str` | Sanitized request URL (API key redacted) |

### `NCBIEfetchOutput`

| Field | Type | Description |
|-------|------|-------------|
| `fasta_records` | `List[NCBIFastaRecord]` | Parsed FASTA records with header, sequence, and accession |
| `source_url` | `str` | Sanitized request URL |

### `NCBIFastaRecord`

| Field | Type | Description |
|-------|------|-------------|
| `header` | `str` | Full FASTA header line |
| `sequence` | `str` | Sequence string (amino acids or nucleotides) |
| `accession` | `Optional[str]` | Accession extracted from header (e.g., `NP_000537.3`) |

**Supported export formats:** `json`, `csv` (efetch also `fasta`)

## Interpreting Results

**ESearch results:**
- Empty `ids` list means no matches found. Check search term syntax and try broader queries.
- IDs can be NCBI GI numbers or accessions depending on the database.
- Results are ranked by relevance by default.

**ESummary results:**
- The `summary` dict structure varies by database:
  - **gene**: Contains `Name`, `Description`, `Summary`, `Organism`, `GeneticSource`, etc.
  - **protein/nuccore**: Contains `Title`, `AccessionVersion`, `Organism`, `Length`, etc.

**EFetch results:**
- Each `NCBIFastaRecord` contains one sequence with its header.
- Multiple records may be returned if the ID maps to multiple sequences.
- `fasta_cds_na` returns coding DNA even from a protein accession (by resolving to the genomic record).

**Interpreting edge cases:**
- Gene IDs (e.g., `7157` for TP53) are numeric and database-specific: they are not accessions
- Some accessions resolve to multiple FASTA records (e.g., alternatively spliced transcripts)
- Subsequence extraction (`seq_start`/`seq_stop`) is 1-indexed and inclusive on both ends, following NCBI conventions
- Fetching with strand `"-"` returns the reverse complement of the specified region

## Quick Start Examples

**Example 1: Search and fetch workflow**
```python
from proto_tools.tools.database_retrieval import (
    NCBIFetchConfig, NCBIEsearchInput, NCBIEfetchInput,
    run_ncbi_esearch, run_ncbi_efetch,
)

# Step 1: Search for lacI in E. coli
search_result = run_ncbi_esearch(
    NCBIEsearchInput(
        db="protein",
        search_term="lacI[Gene Name] AND Escherichia coli[Organism]",
        max_results=5,
    ),
    NCBIFetchConfig(),
)
print(f"Found IDs: {search_result.ids}")

# Step 2: Fetch the top result
if search_result.ids:
    fetch_result = run_ncbi_efetch(
        NCBIEfetchInput(db="protein", identifier=search_result.ids[0]),
        NCBIFetchConfig(),
    )
    record = fetch_result.fasta_records[0]
    print(f"Accession: {record.accession}")
    print(f"Length: {len(record.sequence)} aa")
    print(f"Sequence: {record.sequence[:50]}...")
```

**Example 2: Fetch protein by accession**
```python
from proto_tools.tools.database_retrieval import (
    NCBIFetchConfig, NCBIEfetchInput, run_ncbi_efetch,
)

# Fetch TP53 protein sequence
result = run_ncbi_efetch(
    NCBIEfetchInput(db="protein", identifier="NP_000537.3", return_format="fasta"),
    NCBIFetchConfig(),
)
print(f"{result.fasta_records[0].accession}: {len(result.fasta_records[0].sequence)} aa")
```

**Example 3: Get gene summary**
```python
from proto_tools.tools.database_retrieval import (
    NCBIFetchConfig, NCBIEsummaryInput, run_ncbi_esummary,
)

# Get summary for TP53 gene
result = run_ncbi_esummary(
    NCBIEsummaryInput(db="gene", identifier="7157"),
    NCBIFetchConfig(),
)
summary = result.summary
print(f"Gene: {summary.get('Name')}")
print(f"Description: {summary.get('Description')}")
```

**Example 4: Subsequence extraction**
```python
from proto_tools.tools.database_retrieval import (
    NCBIFetchConfig, NCBIEfetchInput, run_ncbi_efetch,
)

# Fetch a specific genomic region (e.g., promoter region)
result = run_ncbi_efetch(
    NCBIEfetchInput(
        db="nuccore",
        identifier="NC_000017.11",  # Human chromosome 17
        seq_start=7668402,
        seq_stop=7669502,
        strand="+",
    ),
    NCBIFetchConfig(),
)
region = result.fasta_records[0].sequence
print(f"Region length: {len(region)} bp")
```

**Example 5: Chained workflow -- NCBI gene -> RefSeq mRNA -> CDS for codon optimization**

Mirrors the `mrna_cds_optimization` design eval: resolve a gene by symbol, retrieve its canonical RefSeq mRNA, then pull just the CDS for use as a codon-optimization template.

```python
from proto_tools.tools.database_retrieval import (
    NCBIEfetchInput, NCBIEsearchInput, NCBIFetchConfig,
    run_ncbi_efetch, run_ncbi_esearch,
)

# 1. Search RefSeq for the canonical human BRCA1 mRNA
search = run_ncbi_esearch(
    NCBIEsearchInput(
        db="nuccore",
        search_term='BRCA1[Gene Name] AND "Homo sapiens"[Organism] AND refseq[Filter] AND mRNA[Filter]',
        max_results=1,
    ),
    NCBIFetchConfig(),
)
assert search.ids, "no RefSeq mRNA hits"

# 2. Pull just the CDS from that mRNA (skips UTRs)
cds = run_ncbi_efetch(
    NCBIEfetchInput(db="nuccore", identifier=search.ids[0], return_format="fasta_cds_na"),
    NCBIFetchConfig(),
)
record = cds.fasta_records[0]
print(f"BRCA1 CDS: {record.accession}, {len(record.sequence)} nt")
assert len(record.sequence) % 3 == 0  # CDS must be a complete reading frame
```

**Example 6: Canonical Entrez triple chain -- esearch -> esummary -> efetch**

The textbook NCBI workflow: discover IDs with `esearch`, screen them with `esummary`, then pull full records with `efetch`. Useful when you need to inspect record metadata before paying the cost of fetching the full sequence.

```python
from proto_tools.tools.database_retrieval import (
    NCBIEfetchInput, NCBIEsearchInput, NCBIEsummaryInput, NCBIFetchConfig,
    run_ncbi_efetch, run_ncbi_esearch, run_ncbi_esummary,
)

cfg = NCBIFetchConfig()

# 1. esearch: find candidate RefSeq mRNA records for human BRCA1.
search = run_ncbi_esearch(
    NCBIEsearchInput(
        db="nuccore",
        search_term='BRCA1[Gene Name] AND "Homo sapiens"[Organism] AND refseq[Filter] AND mRNA[Filter]',
        max_results=5,
    ),
    cfg,
)
assert search.ids, "no hits"

# 2. esummary: cheaply inspect titles/lengths to pick the right record before fetching the sequence.
summaries = run_ncbi_esummary(
    NCBIEsummaryInput(db="nuccore", identifier=",".join(search.ids)),
    cfg,
)
canonical_id = max(
    search.ids,
    key=lambda i: int(summaries.summary.get(i, {}).get("slen", 0)),  # longest mRNA
)

# 3. efetch: pull the full FASTA for the chosen ID.
fasta = run_ncbi_efetch(
    NCBIEfetchInput(db="nuccore", identifier=canonical_id, return_format="fasta"),
    cfg,
)
record = fasta.fasta_records[0]
print(f"Picked {record.accession} ({len(record.sequence)} nt) from {len(search.ids)} esearch hits")
```

**Example 7: Chained workflow -- NCBI gene summary -> UniProt protein record**

Bridge from organism-genome data (NCBI) to protein-centric data (UniProt).

```python
from proto_tools.tools.database_retrieval import (
    NCBIEsummaryInput, NCBIFetchConfig,
    UniProtFetchConfig, UniProtFetchInput,
    run_ncbi_esummary, run_uniprot_fetch,
)

# 1. NCBI esummary returns the full ESummary payload. For gene records the
#    structure is `{"uids": ["<id>"], "<id>": {"name": "...", ...}}`, so the
#    canonical symbol lives at summary[identifier]["name"].
gene_id = "7157"  # NCBI Gene ID for human TP53
gene_summary = run_ncbi_esummary(
    NCBIEsummaryInput(db="gene", identifier=gene_id),
    NCBIFetchConfig(),
)
gene_symbol = gene_summary.summary[gene_id]["name"]  # "TP53"

# 2. UniProt: protein-centric record for the same gene
uniprot = run_uniprot_fetch(
    UniProtFetchInput(target_name=gene_symbol, organism="Homo sapiens", prefer_pdb_crossref=True),
    UniProtFetchConfig(),
)
print(f"NCBI gene {gene_id} ({gene_symbol}) -> UniProt {uniprot.accession}, {uniprot.length} aa")
print(f"  Linked PDB structures: {len(uniprot.pdb_crossrefs)}")
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Not using an API key:** Without an API key, NCBI rate-limits to 3 requests/second. Set `ncbi_api_key` in config for 10 req/sec. Get a free key at https://www.ncbi.nlm.nih.gov/account/settings/
2. **Using wrong database for the ID:** Protein accessions (e.g., NP_000537.3) won't work with `db="nuccore"`. Match the database to the accession type.
3. **Forgetting 1-based indexing:** `seq_start` and `seq_stop` are 1-indexed and inclusive on both ends. Position 1 is the first nucleotide/amino acid.
4. **Confusing ESearch IDs with accessions:** ESearch returns NCBI internal IDs (which may be numeric GI numbers), not always standard accessions. Use EFetch to resolve to FASTA records with proper accessions.

**Tips for optimal results:**
- Use NCBI field tags in search terms for precision: `[Gene Name]`, `[Organism]`, `[Protein Name]`, `[Title]`
- Chain ESearch → EFetch for the standard workflow: find IDs first, then retrieve sequences
- Provide `ncbi_email` in config: NCBI recommends this for responsible API usage
- For coding sequences, use `return_format="fasta_cds_na"` instead of manually extracting CDS from genomic records

**Edge cases to watch for:**
- Gene database IDs are numeric (e.g., `7157`) and differ from protein/nuccore accessions
- Some accessions are versioned (e.g., `NP_000537.3`): omitting the version may return the latest version
- Obsolete or suppressed records may return empty results or errors
- Very large genomic regions (>100 Mb) may timeout; use subsequence extraction for specific regions

## References

**Primary publication:**
- Sayers, E.W. et al. (2022). "GenBank." *Nucleic Acids Research*, 50(D1), D161-D164. [DOI: 10.1093/nar/gkab1135](https://doi.org/10.1093/nar/gkab1135)
- Summary: Describes NCBI's GenBank database and Entrez retrieval system providing programmatic access to biological sequence data across multiple databases.

**Implementation:**
- NCBI Entrez E-utilities: [https://www.ncbi.nlm.nih.gov/books/NBK25501/](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- NCBI Datasets (newer API): [https://www.ncbi.nlm.nih.gov/datasets/](https://www.ncbi.nlm.nih.gov/datasets/)

## Related Tools

**Tools often used together:**
- **`sequence-fetch`**: Multi-source orchestrator across NCBI, UniProt, and PDB. Use when you want automatic database routing.
- **`blast-search`**: Search for homologs of sequences retrieved from NCBI.
- **`uniprot-fetch`**: Protein-centric queries with curated annotations.

**Alternative tools (similar function):**
- **`uniprot-fetch`**: Better curation for protein-specific queries. Use NCBI for nucleotide, genomic, or gene-level data.
- **`pdb-fetch-entry`** / **`pdb-fetch-fasta`**: PDB-specific metadata and chain sequences.
- **`sequence-fetch`**: Higher-level orchestrator for automatic database selection.
