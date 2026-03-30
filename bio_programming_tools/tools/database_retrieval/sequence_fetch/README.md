<a href="https://bio-pro.mintlify.app/tools/database-retrieval/sequence-fetch"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Unified Sequence Fetch

## Overview

`sequence-fetch` retrieves DNA, RNA, protein, and structure data for named targets across [NCBI](https://www.ncbi.nlm.nih.gov/) Entrez, [UniProt](https://www.uniprot.org/), and [PDB](https://www.rcsb.org/). It supports ID-first resolution, name+organism fallback, and strict molecular type checks.

## When to Use This Tool

**Primary use cases:**
- Fetching protein, transcript, and genomic sequences for a known target + organism
- Running mixed requests where some entries have IDs and others are name-only
- Building small to medium sequence manifests with per-item provenance and errors
- Pulling structure metadata from PDB using `pdb_id`, UniProt cross-references, or name+organism resolution

**When NOT to use this tool:**
- Large genome-wide extraction workflows: use local FASTA/GFF pipelines
- Precision transcriptome annotation: use [Ensembl](https://www.ensembl.org/)/[GENCODE](https://www.gencodegenes.org/) pipelines
- De novo gene calling: use ORF/gene annotation tools first

**Comparison with alternatives:**
- **`sequence-fetch` vs BLAST (`blast-search`)**: use this tool for record retrieval; use BLAST for similarity search.
- **`sequence-fetch` vs ORF tools (`prodigal`, `orfipy`)**: use this tool for known targets; ORF tools infer coding regions from raw DNA.
- **`sequence-fetch` vs MMseqs2 (`mmseqs-search-proteins`)**: use this tool for canonical sequence fetch; MMseqs2 is for large-scale homology search.

## Biological Background

**What does this tool retrieve?**
It fetches canonical molecular records from public databases: genomic DNA (`dna_genomic`), coding DNA (`dna_cds`), transcript RNA (`rna_transcript`), inferred pre-mRNA (`rna_premrna`), proteins (`protein`), and PDB structures (`structure`).

**Why is this important?**
- Reproducible sequence retrieval with explicit accession provenance
- ID-aware routing across NCBI, UniProt, and PDB in one interface
- Type-safe validation to avoid invalid requests (for example, protein for ncRNA)
- Batch-friendly results with per-request status and error reporting

**Scientific foundation:**
The tool wraps source APIs rather than using an internal predictive model. NCBI retrieval uses Entrez E-utilities (`esearch`, `efetch`), protein-centric retrieval prefers UniProt when IDs are present, and structure retrieval uses RCSB PDB entry endpoints. Resolution is deterministic and priority-based: provided IDs first, then name+organism fallback.

## How It Works

**Method overview:**
1. Validate each request and normalize requested molecule types.
2. Apply type compatibility checks (ncRNA/protein mismatch guardrails).
3. Resolve via ID-first logic (`uniprot_id`, accessions, `pdb_id`, etc.).
4. Fallback to name+organism search where IDs are missing.
5. Return per-item sequences/structures, warnings, errors, and resolved IDs.

For `dna_genomic` name-based lookups, the tool tries multiple `nuccore`
candidates if the first hit has no FASTA payload, then uses the first valid fallback.
If all `nuccore` candidates fail, it falls back to `gene` locus coordinates and
fetches the genomic interval directly from the chromosome accession.

**Key assumptions:**
- Input target names are biologically meaningful in the requested organism.
- Accessions provided by the user are valid for the requested molecule type.
- Source APIs are available and responsive.

**Limitations:**
- Name-only lookups can be ambiguous and may choose top-ranked hits.
- `rna_premrna` is inferred from genomic DNA and is annotation-dependent.
- Coordinate-based protein backtracking is indirect and may map to multiple isoforms.
- Coordinate interpretation depends on strand context.

**Important intron caveat:**
- `dna_genomic` is not directly translatable in eukaryotes.
- Protein retrieval from genomic coordinates is inferred through annotation, not direct translation.
- Alternative splicing can map one locus to multiple proteins.

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `requests` | `List[SequenceFetchRequest]` | One or more retrieval requests; single dict is accepted and auto-wrapped. |

### `SequenceFetchRequest`

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_name` | `str` | Gene/protein/RNA target name. |
| `organism` | `str` | Organism label for disambiguation. |
| `sequence_types` | `List[str]` | Any of: `protein`, `dna_genomic`, `dna_cds`, `rna_transcript`, `rna_premrna`, `structure`. |
| `uniprot_id` | `Optional[str]` | Optional UniProt accession override. |
| `genbank_accession` | `Optional[str]` | Optional GenBank accession override. |
| `refseq_accession` | `Optional[str]` | Optional RefSeq accession override. |
| `pdb_id` | `Optional[str]` | Optional PDB ID override. |
| `gene_id` | `Optional[str]` | Optional NCBI Gene ID override. |
| `protein_id` | `Optional[str]` | Optional NCBI protein accession override. |
| `transcript_id` | `Optional[str]` | Optional transcript accession override. |
| `genomic_coordinates` | `Optional[str]` | Optional coordinates like `NC_000913.3:1-100:+`. |
| `additional_ids` | `Dict[str, str]` | Optional extra IDs for custom routing. |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_timeout_seconds` | `int` | `15` | HTTP timeout in seconds. |
| `http_retries` | `int` | `2` | Retries for upstream HTTP requests. |
| `backoff_seconds` | `float` | `1.0` | Seconds to wait between retries (doubles after each attempt). |
| `max_candidates_per_source` | `int` | `5` | Maximum database candidates to evaluate per name-based search. |
| `strict_type_checks` | `bool` | `True` | Reject requests where molecule type conflicts with target (e.g. protein for an ncRNA gene). |
| `fail_on_type_mismatch` | `bool` | `True` | Treat molecule type mismatches as errors instead of warnings. |
| `include_sequence_checksums` | `bool` | `True` | Include SHA256 checksums per sequence. |
| `ncbi_api_key` | `Optional[str]` | `None` | Optional NCBI API key. |
| `ncbi_email` | `Optional[str]` | `None` | Optional NCBI contact email. |
| `user_agent` | `str` | `"bio-programming-tools/sequence-fetch-v1"` | Identifier string sent to database APIs with each request. |

## Output Specification

`run_sequence_fetch` returns `SequenceFetchOutput`:

- `results`: per-request outcomes (`success`, `warning`, or `failed`)
- `num_success`: pure success count (no warnings)
- `num_warning`: warning count (completed with caveats)
- `num_completed`: `success + warning` count
- `num_failed`: failure count

Each `SequenceFetchResult` includes:
- `fetched_sequences`: sequence records with type/source/accession/checksum
- `fetched_structures`: PDB structure metadata records
- `resolved_ids`: IDs used or resolved during retrieval
- `warnings` and `errors`: explicit diagnostics

**Supported export formats:** `json`, `fasta`

## Interpreting Results

- **`success`**: all requested molecule types fetched without warnings/errors.
- **`warning`**: partial success, ambiguity, or inference caveats (for example, genomic-to-protein caveat).
- **`failed`**: no useful data returned for the request.

Common warning/error patterns:
- `TYPE_MISMATCH`: requested molecule type conflicts with target hints.
- `NOT_FOUND[...]`: source lookup produced no candidate records.

## Quick Start Examples

```python
from bio_programming_tools.tools.database_retrieval import (
    SequenceFetchConfig,
    SequenceFetchInput,
    run_sequence_fetch,
)

inputs = SequenceFetchInput(
    requests=[
        {
            "request_id": "lacI_ecoli",
            "target_name": "lacI",
            "organism": "Escherichia coli",
            "sequence_types": ["protein", "dna_genomic"],
        },
        {
            "request_id": "p53_human",
            "target_name": "TP53",
            "organism": "Homo sapiens",
            "sequence_types": ["protein", "structure"],
        },
    ]
)

config = SequenceFetchConfig()
result = run_sequence_fetch(inputs, config)

print("Requests:", result.num_requests)
print("Completed:", result.num_completed)
print("Success:", result.num_success)
print("Warning:", result.num_warning)
print("Failed:", result.num_failed)
for item in result.results:
    print(item.request_id, item.status, len(item.fetched_sequences), len(item.fetched_structures))
```

```python
# Export to FASTA (all fetched sequences)
result.export(name="sequence_fetch_run", export_path="./outputs", file_format="fasta")
```

## Best Practices & Gotchas

1. Provide IDs whenever possible. ID-first routing is far more reliable than name-only lookup.
2. Include strand with genomic coordinates for better reproducibility.
3. For protein from loci, provide transcript/CDS IDs when possible to avoid isoform ambiguity.
4. Expect partial success in mixed batches and inspect per-request errors instead of failing whole jobs.
5. Use `strict_type_checks=True` in production to catch ncRNA/protein mismatches early.
6. Iterable-level tool caching is enabled (same pattern as other batch tools) when running in a Program/Optimizer context.
7. A UniProt hit does not guarantee a PDB structure exists; structure fetch requires a linked PDB entry.

## References

- Sayers EW, et al. GenBank. Nucleic Acids Research. 2022;50(D1):D161-D164. doi:10.1093/nar/gkab1135
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research. 2025;53(D1):D609-D617. doi:10.1093/nar/gkae1010
- Burley SK, et al. RCSB Protein Data Bank: Celebrating 50 years of the PDB. Protein Science. 2022;31(1):187-208. doi:10.1002/pro.4213

## Related Tools

- `blast-search`: sequence similarity search after retrieval
- `mmseqs-search-proteins`: scalable protein homology search
- `prodigal` / `orfipy`: coding region discovery from raw DNA
