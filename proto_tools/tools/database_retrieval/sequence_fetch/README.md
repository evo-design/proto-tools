<a href="https://bio-pro.mintlify.app/tools/database-retrieval/sequence-fetch"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Unified Sequence Fetch

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

`sequence-fetch` retrieves DNA, RNA, protein, and structure data for named targets across [NCBI](https://www.ncbi.nlm.nih.gov/) Entrez, [UniProt](https://www.uniprot.org/), and [PDB](https://www.rcsb.org/). It supports ID-first resolution, name+organism fallback, and strict molecular type checks.

## Background

**What does this tool retrieve?**
It fetches canonical molecular records from public databases: genomic DNA (`dna_genomic`), coding DNA (`dna_cds`), transcript RNA (`rna_transcript`), inferred pre-mRNA (`rna_premrna`), proteins (`protein`), and PDB structures (`structure`).

**Why is this important?**
- Reproducible sequence retrieval with explicit accession provenance
- ID-aware routing across NCBI, UniProt, and PDB in one interface
- Type-safe validation to avoid invalid requests (for example, protein for ncRNA)
- Batch-friendly results with per-request status and error reporting

**Scientific foundation:**
The tool wraps source APIs rather than using an internal predictive model. NCBI retrieval uses Entrez E-utilities (`esearch`, `efetch`), protein-centric retrieval prefers UniProt when IDs are present, and structure retrieval uses RCSB PDB entry endpoints. Resolution is deterministic and priority-based: provided IDs first, then name+organism fallback.

## Tools

### Multi-source Sequence Fetch (`sequence-fetch`)

Fetch DNA, RNA, protein, and structure records from NCBI, UniProt, and PDB.

This tool resolves IDs and names across NCBI Entrez, UniProt, and PDB for
sequence and structure retrieval.

Routing priority (per request):
    Protein: `uniprot_id` → `protein_id` / `preferred_accession` →
        `pdb_id` → name search.
    Genomic: `genomic_coordinates` → `preferred_accession` →
        name search → gene-locus fallback.
    `additional_ids` is consulted last; the key `"accession"` is
    used as a generic fallback when no typed override is set.

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
| `max_candidates_per_source` | `int` | `5` | Maximum database candidates to evaluate per name-based search. |
| `type_check_mode` | `Literal["off", "warn", "error"]` | `"error"` | How to handle molecule-type mismatches (e.g. requesting "protein" for an ncRNA gene). `"off"` skips validation, `"warn"` logs a warning and continues, `"error"` fails the request. |
| `ncbi_api_key` | `Optional[str]` | `None` | Optional NCBI API key (lifts rate limit from 3 to 10 req/s). |
| `ncbi_email` | `Optional[str]` | `None` | Optional contact email; recommended by NCBI for traceability. |

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
from proto_tools.tools.database_retrieval import (
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

**Chained workflow -- multi-source fetch -> per-source follow-up calls**

`sequence_fetch` *is* the chaining orchestrator -- under the hood it dispatches to
`uniprot-fetch`, `ncbi-efetch`, and `pdb-fetch-entry` -- but downstream you often
want to enrich the result with calls that the orchestrator itself doesn't make.

```python
from proto_tools.tools.database_retrieval import (
    PdbFetchConfig, PdbFetchEntryInput,
    SequenceFetchConfig, SequenceFetchInput,
    run_pdb_fetch_entry, run_sequence_fetch,
)

# 1. Multi-source: pull the protein sequence + linked PDB IDs in one call
result = run_sequence_fetch(
    SequenceFetchInput(
        requests=[{
            "request_id": "kras",
            "target_name": "KRAS",
            "organism": "Homo sapiens",
            "sequence_types": ["protein", "structure"],
        }]
    ),
    SequenceFetchConfig(),
)
kras = result.results[0]

# 2. Follow-up: walk the structure refs and pick the best-resolution X-ray entry.
candidates = []
for s in kras.fetched_structures[:10]:
    meta = run_pdb_fetch_entry(PdbFetchEntryInput(pdb_id=s.pdb_id), PdbFetchConfig())
    if meta.success and meta.resolution is not None and (meta.method or "").startswith("X-RAY"):
        candidates.append((s.pdb_id, meta.resolution))
candidates.sort(key=lambda c: c[1])
print(f"KRAS best X-ray template: {candidates[0]}")
```

## Best Practices & Gotchas

1. Provide IDs whenever possible. ID-first routing is far more reliable than name-only lookup.
2. Include strand with genomic coordinates for better reproducibility.
3. For protein from loci, provide transcript/CDS IDs when possible to avoid isoform ambiguity.
4. Expect partial success in mixed batches and inspect per-request errors instead of failing whole jobs.
5. Use `type_check_mode="error"` (the default) in production to catch ncRNA/protein mismatches early.
6. Iterable-level tool caching is enabled (same pattern as other batch tools) when running in a Program/Optimizer context.
7. A UniProt hit does not guarantee a PDB structure exists; structure fetch requires a linked PDB entry.

## References

- Sayers EW, et al. GenBank. Nucleic Acids Research. 2022;50(D1):D161-D164. doi:10.1093/nar/gkab1135
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research. 2025;53(D1):D609-D617. doi:10.1093/nar/gkae1010
- Burley SK, et al. RCSB Protein Data Bank: Celebrating 50 years of the PDB. Protein Science. 2022;31(1):187-208. doi:10.1002/pro.4213

## Related Tools

- `blast-search`: sequence similarity search after retrieval
- `mmseqs2-search-proteins`: scalable protein homology search
- `prodigal` / `orfipy`: coding region discovery from raw DNA
