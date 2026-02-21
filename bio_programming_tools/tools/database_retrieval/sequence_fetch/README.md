# Multi-source Sequence Fetch

## Overview

`sequence-fetch` retrieves DNA, RNA, protein, and structure data for named targets across NCBI Entrez, UniProt, and PDB. It supports ID-first resolution, name+organism fallback, strict molecular type checks, and batch mode with adaptive chunk splitting.

## When to Use This Tool

**Primary use cases:**
- Fetching protein, transcript, and genomic sequences for a known target + organism
- Running mixed requests where some entries have IDs and others are name-only
- Building small to medium sequence manifests with per-item provenance and errors
- Pulling structure metadata from PDB using `pdb_id`, UniProt cross-references, or name+organism resolution

**When NOT to use this tool:**
- Large genome-wide extraction workflows: use local FASTA/GFF pipelines
- Precision transcriptome annotation: use Ensembl/GENCODE pipelines
- De novo gene calling: use ORF/gene annotation tools first

**Comparison with granular tools:**
- For single-database operations, use `ncbi-fetch`, `uniprot-fetch`, or `pdb-fetch` directly
- Use `sequence-fetch` when you need multi-source orchestration with fallback chains

## How It Works

1. Validate each request and normalize requested molecule types
2. Apply type compatibility checks (ncRNA/protein mismatch guardrails)
3. Resolve via ID-first logic (accessions, `uniprot_id`, `pdb_id`, etc.)
4. Fallback to name+organism search where IDs are missing
5. Return per-item sequences/structures, warnings, errors, and resolved IDs

Internally delegates to `ncbi-fetch`, `uniprot-fetch`, and `pdb-fetch` for database operations.

## Quick Start

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

config = SequenceFetchConfig(chunk_size=10, chunk_timeout_seconds=45)
result = run_sequence_fetch(inputs, config)

for item in result.results:
    print(item.request_id, item.status, len(item.fetched_sequences))
```

## References

- Sayers EW, et al. GenBank. Nucleic Acids Research. 2022;50(D1):D161-D164. doi:10.1093/nar/gkab1135
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research. 2025;53(D1):D609-D617. doi:10.1093/nar/gkae1010
- Burley SK, et al. RCSB Protein Data Bank: Celebrating 50 years of the PDB. Protein Science. 2022;31(1):187-208. doi:10.1002/pro.4213
