# UniProt Fetch

## Overview

`uniprot-fetch` retrieves protein entries from UniProt by accession ID or by searching with target name and organism. It returns the protein sequence, gene names, and PDB cross-references.

## When to Use This Tool

**Primary use cases:**
- Fetching a protein sequence by UniProt accession (e.g. P04637)
- Searching UniProt by gene name + organism to resolve accessions
- Extracting PDB cross-references from UniProt entries

**When NOT to use this tool:**
- Multi-source orchestrated fetches across NCBI, UniProt, and PDB: use `sequence-fetch`
- DNA or RNA sequence retrieval: use `ncbi-fetch`
- PDB structure metadata: use `pdb-fetch`

## How It Works

The tool wraps the UniProt REST API. Entry lookups fetch directly by accession. Name-based searches use the UniProt search endpoint with organism filtering, then rank results by review status and gene name match quality.

## Quick Start

```python
from bio_programming_tools.tools.database_retrieval import (
    UniProtFetchConfig,
    UniProtFetchInput,
    run_uniprot_fetch,
)

# Fetch TP53 by accession
inputs = UniProtFetchInput(uniprot_id="P04637")

output = run_uniprot_fetch(inputs, UniProtFetchConfig())
print(output.accession, output.sequence[:20], output.length)
```

## References

- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research. 2025;53(D1):D609-D617. doi:10.1093/nar/gkae1010
