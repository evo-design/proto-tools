# PDB Fetch

## Overview

`pdb-fetch` retrieves structure metadata and chain sequences from the RCSB Protein Data Bank. It supports two operations: fetching entry metadata (title, method, resolution) and fetching FASTA chain sequences with protein/nucleic acid classification.

## When to Use This Tool

**Primary use cases:**
- Fetching PDB entry metadata (experimental method, resolution, title)
- Retrieving chain sequences from PDB FASTA with protein/DNA/RNA classification
- Getting structure information for a known PDB accession

**When NOT to use this tool:**
- Multi-source orchestrated fetches across NCBI, UniProt, and PDB: use `sequence-fetch`
- Structure prediction: use `esmfold`, `boltz2`, or `alphafold3-prediction`
- Finding PDB IDs from gene names: use `sequence-fetch` which resolves via UniProt cross-refs

## How It Works

The tool wraps the RCSB PDB REST API. Entry metadata comes from the core entry endpoint. FASTA chains are fetched from the RCSB FASTA endpoint and classified as protein or nucleic acid based on sequence composition.

## Quick Start

```python
from bio_programming_tools.tools.database_retrieval import (
    PdbFetchConfig,
    PdbFetchInput,
    run_pdb_fetch,
)

# Fetch entry metadata for 1LBG
inputs = PdbFetchInput(pdb_id="1LBG", operation="entry")
output = run_pdb_fetch(inputs, PdbFetchConfig())
print(output.pdb_id, output.title, output.method, output.resolution)

# Fetch FASTA chain sequences
inputs = PdbFetchInput(pdb_id="1LBG", operation="fasta")
output = run_pdb_fetch(inputs, PdbFetchConfig())
for chain in output.chains:
    print(chain.chain_id, chain.entity_type, len(chain.sequence))
```

## References

- Burley SK, et al. RCSB Protein Data Bank: Celebrating 50 years of the PDB. Protein Science. 2022;31(1):187-208. doi:10.1002/pro.4213
