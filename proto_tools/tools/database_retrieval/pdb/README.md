<a href="https://bio-pro.mintlify.app/tools/database-retrieval/pdb"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# PDB

## Overview

Two tools for retrieving data from the RCSB Protein Data Bank:

- **`pdb-fetch-entry`**: Fetch structure metadata (title, experimental method, resolution)
- **`pdb-fetch-fasta`**: Fetch chain sequences with automatic protein/nucleotide classification

Both are CPU-only tools that wrap the RCSB PDB REST API.

## Tool Catalog

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `pdb-fetch-entry` | PDB ID | Title, method, resolution | Get structure metadata to assess quality |
| `pdb-fetch-fasta` | PDB ID | Chain sequences + classification | Extract protein/nucleotide sequences from a structure |

## Background

**What does this tool measure/predict?**
These tools retrieve information from the [RCSB Protein Data Bank](https://www.rcsb.org/), the global archive for experimentally determined 3D structures of biological macromolecules. PDB entries contain structures solved by [X-ray crystallography](https://en.wikipedia.org/wiki/X-ray_crystallography), [cryo-EM](https://en.wikipedia.org/wiki/Cryogenic_electron_microscopy), [NMR spectroscopy](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy_of_proteins), and other methods.

**Why is this important?**
- Structure quality assessment: resolution and experimental method indicate reliability of atomic coordinates
- Protein design: extract reference sequences from specific chains of experimental structures
- Structural analysis: identify which chains are protein vs nucleic acid in multi-component complexes
- Benchmarking: retrieve experimental structures to compare against computational predictions

**Scientific foundation:**
The PDB was established in 1971 and now contains over 220,000 structures. Key metadata:
- **Resolution** (for X-ray/cryo-EM): lower values indicate higher-quality atomic models (1.0-2.0 A is considered high resolution)
- **Experimental method**: X-ray crystallography (most entries), cryo-EM (growing), NMR (small proteins), neutron diffraction (rare)
- **Chain classification**: protein chains contain amino acids; nucleotide chains contain DNA/RNA bases

## How It Works

**Method overview:**
- `pdb-fetch-entry` queries the RCSB core entry API (`/entry/{pdb_id}`) for structure metadata
- `pdb-fetch-fasta` queries the RCSB FASTA API (`/fasta/entry/{pdb_id}`) for chain sequences, then classifies each chain as protein or nucleic acid based on amino acid composition

**Chain classification logic:** A chain is classified as protein if it contains amino acid residues specific to proteins (F, W, Y, H, E, D, K, R, etc.). Chains composed purely of A, T/U, G, C are classified as nucleotide.

**Key assumptions:**
- PDB IDs are valid 4-character identifiers (case-insensitive, automatically uppercased)
- Network access to data.rcsb.org is available

**Limitations:**
- Returns metadata and sequences only, not 3D coordinates or full PDB files
- Chain classification is heuristic-based; unusual sequences (e.g., peptide nucleic acids) may be misclassified
- Some PDB entries may have incomplete metadata (e.g., NMR structures lack resolution)

**Computational requirements:**
- **Hardware:** CPU only, network access required
- **Runtime:** 1-3 seconds per query
- **Scalability:** Sequential queries; loop for batch retrieval

## Input Parameters

### `PdbFetchEntryInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_id` | `str` | *required* | PDB accession (e.g., `"1LBG"`, `"6VXX"`). Case-insensitive. |

### `PdbFetchFastaInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_id` | `str` | *required* | PDB accession (e.g., `"1LBG"`, `"6VXX"`). Case-insensitive. |

## Configuration

Both tools share `PdbFetchConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_timeout_seconds` | `int` | `15` | HTTP timeout per request |
| `http_retries` | `int` | `3` | Max HTTP retries |
| `backoff_seconds` | `float` | `1.0` | Initial wait between retries (doubles after each attempt) |
| `user_agent` | `str` | `"proto-tools/pdb-fetch-v1"` | Identifier string sent with each request |

## Output Specification

### `PdbFetchEntryOutput`

```python
PdbFetchEntryOutput(
    title: Optional[str],       # Structure title
    method: Optional[str],      # Experimental method (e.g., "X-RAY DIFFRACTION")
    resolution: Optional[float], # Resolution in Angstroms (lower = better)
    source_url: Optional[str],  # RCSB API URL
)
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | `Optional[str]` | Structure title from PDB entry |
| `method` | `Optional[str]` | Experimental method (X-RAY DIFFRACTION, ELECTRON MICROSCOPY, NMR, etc.) |
| `resolution` | `Optional[float]` | Resolution in Angstroms. None for NMR structures. |
| `source_url` | `Optional[str]` | RCSB API request URL |

### `PdbFetchFastaOutput`

```python
PdbFetchFastaOutput(
    chains: List[PdbChain],     # Parsed chain sequences
    source_url: Optional[str],  # RCSB API URL
)
```

### `PdbChain`

| Field | Type | Description |
|-------|------|-------------|
| `chain_id` | `Optional[str]` | Chain identifier from FASTA header (e.g., "A", "B") |
| `header` | `str` | Full FASTA header line |
| `sequence` | `str` | Chain amino acid or nucleotide sequence |
| `is_protein` | `bool` | True if protein chain, False if nucleic acid |

**Supported export formats:** `json`

## Interpreting Results

**Resolution quality tiers (X-ray/cryo-EM):**
- **High resolution:** `< 2.0 A`: Atomic detail visible; reliable for side-chain conformations and ligand binding
- **Good resolution:** `2.0 - 3.0 A`: Backbone well-defined; side-chain positions approximate
- **Low resolution:** `3.0 - 4.0 A`: Backbone trace visible; individual side chains unreliable
- **Very low resolution:** `> 4.0 A`: Domain-level topology only; use with caution

**Method considerations:**
- X-ray crystallography: resolution directly indicates coordinate quality
- Cryo-EM: resolution varies by region; overall resolution may not reflect local quality
- NMR: no resolution value (set to None); typically limited to proteins <30 kDa

**Interpreting edge cases:**
- Multiple chains with the same sequence indicate a homo-oligomer (e.g., homodimer has chains A and B with identical sequences)
- Very short chains may be peptide ligands rather than structural subunits
- Missing resolution (None) typically indicates an NMR structure

## Quick Start Examples

**Example 1: Fetch entry metadata**
```python
from proto_tools.tools.database_retrieval import (
    PdbFetchConfig, PdbFetchEntryInput, run_pdb_fetch_entry,
)

inputs = PdbFetchEntryInput(pdb_id="1LBG")
output = run_pdb_fetch_entry(inputs, PdbFetchConfig())

print(f"Title: {output.title}")
print(f"Method: {output.method}")
print(f"Resolution: {output.resolution} A")
```

**Example 2: Fetch chain sequences with classification**
```python
from proto_tools.tools.database_retrieval import (
    PdbFetchConfig, PdbFetchFastaInput, run_pdb_fetch_fasta,
)

inputs = PdbFetchFastaInput(pdb_id="6VXX")  # SARS-CoV-2 spike protein
output = run_pdb_fetch_fasta(inputs, PdbFetchConfig())

for chain in output.chains:
    chain_type = "protein" if chain.is_protein else "nucleotide"
    print(f"Chain {chain.chain_id}: {chain_type}, {len(chain.sequence)} residues")
    print(f"  Sequence: {chain.sequence[:50]}...")
```

**Example 3: Get protein sequences from a complex for design**
```python
from proto_tools.tools.database_retrieval import (
    PdbFetchConfig, PdbFetchFastaInput, run_pdb_fetch_fasta,
)

output = run_pdb_fetch_fasta(
    PdbFetchFastaInput(pdb_id="4OO8"),  # Cas9 complex
    PdbFetchConfig(),
)

# Extract only protein chains
protein_chains = [c for c in output.chains if c.is_protein]
print(f"Found {len(protein_chains)} protein chains:")
for chain in protein_chains:
    print(f"  Chain {chain.chain_id}: {len(chain.sequence)} residues")

# Get unique protein sequences (deduplicate homo-oligomer chains)
unique_seqs = list(set(c.sequence for c in protein_chains))
print(f"Unique protein sequences: {len(unique_seqs)}")
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Assuming PDB IDs are case-sensitive:** Both tools automatically uppercase PDB IDs. "1lbg" and "1LBG" are equivalent.
2. **Confusing `pdb-fetch-entry` with structure download:** `pdb-fetch-entry` returns metadata (title, method, resolution), not PDB coordinate files.
3. **Expecting chain classification to handle all edge cases:** The protein/nucleotide classification is heuristic. Peptide nucleic acids (PNAs) or other hybrid molecules may be misclassified.
4. **Assuming all entries have resolution:** NMR structures do not have resolution values; the field will be None.

**Tips for optimal results:**
- Use `pdb-fetch-entry` first to check resolution and method before deciding whether to use a structure as a reference
- For multi-chain structures, iterate over `output.chains`: a single PDB entry may contain many chains
- Combine with `uniprot-fetch` to go from gene name → UniProt accession → PDB IDs → chain sequences

**Edge cases to watch for:**
- Entries with very many chains (e.g., ribosome structures with 50+ chains) will return all chains
- Obsolete PDB entries may return errors or redirect to superseding entries
- Some PDB entries contain only nucleic acids (no protein chains)

## References

**Primary publication:**
- Burley, S.K. et al. (2022). "RCSB Protein Data Bank: Celebrating 50 years of the PDB." *Protein Science*, 31(1), 187-208. [DOI: 10.1002/pro.4213](https://doi.org/10.1002/pro.4213)
- Summary: Describes the RCSB PDB as the global archive for experimentally determined 3D structures of biological macromolecules, with over 200,000 entries.

**Implementation:**
- RCSB PDB: [https://www.rcsb.org](https://www.rcsb.org)
- RCSB Data API: [https://data.rcsb.org](https://data.rcsb.org)

## Related Tools

**Tools often used together:**
- **`uniprot-fetch`**: Get UniProt accessions and PDB cross-references, then fetch specific PDB chains
- **`sequence-fetch`**: Multi-source orchestrator that can route to PDB automatically

**Alternative tools (similar function):**
- **`ncbi-esummary`** / **`ncbi-efetch`**: NCBI Entrez tools for retrieving sequences and metadata. Broader scope but less structure-specific.
- **`sequence-fetch`**: Higher-level orchestrator for automatic database routing.
