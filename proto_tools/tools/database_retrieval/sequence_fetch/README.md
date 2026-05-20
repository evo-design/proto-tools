<a href="https://bio-pro.mintlify.app/tools/database-retrieval/sequence-fetch"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Unified Sequence Fetch

> [!NOTE]
> **License:** Unified Sequence Fetch's own code is licensed under Apache-2.0, and it federates over bundled data sources and components, each under its own license terms.
>
> Bundled dependencies, each under its own license:
>
> - [NCBI Entrez](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi): U.S. Government public domain
> - [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot): CC-BY-4.0
> - [RCSB PDB](https://bio-pro.mintlify.app/tools/database-retrieval/pdb): CC0-1.0
>
> Review each source's terms before commercial use or redistribution.

## Overview

The `sequence-fetch` tool is a multi-source orchestrator that resolves a batch of heterogeneous sequence and structure requests into a uniform result. Each request names a gene, protein, or RNA target and the molecule types to retrieve (protein, genomic DNA, coding DNA, transcript RNA, inferred pre-mRNA, or PDB structure), optionally pinned by accession overrides. It federates over [NCBI Entrez](https://www.ncbi.nlm.nih.gov/search/), [UniProt](https://www.uniprot.org/), and [RCSB PDB](https://www.rcsb.org/), returning per-request sequences, structure metadata, resolved identifiers, and status.

## Background

This tool wraps three public databases rather than a single predictive model, so it has no single primary paper. The underlying sources are [GenBank](https://www.ncbi.nlm.nih.gov/genbank/) ([Sayers et al., 2022](https://doi.org/10.1093/nar/gkab1135)), [UniProt](https://www.uniprot.org/) ([The UniProt Consortium, 2025](https://doi.org/10.1093/nar/gkae1010)), and the [RCSB Protein Data Bank](https://www.rcsb.org/) ([Berman et al., 2000](https://doi.org/10.1093/nar/28.1.235)). Internally, `SequenceFetchInput` wraps a `list[SequenceFetchRequest]`, and each request is resolved independently by molecule type with deterministic, priority-based routing where provided identifiers are consulted before a name-and-organism search. For a protein request the routing priority is a supplied UniProt accession (resolved at `rest.uniprot.org`), then an NCBI protein accession, then a linked PDB entry's FASTA chains, and finally a name-and-organism search. Nucleotide requests use the [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) at `https://eutils.ncbi.nlm.nih.gov/entrez/eutils` (esearch, esummary, efetch), with a gene-locus coordinate fallback that fetches the genomic interval directly from the chromosome accession. Structure requests resolve a PDB identifier and read entry metadata from `https://data.rcsb.org`, with chain sequences pulled from `https://www.rcsb.org/fasta/entry`. Every returned record carries a source URL and a SHA256 checksum for provenance, and the NCBI API key and contact email are sanitized out of provenance URLs. Genomic coordinates are interpreted as 1-indexed, inclusive intervals to match biological residue selection conventions. Results reflect the live databases at query time rather than a fixed release snapshot.

### Learning Resources

- [Entrez Programming Utilities help](https://www.ncbi.nlm.nih.gov/books/NBK25501/) (NCBI) - official documentation for the E-utilities API, including esearch, esummary, and efetch.
- [UniProt help and documentation](https://www.uniprot.org/help) (UniProt) - official documentation covering accessions, query syntax, and the REST API.
- [RCSB PDB Data API](https://data.rcsb.org/) (RCSB PDB) - official documentation for the PDB entry data and FASTA endpoints.

## Tools

### Multi-source Sequence Fetch (`sequence-fetch`)

Resolves a list of `SequenceFetchRequest` objects across NCBI Entrez, UniProt, and RCSB PDB, returning per-request fetched sequences, fetched structures, resolved identifiers, warnings, and errors, with run-level counts of successful, warning, and failed requests.

#### Applications

Use this to pull mixed protein, nucleotide, and structure data for many targets in a single call. Resolve a batch of gene symbols plus organisms to reference protein sequences for analysis or design, retrieve coding DNA and transcript RNA alongside the protein for a codon-optimization workflow, or fetch a protein and its linked PDB structures together to seed structure-aware downstream steps. Partial batches are normal. Each request reports its own status, sequences, structures, and errors independently, so one unresolved target does not fail the job.

#### Usage Tips

- **Provide identifiers whenever possible.** Accession overrides such as `uniprot_id`, `genbank_accession`, or `pdb_id` route directly and are far more reliable than a name-and-organism search, which selects a single top-ranked candidate that may be ambiguous.
- **The strand in `genomic_coordinates` changes the returned sequence.** Coordinates are interpreted as 1-indexed, inclusive intervals to match biological residue selection conventions, and an explicit `+` or `-` strand controls whether the forward or reverse-complement sequence is returned. Omitting the strand can yield the wrong-orientation sequence for genes on the minus strand.
- **A protein hit does not guarantee a structure exists.** Structure retrieval requires a linked PDB entry. A UniProt or NCBI protein match with no PDB cross-reference produces a not-found error for the `structure` type while the protein sequence still returns.
- **`type_check_mode` defaults to `"error"`, the right setting for production.** It rejects obvious molecule-type mismatches early, such as requesting `protein` for a name that looks like a non-coding RNA (ncRNA) gene or for an `NR_` or `XR_` RefSeq transcript. `"warn"` records the mismatch and continues, and `"off"` skips the check entirely.
- **`rna_premrna` is inferred, not curated.** It is transcribed from the genomic DNA sequence and includes introns where present, so it is annotation-dependent and not a directly curated transcript.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Sequence Fetch tool in this toolkit (`sequence-fetch`).

- **Requires network access.** The tool federates the live NCBI, UniProt, and RCSB PDB endpoints. It does not run offline and keeps no local copy of any database.
- **An NCBI API key raises the rate limit for NCBI-backed requests.** Requests routed to NCBI E-utilities are limited to 3 requests per second per IP without credentials. Setting credentials raises this to 10 requests per second. A key is obtained at no cost from the Settings page of a free NCBI account (https://www.ncbi.nlm.nih.gov/account/). NCBI also asks that a contact email be set; it uses the email for abuse handling and IP-block recovery. Provide credentials either via the `ncbi_api_key` / `ncbi_email` config attributes or via the `NCBI_API_KEY` / `NCBI_EMAIL` environment variables. An explicit config value overrides the env var. The UniProt and RCSB PDB backends are keyless and have no equivalent mechanism.
