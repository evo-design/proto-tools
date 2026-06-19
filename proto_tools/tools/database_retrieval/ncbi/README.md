<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ncbi"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# NCBI Entrez

![NCBI Entrez](https://proto-bio.github.io/proto-assets/images/tool/ncbi/hero.png)

> [!NOTE]
> **License:** NCBI Entrez retrieves data from NCBI's Entrez databases, in the public domain (U.S. Government public domain). The client wrapper code is MIT-licensed. Please refer to [the data terms](https://www.ncbi.nlm.nih.gov/home/about/policies/) for full terms.

## Overview

[NCBI Entrez](https://www.ncbi.nlm.nih.gov/search/) is the [National Center for Biotechnology Information](https://www.ncbi.nlm.nih.gov/)'s search and retrieval system over its biological sequence databases, accessed through the Entrez Programming Utilities (E-utilities). This toolkit wraps three E-utilities endpoints: `ncbi-esearch` (query term to matching record UIDs), `ncbi-esummary` (document-summary metadata for a UID or accession), and `ncbi-efetch` (full sequence records as parsed FASTA for the protein and nucleotide databases).

## Background

[NCBI Entrez](https://www.ncbi.nlm.nih.gov/search/) and its underlying sequence archive are described in the GenBank report ([Sayers et al., 2022](https://doi.org/10.1093/nar/gkab1135)), published in *Nucleic Acids Research*. The Entrez system and the [E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) are operated by the [National Center for Biotechnology Information](https://www.ncbi.nlm.nih.gov/) (NCBI), part of the [U.S. National Library of Medicine](https://www.nlm.nih.gov/) (NLM). Entrez unifies search and retrieval across more than forty interconnected databases, including the protein, nucleotide, and gene databases used here, with records drawn from sources such as [RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) and [GenBank](https://www.ncbi.nlm.nih.gov/genbank/).

Internally, each tool issues an HTTP GET to the E-utilities base endpoint `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`. `ncbi-esearch` calls `esearch.fcgi` and returns the JSON `idlist`. `ncbi-esummary` calls `esummary.fcgi` and returns the JSON result map. `ncbi-efetch` calls `efetch.fcgi` with a FASTA `rettype` and parses the response into records. Every request carries a fixed `tool=` identifier. The `email=` and `api_key=` parameters are sent only when `ncbi_email` and `ncbi_api_key` are configured. The request URL surfaced on outputs is sanitized so the API key and email are stripped before it is returned. Records and their provenance come directly from NCBI's live E-utilities, so results reflect the database state at query time rather than a fixed release snapshot.

### Learning Resources

- [Entrez Programming Utilities Help](https://www.ncbi.nlm.nih.gov/books/NBK25501/) (NCBI) - the official E-utilities reference covering each endpoint, parameters, and response formats.
- [General usage guidelines and API key information](https://www.ncbi.nlm.nih.gov/books/NBK25497/) (NCBI) - the official guidance on rate limits, the `tool` and `email` parameters, and obtaining an API key.
- [Entrez Help](https://www.ncbi.nlm.nih.gov/books/NBK3837/) (NCBI) - introduction to Entrez databases, search field tags, and query syntax.

## Tools

### NCBI Entrez ESearch (`ncbi-esearch`)

Runs a query term against a chosen Entrez database and returns the list of matching record UIDs, with optional pagination, sort key, single-field restriction, and date filtering on a modification, publication, or Entrez date axis.

#### Applications

Use this as the entry point of an Entrez retrieval pipeline: resolve a gene symbol and organism to candidate protein or nucleotide UIDs, page through a large hit set with `retstart` and `max_results`, or restrict a literature query by date before downstream processing. The returned UIDs feed directly into [`ncbi-esummary`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi) for metadata screening and [`ncbi-efetch`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi) for sequence retrieval, and a resolved accession pairs naturally with the [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot) and [sequence-fetch](https://bio-pro.mintlify.app/tools/database-retrieval/sequence-fetch) tools.

#### Usage Tips

- **The returned IDs are Entrez UIDs, not always accessions.** Depending on the database they may be numeric GI numbers (GenInfo Identifiers). Resolve them through `ncbi-esummary` or `ncbi-efetch` to obtain accession-bearing records.
- **Date bounds need a date axis.** Setting `mindate`, `maxdate`, or `reldate` without `datetype` is rejected, because NCBI silently ignores date filters that lack an axis.
- **`sort` and `field` are database-specific.** A key valid on `pubmed` may be invalid on `protein`. Consult the Entrez help for the database being queried.

### NCBI Entrez ESummary (`ncbi-esummary`)

Retrieves the document summary for a UID or accession from a chosen Entrez database and returns the summary as a database-specific mapping alongside the sanitized request URL.

#### Applications

Use this to screen candidates cheaply before fetching full records: take the UIDs from [`ncbi-esearch`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi), inspect titles, lengths, or organism fields in the summary, and select the canonical record before paying the cost of a sequence download with [`ncbi-efetch`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi). A `gene`-database summary also bridges organism-level data to protein-centric records in the [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot) tool by resolving the canonical gene symbol.

#### Usage Tips

- **The summary shape depends on the database.** A `gene` summary nests fields under the UID key while `protein` and `nuccore` summaries expose record fields directly. Read the structure for the database queried rather than assuming a fixed schema.
- **Multiple identifiers can be summarized in one call.** Passing a comma-joined list of UIDs returns one entry per UID, which is the efficient way to screen a full `ncbi-esearch` hit set.
- **A missing record raises rather than returning empty.** An unresolved database-and-identifier pair raises an error, so guard identifiers that may be obsolete or suppressed.

### NCBI Entrez EFetch (`ncbi-efetch`)

Fetches full sequence records by UID or accession from the protein, nuccore, or nucleotide databases, returning parsed FASTA records and the sanitized request URL, with optional subsequence and strand selection.

#### Applications

Use this to pull reference sequences into a design or analysis pipeline: retrieve a wild-type protein before sequence design, fetch a coding DNA sequence with `return_format="fasta_cds_na"` for codon-usage analysis, or extract a defined genomic region for regulatory-element work. It is the final stage of the canonical Entrez chain, consuming UIDs produced by [`ncbi-esearch`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi) and screened with [`ncbi-esummary`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi), and complements the [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot) and [sequence-fetch](https://bio-pro.mintlify.app/tools/database-retrieval/sequence-fetch) tools for cross-source retrieval.

#### Usage Tips

- **Restricted to sequence databases.** Only `protein`, `nuccore`, and `nucleotide` return sequence bodies. Metadata databases require `ncbi-esummary` instead.
- **`fasta_cds_na` requires a nucleotide database.** This return format extracts coding DNA and is rejected for `db="protein"`, since CDS extraction has no meaning on a protein record.
- **Subsequence coordinates are 1-indexed and inclusive on both ends, to match biological residue selection conventions.** Position 1 is the first residue, and `seq_stop` is included in the returned span.
- **Strand `"-"` returns the reverse complement.** Antisense retrieval applies to nucleotide databases. It returns the reverse complement of the requested region.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every NCBI tool in this toolkit (`ncbi-esearch`, `ncbi-esummary`, `ncbi-efetch`).

- **Requires network access.** The tools call the live NCBI E-utilities online.
- **An NCBI API key raises the rate limit.** Without credentials, NCBI E-utilities permits 3 requests per second per IP. Setting credentials raises this to 10 requests per second. A key is obtained at no cost from the Settings page of a free NCBI account (https://www.ncbi.nlm.nih.gov/account/). NCBI also asks that a contact email be set; it uses the email for abuse handling and IP-block recovery. Provide credentials either via the `ncbi_api_key` / `ncbi_email` config attributes or via the `NCBI_API_KEY` / `NCBI_EMAIL` environment variables; an explicit config value overrides the env var.
