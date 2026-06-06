<a href="https://bio-pro.mintlify.app/tools/database-retrieval/uniprot"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# UniProt

![UniProt](https://proto-bio.github.io/proto-assets/images/tool/uniprot/hero.png)

> [!NOTE]
> **License:** UniProt has a CC-BY-4.0 license and may require explicit attribution when utilized. Please refer to [the license](https://www.uniprot.org/help/license) for full terms.

## Overview

UniProt (the Universal Protein Resource) is the reference database of protein sequences and their functional annotation, maintained by the UniProt consortium. The `uniprot-fetch` tool retrieves a UniProtKB entry over the UniProt REST API, either directly by accession or through a ranked gene- or protein-name and organism search, returning the sequence, gene names, review status, linked PDB structures, and the full JSON record. It runs on CPU and requires only network access.

## Background

UniProt ([The UniProt Consortium, 2025](https://doi.org/10.1093/nar/gkae1010)) is the central, freely accessible resource for protein sequence and functional annotation, maintained by [SIB](https://www.sib.swiss/), [EMBL-EBI](https://www.ebi.ac.uk/), and [PIR](https://proteininformationresource.org/). Its core database, UniProtKB, has two sections: [Swiss-Prot](https://en.wikipedia.org/wiki/UniProt#UniProtKB/Swiss-Prot), whose entries are manually reviewed and curated from the literature, and [TrEMBL](https://en.wikipedia.org/wiki/UniProt#UniProtKB/TrEMBL), whose entries are automatically annotated. As of release 2026_01 (January 2026), Swiss-Prot contains 574,627 reviewed entries, alongside hundreds of millions of unreviewed TrEMBL entries; current counts are published on the [UniProt statistics page](https://www.uniprot.org/uniprotkb/statistics).

Internally, the tool calls the UniProt REST API at `rest.uniprot.org`. Given an accession it fetches that UniProtKB entry directly; given a protein or gene name and an organism it runs a UniProt search and selects one entry deterministically, preferring an exact gene-name match, then optionally entries with linked PDB structures, then reviewed Swiss-Prot status. It extracts the sequence, length, review status, gene symbols, and PDB cross-references, and also returns the complete entry JSON; the `fields` option narrows the API response. Results reflect the live database at query time rather than a fixed release snapshot.

Records and their provenance come directly from UniProt's official REST API, maintained by the UniProt consortium.

### Learning Resources

- [UniProt help and documentation](https://www.uniprot.org/help) (UniProt) - official documentation covering accessions, query syntax, return fields, and the REST API.
- [Exploring protein sequence and functional information](https://www.ebi.ac.uk/training/online/courses/uniprot-exploring-protein-sequence-and-functional-info/) (EMBL-EBI Training) - a guided introduction to UniProt's data and how to query it.

## Tools

### UniProt Fetch (`uniprot-fetch`)

Retrieves a single UniProtKB entry, either by accession or by a ranked name-and-organism search, and returns its sequence, length, gene names, review status, PDB cross-references, source URL, and the full entry JSON.

#### Applications

Use this to pull a reference protein sequence and its annotation into a pipeline: fetch a target by accession before sequence design or optimization, resolve a gene symbol plus organism to a canonical reviewed entry, or discover which experimental structures are linked to a protein before structure-based work. The returned PDB identifiers feed directly into the [PDB](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) and [AlphaFold DB](https://bio-pro.mintlify.app/tools/database-retrieval/alphafold-db) tools.

#### Usage Tips

- **Provide either `uniprot_id` or both `target_name` and `organism`.** An accession does a direct lookup; a name requires the organism to disambiguate, and the search returns the single best-ranked entry, not a list.
- **`prefer_pdb_crossref` only affects search ranking.** It biases the name-and-organism search toward entries with linked PDB structures; it has no effect on a direct accession lookup and never filters out entries that lack structures.
- **`fields` narrows the response but can blank typed outputs.** Restricting `fields` shrinks large entries substantially, but the typed outputs are only populated when their source fields are kept, so include `accession`, `sequence`, `reviewed`, `gene_names`, and `xref_pdb` if you read those.
- **Results track the live database.** The same call can return updated annotation as UniProt releases change; it is not pinned to a fixed release.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every UniProt tool in this toolkit (`uniprot-fetch`).

- **Requires network access.** The tool calls the live UniProt REST API; it does not run offline and keeps no local copy of the database.
- **Subject to UniProt rate limits.** Large or rapid batches may be throttled by the UniProt API; space out high-volume requests.
- **Runs on CPU.** There is no model and no GPU; latency is dominated by the network round-trip.
