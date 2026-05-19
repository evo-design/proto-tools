<a href="https://bio-pro.mintlify.app/tools/database-retrieval/pdb"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# PDB

![PDB](https://cdn.rcsb.org/rcsb-pdb/v2/about-us/Logo/rcsb-pdb-logo.png)

> *Image source: [RCSB PDB](https://www.rcsb.org/)*

> [!NOTE]
> **License:** PDB retrieves data from the RCSB Protein Data Bank, distributed under CC0-1.0 (public domain; no attribution required). The client wrapper code is Apache-2.0-licensed. Please refer to [the data terms](https://www.rcsb.org/pages/policies) for full terms.

## Overview

The RCSB Protein Data Bank is a database of experimentally determined three-dimensional structures of proteins, nucleic acids, and their complexes. This toolkit provides two CPU-only tools: `pdb-fetch-entry` retrieves structure metadata (title, experimental method, and resolution) for a PDB accession, and `pdb-fetch-fasta` retrieves the chain sequences of an entry and classifies each as protein or nucleic acid. It runs on CPU and requires only network access.

## Background

The Protein Data Bank ([Berman et al., 2000](https://doi.org/10.1093/nar/28.1.235)) is the single worldwide archive of experimentally determined macromolecular structures, served here through the RCSB PDB. It is operated by the Research Collaboratory for Structural Bioinformatics (RCSB) at Rutgers University and the University of California San Diego, with funding from the National Science Foundation, the National Institutes of Health, and the Department of Energy. Entries are solved by X-ray crystallography, cryo-electron microscopy, nuclear magnetic resonance spectroscopy, and other experimental methods.

The tools call two RCSB HTTP endpoints directly. `pdb-fetch-entry` issues a GET request to the RCSB Data API core entry endpoint (`https://data.rcsb.org/rest/v1/core/entry/{pdb_id}`) and reads the structure title from `struct.title`, the experimental method from the first `exptl` record, and the resolution from `rcsb_entry_info.resolution_combined`, which covers both X-ray and cryo-EM entries; entries solved by NMR have no resolution value. `pdb-fetch-fasta` requests the FASTA endpoint (`https://www.rcsb.org/fasta/entry/{pdb_id}`), parses each record, extracts the author-assigned chain identifiers from the header, and classifies a sequence as protein when it contains amino-acid letters that do not also occur in nucleotide alphabets. Both tools uppercase the supplied accession, retry transient HTTP failures with backoff, and return an empty result when the accession is not found (HTTP 404). Results reflect the live archive at query time rather than a fixed release snapshot.

### Learning Resources

- [RCSB PDB Data API documentation](https://www.rcsb.org/docs/programmatic-access/web-apis-overview) (RCSB PDB) - reference for the REST endpoints, query syntax, and rate limits.
- [PDB-101 training and education](https://pdb101.rcsb.org/) (RCSB PDB) - guided material on PDB data, structure determination methods, and how to interpret entries.

## Tools

### PDB Fetch Entry (`pdb-fetch-entry`)

Retrieves structure metadata for a PDB accession from the RCSB Data API core entry endpoint, returning the structure title, the experimental method, the resolution in angstroms, and the request URL.

#### Applications

Use this to assess whether an experimental structure is suitable as a reference before structure-based design or benchmarking: check the experimental method and resolution, then decide whether to use the entry. It pairs with [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot), whose returned PDB cross-references can be ranked by resolution, and with [PDB Fetch FASTA](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) to pull the chain sequences once a suitable entry is selected.

#### Usage Tips

- **Resolution is absent for some methods.** NMR and fiber-diffraction entries have no resolution value, so `resolution` is `None`; filter on it before sorting entries by quality.
- **This is metadata only.** The tool returns the title, method, and resolution, not atomic coordinates or a structure file.
- **An unknown accession is not an error.** A missing or obsolete accession returns an empty output rather than raising, so check the populated fields before using the result.

### PDB Fetch FASTA (`pdb-fetch-fasta`)

Retrieves the chain sequences of a PDB entry from the RCSB FASTA endpoint, returning one record per unique sequence with the author-assigned chain identifiers that share it, the FASTA header, the sequence, and a protein/nucleic-acid classification, plus the request URL.

#### Applications

Use this to extract reference sequences from an experimental structure for sequence design, alignment, or comparison against computational predictions. Filter `chains` by `is_protein` to separate protein subunits from nucleic-acid chains in a complex, and deduplicate identical sequences to recover the unique entities of a homo-oligomer. It follows [PDB Fetch Entry](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) once a suitable entry is chosen and consumes PDB identifiers surfaced by [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot).

#### Usage Tips

- **One record can cover several chains.** A single `PdbChain` carries every author-assigned chain identifier that shares its sequence, so a homo-oligomer collapses to one record with multiple `chain_ids`.
- **Protein classification is heuristic.** A chain is called protein only when it contains amino-acid letters absent from nucleotide alphabets; peptide nucleic acids and other hybrid molecules may be misclassified.
- **An unknown accession is not an error.** A missing or obsolete accession returns an empty `chains` list rather than raising.
- **Exporting to `fasta` writes the original headers.** The `fasta` export emits each record using its stored FASTA header verbatim; `json` and `csv` are also supported, with the `csv` form joining shared chain identifiers with a semicolon.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PDB tool in this toolkit (`pdb-fetch-entry`, `pdb-fetch-fasta`).

- **Requires network access.** The tools call the live RCSB PDB HTTP endpoints; they do not run offline and keep no local copy of the archive.
- **Subject to RCSB rate limits.** RCSB throttles clients that exceed a few requests per second and returns HTTP 429 when the limit is exceeded; space out high-volume requests, since no account or API key is available to raise the limit.
- **Runs on CPU.** There is no model and no GPU; latency is dominated by the network round-trip.
