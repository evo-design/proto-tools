<a href="https://bio-pro.mintlify.app/tools/database-retrieval/alphafold-db"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# AlphaFold DB

![AlphaFold DB](https://alphafold.ebi.ac.uk/assets/img/whatsnew-2026-AFDB-Collaboration.webp)

> *Image source: [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)*

> [!NOTE]
> **License:** AlphaFold DB retrieves data from the AlphaFold Protein Structure Database, distributed under CC-BY-4.0. Attribution to the AlphaFold Protein Structure Database is required when the data is redistributed. The client wrapper code is Apache-2.0-licensed. Please refer to [the data terms](https://alphafold.ebi.ac.uk/faq) for full terms.

## Overview

The [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) is a public archive of protein structures predicted by [AlphaFold2](https://deepmind.google/science/alphafold/), maintained by Google DeepMind and EMBL-EBI and indexed by UniProt accession. The `alphafold-db-fetch` tool retrieves a single prediction record from the AlphaFold DB REST API, returning a parsed `Structure` (PLDDT B-factors, per-residue pLDDT and optional pAE on `structure.metrics`), the predicted sequence, organism and gene metadata, and the full JSON record. It runs on CPU and requires only network access.

## Background

The AlphaFold Protein Structure Database (AFDB) ([Varadi et al., 2022](https://doi.org/10.1093/nar/gkab1061)) is a freely accessible archive of protein structures predicted by AlphaFold2 (Jumper et al., 2021), maintained by [Google DeepMind](https://deepmind.google/) and [EMBL-EBI](https://www.ebi.ac.uk/). It hosts predicted atomic coordinates for the UniProt reference proteomes. Each entry carries a per-residue confidence score (pLDDT, 0 to 100) and a pairwise predicted aligned error (pAE) matrix in angstroms. AFDB hosts AlphaFold2 single-chain predictions only. Multi-chain complexes are produced by separate pipelines and are not part of this database.

Internally, the tool issues a GET request to the AFDB prediction endpoint at `https://alphafold.ebi.ac.uk/api/prediction/{accession}`, which returns a JSON list of prediction records. It selects the canonical record (`AF-{accession}-F1`) by default, or the record matching the requested isoform, then follows the URLs carried in that record: `pdbUrl` or `cifUrl` for the structure body, `plddtDocUrl` for the per-residue pLDDT array, `paeDocUrl` for the pAE matrix, and `msaUrl` for the input multiple-sequence alignment (an A3M file). The mean pLDDT is read from the record's `globalMetricValue` field. Records and their provenance come directly from the official AlphaFold DB REST API. Results reflect the live database, which always serves the latest version of each prediction.

### Learning Resources

- [AlphaFold DB FAQ](https://alphafold.ebi.ac.uk/faq) (EMBL-EBI) - official guidance on coverage, confidence interpretation, versioning, and downloads.
- [AlphaFold DB API documentation](https://alphafold.ebi.ac.uk/api-docs) (EMBL-EBI) - the REST API specification for prediction records and artifact URLs.
- [AlphaFold Protein Structure Database](https://www.ebi.ac.uk/training/online/courses/alphafold/) (EMBL-EBI Training) - a guided introduction to the database and how to interpret its predictions.

## Tools

### AlphaFold DB Fetch (`alphafold-db-fetch`)

Retrieves a single AlphaFold DB prediction record by UniProt accession and returns the predicted sequence and its 1-indexed coordinates, gene and organism metadata, mean pLDDT, the AFDB artifact URLs, the full JSON record, and an optional parsed `Structure` carrying per-residue pLDDT and optional pAE on `structure.metrics`.

#### Applications

Use this to pull an AlphaFold-predicted structure into a pipeline when no experimental entry is needed: fetch a target by accession before inverse folding, docking, or binder design, screen accessions for AFDB coverage with metadata-only requests, or assess per-residue and pairwise confidence before structure-based work. The returned `Structure` feeds directly into structure-consuming tools such as TM-align, US-align, and structure scoring. The [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot) tool supplies the UniProt accession from a gene name and organism, and the [PDB](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) tool provides the experimental counterpart when one exists.

#### Usage Tips

- **Coverage is broad but not universal.** When AFDB has no prediction for an accession the tool raises `ValueError`. Catch that error and fall back to predicting the structure from sequence.
- **A high `mean_plddt` can hide locally unreliable regions.** Inspect the per-residue pLDDT on `structure.metrics` before trusting any specific residue.
- **`latest_version` advances when AFDB refreshes a prediction.** Cache it alongside any structure you persist and refetch when it moves past the cached value.
- **Multiple records signal isoforms or fragments.** The canonical record is selected by default and a warning lists the alternatives. To select a non-canonical isoform, pass the `isoform` input, and check `entry_id`, `sequence_start`, and `sequence_end` to confirm which record was returned.
- **Low-confidence regions are usually real disorder, not a prediction error.** Disordered or flexibly linked regions get very low per-residue confidence (pLDDT) and high predicted aligned error (pAE) between regions because they have no single fixed shape. Find those residue ranges from the per-residue pLDDT array and trim or down-weight just those residues. Do not throw away the whole prediction, because the confident domains are still reliable.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every AlphaFold DB tool in this toolkit (`alphafold-db-fetch`).

- **Requires network access.** The tool calls the live AlphaFold DB REST API. It does not run offline and keeps no local copy of the database.
- **Subject to AlphaFold DB rate limits.** The EMBL-EBI API is unauthenticated and applies per-IP fair-use limits ([EMBL-EBI Terms of Use](https://www.ebi.ac.uk/about/terms-of-use/)). Space out high-volume requests.
