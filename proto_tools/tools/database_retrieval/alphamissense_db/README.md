<a href="https://bio-pro.mintlify.app/tools/database-retrieval/alphamissense-db"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# AlphaMissense DB

![AlphaMissense DB](https://proto-bio.github.io/proto-assets/images/tool/alphamissense_db/hero.png)

> [!NOTE]
> **License:** AlphaMissense DB retrieves data from the AlphaFold Protein Structure Database, distributed under CC-BY-4.0. Attribution to the AlphaFold Protein Structure Database is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://alphafold.ebi.ac.uk/faq) for full terms.

## Overview

AlphaMissense is a Google DeepMind model that predicts the pathogenicity of every possible missense substitution for the human proteome, precomputed and served as static CSV files by the AlphaFold Protein Structure Database. The `alphamissense-db-fetch` tool retrieves the full per-substitution prediction set for one human UniProt accession, returning each substitution's pathogenicity score in `[0, 1]`, its classification (`likely_benign`, `ambiguous`, or `likely_pathogenic`), the prediction count, and the mean pathogenicity. It runs on CPU and requires only network access.

## Background

AlphaMissense ([Cheng et al., 2023](https://doi.org/10.1126/science.adg7492)) is a deep-learning model that scores the pathogenicity of human missense variants. It is adapted from AlphaFold and fine-tuned on human and primate population variant frequencies, treating variants common in healthy populations as benign and rare variants as putatively pathogenic. For each canonical UniProt sequence it scores all 19 alternate amino acids at every position, covering every possible single missense substitution. Its classification thresholds are set to a cutoff that reaches about 90% precision on ClinVar variants. The paper reports that the model classifies 89% of all 71 million possible human missense variants, labeling 32% likely pathogenic and 57% likely benign at the default thresholds.

The predictions are not computed at query time. They are precomputed by [Google DeepMind](https://deepmind.google/) and distributed as static CSV files by the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), maintained by [EMBL-EBI](https://www.ebi.ac.uk/), keyed by UniProt accession at `https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-{suffix}.csv`.

Internally, the tool strips and uppercases the supplied accession, builds the AlphaFold DB CSV URL for the requested coordinate system, and issues a single HTTP GET. The `uniprot` coordinate system fetches the `aa-substitutions` CSV, which holds the full protein-coordinate grid of every possible substitution. The `hg19` and `hg38` coordinate systems fetch the genomic CSVs, which cover only substitutions reachable by a single-nucleotide change (a single-nucleotide variant, SNV) and additionally carry chromosome, position, reference allele, alternate allele, and GENCODE transcript identifier. Each CSV row is parsed into one prediction record, with the genomic fields populated only in genomic mode. A 404 response means the accession is not covered and surfaces as a clear error. Predictions reflect the fixed CSV snapshot published by AlphaFold DB rather than a value recomputed per request.

### Learning Resources

- [AlphaFold DB FAQ](https://alphafold.ebi.ac.uk/faq) (EMBL-EBI) - official documentation covering the AlphaMissense CSV files, coverage, and the genomic and protein coordinate variants.
- [AlphaMissense GitHub](https://github.com/google-deepmind/alphamissense) (Google DeepMind) - the official repository with usage notes for the released code and prediction tables.

## Tools

### AlphaMissense DB Fetch (`alphamissense-db-fetch`)

Retrieves the complete AlphaMissense prediction set for a single human UniProt accession and returns every per-substitution prediction, the prediction count, the mean pathogenicity score, and the source CSV URL. The `coordinate_system` configuration selects the protein-coordinate grid of every possible substitution or one of the genomic-coordinate tables, which are limited to substitutions reachable by a single-nucleotide change. The genomic tables additionally populate chromosome, position, reference allele, alternate allele, and transcript identifier on each prediction.

#### Applications

Use this to pull model-based missense pathogenicity into a pipeline: triage missense variants of uncertain significance from clinical sequencing, prioritize candidate disease-causing variants from case cohorts, avoid disruptive substitutions during sequence design or optimization, or apply a per-residue pathogenicity penalty in a generative-design loop. The accession can come from the [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot) tool, which resolves a gene symbol and organism to a canonical reviewed human accession. The same accession also drives the [AlphaFold DB](https://bio-pro.mintlify.app/tools/database-retrieval/alphafold-db) tool, aligning per-residue pathogenicity scores with predicted backbone coordinates.

#### Usage Tips

- **The tool always returns the entire prediction set.** There is no server-side filtering on the static CSV. Filter the returned `predictions` list in your own code by position, score, or classification.
- **Cache the output once per accession.** A typical protein has roughly 7,000 to 20,000 substitution rows. Fetch once and reuse the result rather than refetching inside tight loops.
- **Group by `position` for hotspot analysis.** `mean_pathogenicity` over a wide region is a coarse summary. Inspect predictions grouped by residue position to surface hotspots.
- **Coverage is human canonical isoforms only.** Non-canonical isoforms and non-human accessions return a 404 and surface as a clear error. Resolve the accession with the UniProt tool first if the organism is uncertain.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every AlphaMissense DB tool in this toolkit (`alphamissense-db-fetch`).

- **Requires network access.** The tool downloads the AlphaMissense CSV from the AlphaFold Protein Structure Database. It does not run offline and keeps no local copy of the predictions.
- **Subject to EMBL-EBI fair use.** The CSV is an anonymous static download from AlphaFold DB with no API key or account. Observe the [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/) and space out high-volume requests.
