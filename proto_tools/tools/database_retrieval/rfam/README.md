<a href="https://bio-pro.mintlify.app/tools/database-retrieval/rfam"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Rfam

![Rfam](https://proto-bio.github.io/proto-assets/images/tool/rfam/hero.png)

> [!NOTE]
> **License:** Rfam retrieves data from the Rfam database, distributed under CC0-1.0 (public domain; no attribution required). The client wrapper code is MIT-licensed. Please refer to [the data terms](https://rfam.org) for full terms.

## Overview

[Rfam](https://rfam.org) is a curated database of non-coding RNA families. Each family is a seed alignment of representative sequences, a consensus secondary structure, and a covariance model that annotates the family across sequenced genomes. This toolkit wraps two Rfam endpoints: `rfam-family` (a family's curation record, cutoffs, and consensus structure) and `rfam-regions` (every genome region the family is annotated in, with coordinates and strand).

## Background

Rfam is described in the Rfam 15 database report ([Ontiveros-Palacios et al., 2025](https://doi.org/10.1093/nar/gkae1023)), published in *Nucleic Acids Research*. It is maintained at [EMBL-EBI](https://www.ebi.ac.uk/) and released in numbered versions; each family records its authors, the source of its seed alignment and structure, and the bit-score cutoffs that define membership. Families with related structures are grouped into clans. Full-region annotations come from searching each family's covariance model against a genome database with [Infernal](https://eddylab.org/infernal/).

Internally, both tools issue HTTP GET requests to the Rfam website under `https://rfam.org/family/<family>`. `rfam-family` reads the family record as JSON and the seed alignment as Stockholm, from which it extracts the consensus structure (`#=GC SS_cons`) and the reference consensus sequence (`#=GC RF`). `rfam-regions` reads the family's regions table as plain text and normalizes each hit so that `start <= end`, with the strand stated separately. Results reflect the current Rfam release, which the outputs report.

### Learning Resources

- [Rfam documentation](https://docs.rfam.org/) (EMBL-EBI) - user guides covering families, clans, and how Rfam annotations are built.
- [Rfam API](https://docs.rfam.org/en/latest/api.html) (EMBL-EBI) - the family, regions, and alignment endpoints these tools call.
- [Infernal user guide](https://eddylab.org/infernal/) (Eddy lab) - the covariance-model software behind Rfam's alignments and annotations.

## Tools

### Rfam Family (`rfam-family`)

Fetches an Rfam family by accession or ID and returns its description, RNA type, authors, seed and structure sources, seed and full counts, clan, gathering, trusted, and noise cutoffs, release, and the consensus secondary structure with its reference sequence. The full Stockholm seed alignment is returned on request.

#### Applications

Use this to understand an RNA family before designing or annotating against it: read the consensus structure to locate stems and loops, check the gathering cutoff before interpreting a bit score, or take the seed alignment as a starting point for structure-aware sequence design. Pair it with [`rfam-regions`](https://bio-pro.mintlify.app/tools/database-retrieval/rfam) to find real instances of the family in a genome.

#### Usage Tips

- **Accession or ID both work.** `RF01731` and `TwoAYGGAY` name the same family, and the ID is matched case-insensitively. The output always reports the accession.
- **The consensus lines are column-aligned.** `consensus_structure` and `consensus_sequence` have one character per alignment column, in WUSS notation, where matching `<` `>` (or `(` `)`) brackets are base pairs and `_` marks hairpin loops.
- **The seed alignment can be large.** Set `include_seed_alignment=True` only when the alignment itself is needed; export it with the `sto` format.

### Rfam Regions (`rfam-regions`)

Lists every sequence region an Rfam family is annotated in, with sequence accession, bit score, start, end, strand, species, and NCBI taxonomy ID, optionally narrowed by taxonomy ID, species, or sequence accession.

#### Applications

Use this to find real copies of a structured RNA in an organism: list a family's hits in one genome, then pull each locus with flanking sequence through [`ncbi-efetch`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi) for scoring with an RNA or genomic language model, or for comparison against designed variants. It also gives the species distribution of a family for choosing natural homologs.

#### Usage Tips

- **Coordinates are ready for `ncbi-efetch`.** `start` and `end` are 1-indexed and inclusive with `start <= end`, and `strand` is `+` or `-`, matching `ncbi-efetch`'s `seq_start`, `seq_stop`, and `strand`. Subtract and add a margin to include flanks.
- **Filters apply after download.** `taxid`, `species` (case-insensitive substring), and `sequence_accession` (version optional) narrow the list; `total_regions` still reports the whole family.
- **Very large families cannot be listed.** Rfam refuses to return regions for families with millions of hits, such as tRNA (`RF00005`), and the tool raises an error that relays Rfam's message.
- **Results are capped.** At most `max_regions` hits are returned (500 by default); `matched_regions` and `truncated` say how many matched.

## Toolkit Notes

73-<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Rfam tool in this toolkit (`rfam-family`, `rfam-regions`).

- **Requires network access.** The tools call the live Rfam website.
