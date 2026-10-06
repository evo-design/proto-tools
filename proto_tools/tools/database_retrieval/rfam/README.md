<a href="https://bio-pro.mintlify.app/tools/database-retrieval/rfam"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Rfam

![Rfam](https://proto-bio.github.io/proto-assets/images/tool/rfam/hero.png)

> [!NOTE]
> **License:** Rfam retrieves data from the Rfam database, distributed under CC0-1.0 (public domain; no attribution required). The client wrapper code is MIT-licensed. Please refer to [the data terms](https://docs.rfam.org/en/latest/#license) for full terms.

## Overview

[Rfam](https://rfam.org) is a database of non-coding RNA families represented by curated sequence alignments, consensus secondary structures, and covariance models. The toolkit provides two retrieval tools: `rfam-family` returns a family record and its consensus annotations, with an optional seed alignment; `rfam-regions` returns annotated sequence regions with coordinates, strand, and taxonomic information. Both tools access the Rfam website over HTTPS and run in process without a GPU or a separate tool environment.

## Background

Rfam is developed at [EMBL-EBI](https://docs.rfam.org/en/latest/). An RNA family is a group of sequences believed to be evolutionarily related through similarity in sequence or secondary structure. Related families may be grouped into [clans](https://docs.rfam.org/en/latest/glossary.html#clan). The database and its release 15.0 updates are described in [*Rfam 15: RNA families database in 2025*](https://doi.org/10.1093/nar/gkae1023) by Ontiveros-Palacios et al., published in *Nucleic Acids Research*.

Each family has a manually curated **seed alignment**, a representative set of sequences annotated with a consensus secondary structure. Rfam uses this alignment to build a **covariance model**, a statistical model that scores both sequence and secondary structure similarity. [Infernal](https://eddylab.org/infernal/) searches these models against the Rfamseq sequence database to identify additional candidate homologues. A curator-defined gathering cutoff specifies the bit-score threshold for inclusion in the family. The [family-building documentation](https://docs.rfam.org/en/latest/building-families.html) describes this process and the sources of structural annotations.

The toolkit retrieves these existing records through the [Rfam API](https://docs.rfam.org/en/latest/api.html). `rfam-family` reads the family description as JSON and extracts the consensus structure (`#=GC SS_cons`) and reference annotation (`#=GC RF`) from the Stockholm seed alignment. `rfam-regions` parses the family's region table and separates strand orientation from the start and end coordinates. The family output includes the database release and release date; the regions output includes the release when it is present in the table header.

### Learning Resources

- [How Rfam families are built](https://docs.rfam.org/en/latest/building-families.html) (Rfam) - seed alignments, structural annotations, and covariance-model searches.
- [Rfam glossary](https://docs.rfam.org/en/latest/glossary.html) (Rfam) - definitions of families, clans, gathering cutoffs, and alignment formats.
- [Rfam API](https://docs.rfam.org/en/latest/api.html) (Rfam) - reference for family records, sequence regions, and alignments.
- [Infernal documentation](https://eddylab.org/infernal/) (Eddy lab) - the software used to build and search RNA covariance models.

## Tools

### Rfam Family (`rfam-family`)

Retrieves a family by accession or family ID and returns its description, RNA type, curation information, sequence and species counts, clan membership when available, and gathering, trusted, and noise cutoffs. The output also contains the consensus secondary structure, reference annotation, and database release information. The complete Stockholm seed alignment can be included through configuration.

#### Applications

Family records provide context for interpreting RNA annotations. The consensus structure describes conserved pairing patterns across the alignment, while the curation fields identify the sources of the alignment and structure. These records support comparisons of representative family sequences and interpretation of model scores alongside the reported cutoffs. The seed alignment can also be used for further alignment or structural analysis.

#### Usage Tips

- **Families can be identified by accession or ID.** For example, `RF01731` and `TwoAYGGAY` identify the same [Rfam family](https://rfam.org/family/RF01731). The output reports both identifiers.
- **Consensus annotations use alignment coordinates.** `consensus_structure` contains the Stockholm `SS_cons` annotation in [WUSS notation](https://docs.rfam.org/en/latest/glossary.html#wuss-format); `consensus_sequence` contains the `RF` reference annotation. These strings include alignment columns and should not be interpreted as an unaligned nucleotide sequence or genomic coordinates.
- **Structural annotations have different sources.** Rfam includes both experimentally supported and computationally predicted structures. The `structure_source` field records provenance when available; the [Rfam documentation](https://docs.rfam.org/en/latest/building-families.html) explains why an underlying publication may be needed to establish the type of evidence.
- **The seed alignment is optional in the output.** Set `include_seed_alignment=True` to retain it and enable `sto` export. The tool downloads the alignment to extract the consensus annotations even when this option is disabled. Family records can also be exported as JSON.

### Rfam Regions (`rfam-regions`)

Retrieves the annotated sequence regions for a family, with optional filters for NCBI taxonomy ID, species name, or sequence accession. Each region contains a versioned sequence accession, Infernal bit score, start and end coordinates, strand, sequence description, species name, and taxonomy ID. The output includes the family identifiers, total and filtered region counts, and a flag indicating whether the returned list was truncated.

#### Applications

Region records locate candidate family members in the sequences represented by Rfam. Filtering by species or sequence accession supports examination of annotated loci in a particular organism or genome record. The accession, coordinates, and strand can be passed to [`ncbi-efetch`](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi) to retrieve the corresponding nucleotide subsequence for comparative analysis. Taxonomic fields also support examination of a family's distribution within the Rfam dataset.

#### Usage Tips

- **Coordinates are 1-indexed and inclusive.** The tool normalizes each region to `start <= end` and reports orientation separately as `+` or `-`. These values correspond to `ncbi-efetch`'s `seq_start`, `seq_stop`, and `strand` inputs.
- **Filters are applied after download.** `taxid` matches an exact taxonomy ID, `species` matches a case-insensitive substring, and `sequence_accession` accepts either a versioned or an unversioned accession. When several filters are provided, a region must satisfy all of them.
- **The return limit applies after filtering.** `max_regions` defaults to 500. `total_regions` reports the family-wide count, `matched_regions` counts all regions satisfying the filters, and `truncated` indicates that some matching regions were omitted. The limit does not reduce the size of the download.
- **Some families are too large for the endpoint.** The [Rfam API documentation](https://docs.rfam.org/en/latest/api.html#sequence-regions) states that the server can refuse region downloads for very large families. Local filters cannot bypass this restriction.
- **Region tables can be exported.** JSON preserves the full output, including counts and release information; TSV and CSV contain the returned region rows.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Rfam tool in this toolkit (`rfam-family`, `rfam-regions`).

- **Requires network access.** Both tools retrieve data from the Rfam website using HTTPS requests and execute in the current Python process.
- **Results depend on the Rfam release.** The tools query the live website rather than selecting a fixed database release. Retain the reported release information with exported results for provenance.
