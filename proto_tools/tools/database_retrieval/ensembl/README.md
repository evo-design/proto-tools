<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ensembl"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Ensembl

![Ensembl](https://proto-bio.github.io/proto-assets/images/tool/ensembl/hero.png)

> [!NOTE]
> **License:** Ensembl retrieves data from the Ensembl project, distributed under the EMBL-EBI Terms of Use. Attribution to the Ensembl project is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://www.ebi.ac.uk/about/terms-of-use/) for full terms.

## Overview

[Ensembl](https://www.ensembl.org/) is a genome annotation resource for vertebrate and model-organism genomes, providing genes, transcripts, exons, regulatory features, cross-references, and variant annotation, maintained by [EMBL-EBI](https://www.ebi.ac.uk/). This toolkit exposes five tools over the [Ensembl REST API](https://rest.ensembl.org/), namely `ensembl-lookup` (gene record by Ensembl ID or symbol), `ensembl-sequence` (DNA, cDNA, CDS, or protein sequence), `ensembl-overlap` (features overlapping a region), `ensembl-xrefs` (external-database identifiers), and `ensembl-vep` (per-transcript variant consequences from HGVS).

## Background

[Ensembl](https://www.ensembl.org/) ([Dyer et al., 2025](https://doi.org/10.1093/nar/gkae1071)) is a genome annotation resource maintained by [EMBL-EBI](https://www.ebi.ac.uk/). It integrates gene and transcript models, the [Ensembl Regulatory Build](https://www.ensembl.org/info/genome/funcgen/regulatory_build.html), cross-references to external databases, and variant consequence prediction with the [Variant Effect Predictor](https://www.ensembl.org/info/docs/tools/vep/index.html) (VEP) for human and other supported species. Coordinates returned by Ensembl are 1-indexed and inclusive, to match biological residue selection conventions.

Each tool issues a single HTTP GET to the [Ensembl REST API](https://rest.ensembl.org/), whose base URL is `https://rest.ensembl.org` for the [GRCh38](https://www.ncbi.nlm.nih.gov/grc/human) assembly. Setting `assembly="GRCh37"` routes requests to `https://grch37.rest.ensembl.org` instead. The endpoints used are `/lookup/id/{id}` and `/lookup/symbol/{species}/{symbol}` for `ensembl-lookup`, `/sequence/id/{id}` for `ensembl-sequence`, `/overlap/id/{id}` for `ensembl-overlap`, `/xrefs/id/{id}` for `ensembl-xrefs`, and `/vep/{species}/hgvs/{hgvs}` for `ensembl-vep`. Responses are parsed into typed Pydantic records, with the full upstream JSON preserved alongside in a `raw_payload` field. PascalCase keys such as `Transcript`, `Exon`, and `Translation` are kept verbatim so records round-trip cleanly. Results reflect the live Ensembl database at query time rather than a fixed release snapshot.

### Learning Resources

- [Ensembl REST API documentation](https://rest.ensembl.org/) (Ensembl) - the live endpoint reference with request parameters, response shapes, and an interactive console.
- [Ensembl and the Ensembl REST API](https://www.ebi.ac.uk/training/services/ensembl) (EMBL-EBI Training) - guided courses on Ensembl data and programmatic access.

## Tools

### Ensembl Lookup (`ensembl-lookup`)

Retrieves a single gene record, either directly by Ensembl gene ID or by gene symbol scoped to a species, returning the typed `EnsemblGene` (identifier, symbol, biotype, genomic coordinates, canonical transcript) plus the source URL and raw payload. With `expand` enabled, the response includes the nested transcript, translation, and exon hierarchy.

#### Applications

Use this to resolve a gene of interest as the entry point of nearly any Ensembl workflow. Convert a gene symbol such as `BRCA1` into its stable Ensembl gene ID, read off the canonical transcript and genomic coordinates for downstream transcriptomics, GWAS annotation, or sequence retrieval with [`ensembl-sequence`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl), or expand the transcript-and-exon hierarchy for splice-isoform analysis. The returned gene ID also feeds [`ensembl-overlap`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl) and [`ensembl-xrefs`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl).

#### Usage Tips

- **Provide exactly one of `ensembl_id` or `symbol`.** Supplying both or neither raises a validation error. A symbol lookup also requires `config.species` to disambiguate.
- **Nested transcripts and exons are absent unless `expand` is set.** The default matches Ensembl REST and returns the gene record only, so request expansion explicitly when you need the transcript or exon hierarchy.
- **`mane`, `phenotypes`, and `utr` apply only to ID-based lookup.** They are sent only on the `/lookup/id` path and are ignored for a symbol lookup. `mane` and `utr` additionally require `expand`.

### Ensembl Sequence (`ensembl-sequence`)

Retrieves the sequence for an Ensembl gene, transcript, or protein ID and returns one or more `EnsemblSequence` records (stable ID, description, molecule type, sequence string) alongside the source URL and raw payload.

#### Applications

Use this to retrieve a reference sequence at gene, transcript, or protein granularity for any downstream analysis. Fetch the spliced mRNA or coding sequence of a canonical transcript resolved by [`ensembl-lookup`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl) for primer design, codon-usage analysis, or sequence comparison, pull the genomic span with introns for promoter or splice-site studies, or obtain the protein translation for multiple-sequence alignment or structure prediction. Repeat masking and feature masking support cis-element and intron-aware analyses.

#### Usage Tips

- **The returned `id` may differ from the input ID.** Requesting a protein sequence for a transcript ID resolves to the corresponding protein ID, so read the record's `id` rather than assuming it echoes the input.
- **`mask` and `mask_feature` are mutually exclusive, as are the expand and trim pairs.** Setting both masks, or both `expand_5prime` and `start`, or both `expand_3prime` and `end`, raises a validation error.
- **Repeat masking and span expansion apply to genomic sequence only.** They have no effect on `cdna`, `cds`, or `protein` requests.
- **Set `multiple_sequences` when an ID maps to more than one record.** Without it, IDs that resolve to multiple sequences (patches, alternative haplotypes) return only the first.

### Ensembl Overlap (`ensembl-overlap`)

Retrieves features overlapping the genomic region of a given Ensembl ID and returns a list of `EnsemblOverlapFeatureRecord` entries, each exposing the common typed fields (feature type, identifier, biotype, coordinates, strand, region) plus a `raw` dict carrying the full upstream record.

#### Applications

Use this to annotate a genomic locus by listing what overlaps it. Identify which gene or transcript contains a GWAS hit, ChIP-seq peak, or ATAC-seq peak, enumerate the regulatory-build features (promoters, enhancers, transcription-factor binding sites) within a region for functional-genomics analysis, or pull overlapping variants filtered to a named set such as ClinVar to ask whether the region is clinically annotated. The locus is typically obtained from [`ensembl-lookup`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl).

#### Usage Tips

- **Records are typed only on the common fields. Feature-specific keys live in `raw`.** Different feature classes return divergent payload shapes, so read per-feature attributes from each record's `raw` dict.
- **`biotype` is most meaningful for gene and transcript features.** It filters those classes. Pairing it with unrelated feature types is unlikely to narrow results.
- **`so_term` and `variant_set` apply to variation features.** They have no effect when the feature class is not a variation type.

### Ensembl Xrefs (`ensembl-xrefs`)

Resolves an Ensembl ID to its external-database cross-references and returns a list of `EnsemblXref` records (external database name, display and primary identifiers, description, cross-reference type) plus the source URL and raw payload.

#### Applications

Use this to convert identifiers between Ensembl and the other major sequence, gene, and protein resources. Map an Ensembl gene or protein to a UniProt accession before fetching its entry with [`uniprot-fetch`](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot), which then bridges through to PDB structures via SIFTS, recover EntrezGene or RefSeq identifiers for NCBI-side retrieval, or follow GO and InterPro cross-references for functional annotation. The Ensembl ID is commonly produced by [`ensembl-lookup`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl).

#### Usage Tips

- **Filter on `dbname` in the result, not just `external_db`.** A single query can return several UniProt-related and RefSeq-related entries, so select the row by its `dbname` rather than assuming one record per database.
- **`all_levels` changes result scope on gene queries.** It fans cross-references out to child transcripts and translations, which can substantially enlarge the result.
- **Set `object_type` when a stable ID resolves ambiguously.** It restricts results to one feature type when the ID could map to a gene, transcript, or translation.

### Ensembl VEP (`ensembl-vep`)

Submits an HGVS notation to the Ensembl Variant Effect Predictor REST endpoint and returns a list of `EnsemblVEPConsequence` records (echoed input, most severe consequence as a Sequence Ontology term, region and coordinates, allele string, raw per-transcript consequences, co-located variants) plus a derived `num_consequences` count.

#### Applications

Use this to predict the functional consequence of a variant and assess its likely impact. Classify a coding or genomic HGVS notation as missense, synonymous, stop-gain, splice-disrupting, or noncoding, then read per-transcript SIFT and PolyPhen predictions alongside optional human-only AlphaMissense, REVEL, and CADD pathogenicity scores for clinical or research variant interpretation. Co-located variant lookups surface population-frequency context from gnomAD and ClinVar annotations. Candidate variants are often identified from features returned by [`ensembl-overlap`](https://bio-pro.mintlify.app/tools/database-retrieval/ensembl) or from a designed or observed substitution in a downstream design workflow.

#### Usage Tips

- **`transcript_consequences` and `colocated_variants` are returned as raw dicts.** Their field sets vary by consequence type and annotation toggles, so read them defensively rather than expecting a fixed shape.
- **`pick` and `per_gene` cannot be combined.** Setting both raises a validation error. Choose one collapse strategy.
- **Several annotations are species- or assembly-restricted.** MANE applies to GRCh38 only. AlphaMissense, REVEL, and CADD are human only. APPRIS, TSL, and CCDS are human and mouse only. Enabling a restricted annotation outside its scope simply yields no extra data.
- **A coding or genomic HGVS form is more reliable than a protein form.** A protein-level notation can map to multiple transcripts ambiguously, so prefer coding or genomic notation when available.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Ensembl tool in this toolkit (`ensembl-lookup`, `ensembl-sequence`, `ensembl-overlap`, `ensembl-xrefs`, `ensembl-vep`).

- **Requires network access.** Every tool calls the live Ensembl REST API. None runs offline and no local copy of the database is kept.
- **Subject to the Ensembl REST rate limit.** Ensembl REST enforces a uniform per-IP limit of roughly 55,000 requests per hour, returning HTTP 429 with a `Retry-After` header when exceeded. There is no account or API key that raises this limit.
