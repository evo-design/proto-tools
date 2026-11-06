<a href="https://bio-pro.mintlify.app/tools/database-retrieval/encode"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# ENCODE

![ENCODE](https://proto-bio.github.io/proto-assets/images/tool/encode/hero.png)

> [!NOTE]
> **License:** ENCODE retrieves data from the ENCODE portal, distributed under the ENCODE Data Use Policy. Attribution to the ENCODE portal is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://www.encodeproject.org/help/citing-encode/) for full terms.

## Overview

The [ENCODE portal](https://www.encodeproject.org) hosts the experiments and uniformly processed files of the Encyclopedia of DNA Elements, including ChIP-seq, ATAC-seq, DNase-seq, RNA-seq and Hi-C across human and mouse biosamples. The `encode-search` tool runs faceted searches of the portal by assay, target, biosample, output type and assembly, and returns each file's or experiment's accession, metadata and direct download URL.

## Background

The ENCODE Consortium maps functional elements in the human and mouse genomes: transcription factor binding, histone modifications, chromatin accessibility, transcription and 3D genome contacts. The ENCODE Data Coordination Center at Stanford runs the portal, which is described in [*New developments on the Encyclopedia of DNA Elements (ENCODE) data portal*](https://doi.org/10.1093/nar/gkz1062) by Luo et al., published in *Nucleic Acids Research*.

Data is organized as **experiments** (`ENCSR...` accessions), each with a biosample, an assay and, for ChIP-seq, a target. Each experiment holds **files** (`ENCFF...` accessions): raw reads, and the outputs of the uniform processing pipelines such as alignments, signal tracks and peak calls on a given genome assembly. For each output type and assembly, the portal flags one file as the **preferred default**, for example the replicated IDR thresholded peaks of a TF ChIP-seq experiment.

The tool queries the portal's [REST API](https://www.encodeproject.org/help/rest-api/), whose search endpoint accepts the same facet filters as the portal's search pages and returns JSON.

### Learning Resources

- [ENCODE REST API](https://www.encodeproject.org/help/rest-api/) (ENCODE DCC) - the search endpoint, facet filters and JSON responses.
- [Getting started with ENCODE](https://www.encodeproject.org/help/getting-started/) (ENCODE DCC) - experiments, files, biosamples and how to browse them.
- [ENCODE file formats](https://www.encodeproject.org/help/file-formats/) (ENCODE DCC) - bed narrowPeak, bigWig, BAM and other processed formats.
- [ENCODE pipelines](https://www.encodeproject.org/pipelines/) (ENCODE DCC) - the uniform processing pipelines that produce the processed files.

## Tools

### ENCODE Search (`encode-search`)

Runs one portal search per query. A query is a mapping of ENCODE facet filters, such as `assay_title`, `target.label`, `biosample_ontology.term_name`, `output_type`, `assembly` and `file_format`, with optional free text. It returns files (default) or experiments with accession, parent experiment, assay, target, biosample, output type, assembly, file format and type, size, MD5 checksum, replicates, preferred-default flag, lab, portal page URL and, for files, the direct download URL. Each result also reports the total match count and the request URL.

#### Applications

Searches locate specific processed ENCODE files, such as the default IDR thresholded peaks for a transcription factor in a cell line or the fold-change signal for a histone mark. The download URLs and MD5 checksums can be cited or used to fetch the files for analysis. Experiment searches list the assays available for a target or biosample before choosing files.

#### Usage Tips

- **Filter values must match portal terms exactly.** Facets match exact strings, e.g. `"TF ChIP-seq"`, `"IDR thresholded peaks"`, `"HepG2"`. Copy terms from the facet lists on the [portal search page](https://www.encodeproject.org/search/?type=File); a misspelled value or unknown facet name returns no hits rather than an error.
- **Use `default_only` for the canonical file.** `default_only=True` adds `preferred_default=true`, which usually leaves one file per experiment, output type and assembly. Leave it off to see per-replicate and superseded outputs.
- **`released_only` drops archived and revoked items.** It adds `status=released` unless the query sets its own `status` filter. A key ending in `!`, e.g. `"assembly!": "hg19"`, negates a filter.
- **Free text is fuzzy.** `search_term` matches across descriptions and metadata, which helps when facet terms are unknown. Combine it with filters to narrow the hits.
- **The tool returns metadata, not files.** Fetch a file from its `download_url`. Results export as JSON (full output) or CSV (one row per hit).

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ENCODE tool in this toolkit (`encode-search`).

- **Requires network access.** The tool calls the live ENCODE portal over HTTPS and runs in the current Python process.
- **Results follow the live portal.** New releases and reprocessing can add, archive or change default files; keep the accessions and MD5 checksums with exported results for provenance.
