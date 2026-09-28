<a href="https://bio-pro.mintlify.app/tools/database-retrieval/paperclip"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Paperclip

> [!NOTE]
> **License:** Paperclip retrieves data from literature records returned by Paperclip, distributed under source-specific terms. Attribution to literature records returned by Paperclip is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://paperclip.gxl.ai/) for full terms.

## Overview

[Paperclip](https://paperclip.gxl.ai/) is a hosted literature search and retrieval service from [GXL](https://gxl.ai/) covering more than 11 million biomedical papers and preprints. This toolkit exposes two tools over the Paperclip API, namely `paperclip-literature-search` (ranked hits with citation metadata, abstracts, and matched passages for a free-text query) and `paperclip-literature-document` (exact metadata and, when available, complete line-numbered full text for one paper). Both run on CPU in process and need only network access and a Paperclip API key.

## Background

[Paperclip](https://paperclip.gxl.ai/) is operated by [GXL](https://gxl.ai/) as a searchable index over scholarly corpora, including [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) full-text papers, [bioRxiv](https://www.biorxiv.org/) and [medRxiv](https://www.medrxiv.org/) preprints, [arXiv](https://arxiv.org/) preprints, and an abstract-level index spanning all of its scholarly sources. Search ranks results by combining lexical BM25 matching with embedding similarity, so a query can surface a relevant passage even when it shares few exact words with the text. Each paper is addressed by a stable identifier, such as `PMC11754230` or a `bio_`, `med_`, or `arx_` preprint ID, and is served as files in a virtual directory: `meta.json` holds its citation metadata, and `content.lines` holds its full text as numbered lines. The service also indexes regulatory documents, clinical trial registries, and protein databases, which these tools do not expose.

Internally, both tools send commands to the Paperclip API at `https://paperclip.gxl.ai` over HTTPS, authenticated with an API key header, using the same command protocol as the official `paperclip` command-line client. `paperclip-literature-search` runs one `search` over the selected corpora, which saves its hits on the server under a result ID, then reads the structured hit list back from that saved result. `paperclip-literature-document` reads the paper's `meta.json`, then its `content.lines` when full text is requested, and asks the service for that file's byte count. Text is labelled complete full text only when its length matches that count; anything shorter is returned as a preview. Results reflect the live service at query time rather than a fixed corpus snapshot.

Each hit and document records its source provenance, including the Paperclip identifier and URL, the retrieval time, and the observed access level. Document retrieval also records the SHA-256 checksum and byte size of the exact metadata and full-text bytes returned, so a later retrieval can be checked against them.

### Learning Resources

- [Paperclip documentation](https://paperclip.gxl.ai/docs) (GXL) - official guide to the corpora, search options, and the command-line client.

## Tools

### Paperclip Literature Search (`paperclip-literature-search`)

Searches selected corpora and normalizes stable identifiers, citation metadata, matched text, access level, and provenance. Exact-document retrieval records separate provenance, with a SHA-256 and byte size, for the metadata bytes and, when complete full text is available, the full-text bytes.

#### Applications

Use search to find natural sequences, experimental context, and design methods relevant to a biological objective.

#### Usage Tips

- Select only the corpora relevant to the query.
- Preserve stable identifiers and provenance when using a hit downstream.

### Paperclip Literature Document (`paperclip-literature-document`)

Retrieves exact metadata and complete line-addressable content when available. Truncated content is retained only as a preview and is never labeled full text.

#### Applications

Use these tools to collect source-backed design evidence and natural examples before applying sequence, structure, or mechanistic scoring tools.

#### Usage Tips

- Set `PAPERCLIP_API_KEY` before retrieval; the tools do not read credentials
  saved by an interactive `paperclip login`.
- Treat literature as evidence for candidate generation, not functional validation.
- Inspect `access_kind` before relying on content beyond citation metadata.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These notes apply to both Paperclip operations.

- **Requires authenticated network access.** No literature database is bundled or synchronized locally; every call is an HTTPS request to the Paperclip service.
- **Runs in process on CPU.** There is no environment to build and no worker to start; runtime is dominated by remote retrieval.
