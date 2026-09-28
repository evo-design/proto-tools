<a href="https://bio-pro.mintlify.app/tools/database-retrieval/paperclip"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Paperclip

> [!NOTE]
> **License:** Paperclip retrieves data from literature records returned by Paperclip, distributed under source-specific terms. Attribution to literature records returned by Paperclip is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://paperclip.gxl.ai/) for full terms.

## Overview

[Paperclip](https://paperclip.gxl.ai/) is a hosted literature search and retrieval service from [GXL](https://gxl.ai/) covering more than 3.4 million papers and preprints. This toolkit exposes two tools over the Paperclip API, namely `paperclip-literature-search` (ranked hits with citation metadata, abstracts, and matched passages for a free-text query) and `paperclip-literature-document` (exact metadata and, when available, complete line-numbered full text for one paper). Both run on CPU in process and need only network access and a Paperclip API key.

## Background

[Paperclip](https://paperclip.gxl.ai/) is built by [GXL](https://gxl.ai/) to make the biomedical literature searchable and readable by researchers and AI agents. It indexes full-text papers from [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) together with preprints from [bioRxiv](https://www.biorxiv.org/), [medRxiv](https://www.medrxiv.org/), and [arXiv](https://arxiv.org/), plus a separate abstract-level index spanning all of these sources. Searches combine keyword matching with matching by meaning, so a relevant passage can be found even when it is worded differently from the query. Every paper has a stable identifier, such as `PMC11754230`, and its full text is served with numbered lines, which makes it easy to cite the exact lines that support a claim.

Search returns the most relevant papers for a query, with citation details, abstracts, and the passages that matched. Document retrieval then fetches one paper's metadata and, when it is available, its complete full text. Both tools query the live Paperclip service, so results reflect the literature at the time of the request rather than a fixed snapshot. Each result also records where and when it was retrieved, so the same record can be traced back and checked later.

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
