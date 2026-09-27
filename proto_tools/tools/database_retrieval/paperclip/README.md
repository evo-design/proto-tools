<a href="https://bio-pro.mintlify.app/tools/database-retrieval/paperclip"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Paperclip

> [!NOTE]
> **License:** Paperclip retrieves data from Literature records returned by Paperclip, distributed under source-specific terms. Attribution to Literature records returned by Paperclip is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://paperclip.gxl.ai/) for full terms.

## Overview

Paperclip provides live literature search and exact-document retrieval without maintaining a local corpus.

## Background

Paperclip exposes searchable publication metadata, abstracts, matched passages, and available full text from PMC and supported preprint and abstract corpora. The corpus, search index, and full text all live on GXL's servers; Proto Tools talks to the service directly over HTTPS with the same command protocol the official `paperclip` CLI uses, and results record their observed access level and source provenance.

## Installation

Nothing to install. The tools run in process and need only network access to
`https://paperclip.gxl.ai` plus a credential. There is no local corpus, model, or
managed environment, and the `gxl-paperclip` package is not required.

Export `PAPERCLIP_API_KEY` (or `PAPERCLIP_BEARER_TOKEN`) before calling either
tool, or pass `paperclip_api_key` in the tool config. The key is sent as a request
header and is excluded from cache keys and provenance. Set `PAPERCLIP_BASE_URL` to
point at a different Paperclip deployment.

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
