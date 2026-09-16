<a href="https://bio-pro.mintlify.app/tools/database-retrieval/paperclip"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

<img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Paperclip

> [!NOTE]
> **License:** Paperclip retrieves data from Literature records returned by Paperclip, distributed under source-specific terms. Attribution to Literature records returned by Paperclip is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://paperclip.gxl.ai/) for full terms.

## Overview

Paperclip provides live literature search and exact-document retrieval without maintaining a local corpus.

## Background

Paperclip exposes searchable publication metadata, abstracts, matched passages, and available full text from PMC and supported preprint and abstract corpora. Proto Tools runs the authenticated client in the tool's managed standalone environment, and results record their observed access level and source provenance.

## Installation

No pre-existing Paperclip installation is required. On the first Paperclip tool
call, Proto Tools creates the standard isolated `paperclip` environment and runs
`standalone/setup.sh`. That setup downloads the repository-pinned client wheel,
verifies its SHA-256, and installs its declared dependencies.

Paperclip currently publishes its SDK at one mutable wheel URL rather than at
immutable versioned URLs. Proto Tools therefore pins the reviewed version and
checksum together and fails closed if that upstream artifact changes. A
Paperclip upgrade requires updating and reviewing both values in
`standalone/setup.sh`.

Export `PAPERCLIP_API_KEY` for a new non-interactive environment. Existing
credentials created by `paperclip login` are also reused when present. The
optional `PAPERCLIP_LIBRARY_PATH` setting remains as a compatibility fallback,
but is not needed for normal Proto Tools execution.

## Tools

### Paperclip Literature Search (`paperclip-literature-search`)

Searches selected corpora and normalizes stable identifiers, citation metadata, matched text, access level, and provenance. Exact-document retrieval separately checksums the client-returned metadata bytes and, when complete full text is available, the full-text bytes; no second content copy is stored in the artifact descriptors.

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

- Set `PAPERCLIP_API_KEY` before retrieval. A pre-existing Paperclip login is
  also detected, but is not required when the API key is configured.
- Use `PAPERCLIP_LIBRARY_PATH` only as a compatibility fallback for an existing
  client installation.
- Treat literature as evidence for candidate generation, not functional validation.
- Inspect `access_kind` before relying on content beyond citation metadata.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These notes apply to both Paperclip operations.

- **Requires authenticated network access.** No literature database is bundled or synchronized locally.
- **Runs on CPU.** Runtime is dominated by remote retrieval.
