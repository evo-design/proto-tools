<a href="https://bio-pro.mintlify.app/tools/database-retrieval/synbiohub"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# SynBioHub

> [!NOTE]
> **License:** SynBioHub retrieves data from caller-selected SynBioHub registries, distributed under registry and record-specific terms. Attribution to caller-selected SynBioHub registries is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://wiki.synbiohub.org/) for full terms.

## Overview

The `synbiohub-fetch` tool retrieves exact SBOL, FASTA, and GenBank representations of a top-level design object from a caller-selected SynBioHub registry. It maintains no local parts catalog.

## Background

[SynBioHub](https://synbiohub.org/) is a standards-enabled repository for synthetic biology designs. SBOL preserves components, nested subcomponents, sequence constraints, annotations, and provenance, allowing a circuit or construct to remain a composition of segments rather than only a flattened nucleotide string. The tool uses the documented download endpoints and supports the optional `SYNBIOHUB_TOKEN` header; individual deployments may require login even for records displayed as public.

## Tools

### SynBioHub Fetch (`synbiohub-fetch`)

Downloads recursive or non-recursive SBOL plus optional FASTA and GenBank representations. Every returned document has an exact-byte SHA256 artifact descriptor. The selected object must share an origin with `registry_url`, preventing a design identifier from silently redirecting retrieval to another host.

#### Applications

Use SBOL as the primary representation for composing multi-segment constructs and use FASTA or GenBank as sequence-oriented views for downstream scoring, synthesis checks, or annotation inspection.

#### Usage Tips

- **Prefer recursive SBOL for composition.** It includes referenced child objects needed to reconstruct a hierarchical design.
- **Treat FASTA as a view, not the design model.** Flattened sequence cannot retain all component relationships.
- **Tokens remain private.** `SYNBIOHUB_TOKEN` is excluded from cache keys and provenance. The current public instance returns HTTP 401 for some public-object downloads without a token.
- **Select the registry explicitly.** `object_uri` must have the same origin as `registry_url`.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These notes apply to `synbiohub-fetch`.

- **Requires network access.** No registry snapshot or curated database is stored locally.
- **Runs on CPU.** Runtime is dominated by the selected document downloads.
