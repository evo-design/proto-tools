<a href="https://bio-pro.mintlify.app/tools/database-retrieval/synbiohub"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# SynBioHub

![SynBioHub](https://proto-bio.github.io/proto-assets/images/tool/synbiohub/hero.png)

> [!NOTE]
> **License:** SynBioHub retrieves data from caller-selected SynBioHub registries, distributed under registry and record-specific terms. Attribution to caller-selected SynBioHub registries is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://wiki.synbiohub.org/) for full terms.

## Overview

The `synbiohub-fetch` tool retrieves one design from a caller-selected SynBioHub registry and returns it as typed parts and features, with the source SBOL kept verbatim. It maintains no local parts catalog.

## Background

[SynBioHub](https://synbiohub.org/) is a standards-enabled repository for synthetic biology designs. SBOL preserves components, nested subcomponents, sequence constraints, annotations, and provenance, allowing a circuit or construct to remain a composition of segments rather than only a flattened nucleotide string. The tool uses the documented SBOL download endpoint and authenticates with a SynBioHub login token.

## Tools

### SynBioHub Fetch (`synbiohub-fetch`)

Downloads the recursive SBOL of one ComponentDefinition and parses it. `design` holds the requested part: its sequence, Sequence Ontology roles, and features, each with 1-indexed inclusive locations and a `subpart_uri` when it places a subpart. `subparts` holds every other part in the document. The output exports to JSON, the verbatim SBOL, FASTA, or GenBank. The selected object must share an origin with `registry_url`, preventing a design identifier from silently redirecting retrieval to another host.

#### Applications

Read a construct's composition from `design.features` and `subparts` to rebuild it as multi-segment designs, and export FASTA or GenBank for downstream scoring, synthesis checks, or annotation inspection.

#### Usage Tips

- **Set a SynBioHub token.** The public instance requires a login for every download, including public records. Create a free account at [synbiohub.org](https://synbiohub.org/register), then exchange your email or username and password for a token and export it:

  ```bash
  export SYNBIOHUB_TOKEN=$(curl -s -X POST https://api.synbiohub.org/login \
    -H "Accept: text/plain" \
    --data-urlencode "email=YOUR_EMAIL" \
    --data-urlencode "password=YOUR_PASSWORD")
  ```

  The token can also be passed as `synbiohub_token` in the config. Request a token from the same host as `registry_url`. Tokens can expire when the registry restarts or when you log out; if downloads start failing with a login redirect, request a new one.
- **Treat SBOL as the source of truth.** The typed model covers parts, roles, sequences, and ranged features. Provenance, interactions, and registry-specific annotations stay in the verbatim `sbol` field and its export.
- **Exports are views of the design.** FASTA and GenBank cover the requested design only. GenBank feature keys come from Sequence Ontology roles, and unmapped roles become `misc_feature`.
- **Tokens remain private.** `SYNBIOHUB_TOKEN` is excluded from cache keys and outputs.
- **Select the registry explicitly.** `object_uri` must have the same origin as `registry_url`.
- **Requires network access.** No registry snapshot or curated database is stored locally.
- **Runs on CPU.** Runtime is dominated by one SBOL download.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>
