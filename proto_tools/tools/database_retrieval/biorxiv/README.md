<a href="https://bio-pro.mintlify.app/tools/database-retrieval/biorxiv"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# bioRxiv

![bioRxiv](https://proto-bio.github.io/proto-assets/images/tool/biorxiv/hero.png)

> [!NOTE]
> **License:** bioRxiv retrieves data from preprint metadata served by the bioRxiv API, distributed under source-specific terms. Attribution to preprint metadata served by the bioRxiv API is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://api.biorxiv.org/) for full terms.

## Overview

[bioRxiv](https://www.biorxiv.org/) and [medRxiv](https://www.medrxiv.org/) are preprint servers for the life sciences and health sciences, run by openRxiv, an independent non-profit. This toolkit exposes one tool, `biorxiv-details`, which looks up a preprint by DOI through the bioRxiv API and returns every posted version with its date, the title, authors, category, license, abstract, and the journal DOI once the preprint is published. It runs on CPU in process and needs only network access.

## Background

[bioRxiv](https://www.biorxiv.org/) opened in 2013 so that researchers could share manuscripts before peer review, and [medRxiv](https://www.medrxiv.org/) followed in 2019 for clinical and health research. Preprints are screened but not peer reviewed. Authors can post revised versions, so one DOI may carry several versions, each with its own date and text, and a figure or value can change between them. When a preprint is later published, the servers link it to the journal article's DOI.

The [bioRxiv API](https://api.biorxiv.org/) serves this metadata for both servers. It looks records up by DOI or lists them by posting date, but it has no keyword search.

### Learning Resources

- [bioRxiv API](https://api.biorxiv.org/) (openRxiv) - endpoints and returned fields.
- [About bioRxiv](https://www.biorxiv.org/about-biorxiv) (openRxiv) - screening, versions, and licensing.

## Tools

### bioRxiv Details (`biorxiv-details`)

Looks up preprints by DOI and returns, for each, the latest version's title, authors, corresponding author, category, license, and abstract, the published journal DOI if any, every version with its date and JATS XML link, and the API URL used (`source_url`).

#### Applications

Use details to cite the exact version of a preprint that reports a result, to check whether it has since been published, and to move from a preprint to its journal article.

#### Usage Tips

- The bioRxiv API has no keyword search. To find preprints by topic, use `europepmc-search` with `SRC:PPR` in the query, then pass the DOI of a hit here.
- DOIs may be bare (`10.1101/2022.12.09.519842`) or a `https://doi.org/` link. Newer bioRxiv and medRxiv DOIs use the `10.64898` prefix.
- The default `server="auto"` tries bioRxiv, then medRxiv. Both servers issue the same DOI prefixes, so the prefix does not identify the server. The result's `server` field names the one that answered. Set `server="biorxiv"` or `server="medrxiv"` to query only that server.
- A DOI that is not found, or that the API rejects, returns a record with `error` set and no metadata; the other DOIs in the call still return.
- Take `published_doi` to `europepmc-search` (`DOI:"..."`) to find the published article and its PMCID.
- Each version's `jats_xml_url` is bioRxiv's full text, including for preprints whose full text Europe PMC does not hold. bioRxiv serves these files behind Cloudflare rate limits, so space out bulk downloads and expect occasional HTTP 429 responses.
- `paperclip-literature-*` also indexes bioRxiv and medRxiv full text, with line-numbered reading, but needs a `PAPERCLIP_API_KEY`.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **Requires network access.** Every call is an HTTPS request to the bioRxiv API; no API key is needed.
- **Runs in process on CPU.** There is no environment to build and no worker to start; runtime is dominated by remote retrieval.
