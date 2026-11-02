<a href="https://bio-pro.mintlify.app/tools/database-retrieval/paperclip"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Paperclip

![Paperclip](https://proto-bio.github.io/proto-assets/images/tool/paperclip/hero.png)

> [!NOTE]
> **License:** Paperclip retrieves data from literature records returned by Paperclip, distributed under source-specific terms. Attribution to literature records returned by Paperclip is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://paperclip.gxl.ai/) for full terms.

## Overview

[Paperclip](https://paperclip.gxl.ai/) is a hosted literature search and retrieval service from [GXL](https://gxl.ai/) covering more than 12 million papers and preprints. This toolkit exposes four tools over the Paperclip API that follow GXL's recommended workflow, namely `paperclip-literature-search` (find papers by topic), `paperclip-literature-lookup` (find papers by DOI, PMID, or another metadata field), `paperclip-literature-grep` (find papers whose full text mentions exact text), and `paperclip-literature-document` (read part of one paper). All four run on CPU in process and need only network access and a Paperclip API key.

## Background

[Paperclip](https://paperclip.gxl.ai/) is built by [GXL](https://gxl.ai/) to make the biomedical literature searchable and readable by researchers and AI agents. It indexes full-text papers from [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) together with preprints from [bioRxiv](https://www.biorxiv.org/), [medRxiv](https://www.medrxiv.org/), and [arXiv](https://arxiv.org/), plus a separate abstract-level index spanning all of these sources. Searches combine keyword matching with matching by meaning, so a relevant passage can be found even when it is worded differently from the query. Every paper has a stable identifier, such as `PMC11754230`, and its full text is served with numbered lines, which makes it easy to cite the exact lines that support a claim.

GXL recommends finding papers first, then reading only the parts that matter. Search returns the most relevant papers for a topic, lookup resolves an identifier you already have, and grep finds every paper that mentions an exact term such as a gene name or accession. Reading a paper then means fetching a range of lines or the lines that match a pattern, rather than the whole text. All four tools query the live Paperclip service, so results reflect the literature at the time of the request. Each result also records where and when it was retrieved, so the same record can be traced back and checked later.

### Learning Resources

- [Paperclip documentation](https://paperclip.gxl.ai/docs) (GXL) - official guide to the corpora, search options, and the command-line client.
- [Paperclip install guide](https://paperclip.gxl.ai/install) (GXL) - connecting Paperclip's MCP server to Claude Code, Claude Desktop, Codex, Cursor, and other agents.

## Tools

### Paperclip Literature Search (`paperclip-literature-search`)

Finds papers by topic. Each hit carries its citation details, abstract, matched passage, and citation URL (`source_url`).

#### Applications

Use search to find natural sequences, experimental context, and design methods relevant to a biological objective.

#### Usage Tips

- For general biomedical literature, search `pmc`, the default. Use `papers` to cover PMC, bioRxiv, medRxiv, and arXiv together.
- Narrow results with `year`, `journal`, `author`, or `article_type` rather than adding those words to the query.
- Set `include_abstracts=False` for a compact hit list to triage, then read the promising papers with `paperclip-literature-document`.
- Up to 100 hits, a smaller `limit` returns a prefix of a larger one, as long as the Paperclip index has not been updated in between.

### Paperclip Literature Lookup (`paperclip-literature-lookup`)

Finds papers by one metadata field: DOI, PMID, PMC ID, arXiv ID, title, author, journal, or year.

#### Applications

Use lookup to open a paper that another tool already identified, such as a PMID from NCBI or a DOI from a UniProt reference.

#### Usage Tips

- Pass the returned `document_id` to `paperclip-literature-document` to read the paper.

### Paperclip Literature Grep (`paperclip-literature-grep`)

Searches the full text of every paper for a regular expression and returns each matching paragraph with the paper and corpus it came from.

#### Applications

Use grep to find every paper that mentions an exact gene, protein, dataset, or accession, including mentions in methods or data-availability sections that search can rank low.

#### Usage Tips

- Corpus-wide grep is time-bounded, so repeated calls can return a different set or order of matches.
- If a rare pattern returns nothing, retry with `exhaustive=True` to give the scan more time.

### Paperclip Literature Document (`paperclip-literature-document`)

Reads one paper's metadata plus a range of lines, the lines matching a pattern, or its full text. Every line is numbered, so evidence can be cited by line.

#### Applications

Use these tools to collect source-backed design evidence and natural examples before applying sequence, structure, or mechanistic scoring tools.

#### Usage Tips

- `document_id` takes a Paperclip id (`PMC11754230`, `bio_29a07e4e799b`, `arx_2605.11347`) or one as copied: lowercase `pmc...`, a PMC or Paperclip article link, or an arXiv id or link. Find a paper by DOI or PMID with `paperclip-literature-lookup` first.
- The paper's citation details come back typed in `document.paper`, the same fields a search hit carries.
- Prefer `read="lines"` (the default, first 100 lines) or `read="grep"` with a `pattern`; `max_matches` (20 by default) caps how many matching lines a grep returns. GXL advises against reading whole papers.
- `read="full_text"` returns the complete paper as `content`, which can be long.
- Set `PAPERCLIP_API_KEY` before retrieval, or `PAPERCLIP_TOKEN` for a bearer token; the config fields `paperclip_api_key` and `paperclip_token` take precedence. The tools do not read credentials saved by an interactive `paperclip login`.
- `PAPERCLIP_BASE_URL` points the tools at a different Paperclip deployment.
- Treat literature as evidence for candidate generation, not functional validation.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These notes apply to all four Paperclip tools.

- **For interactive or agentic work, use the Paperclip MCP server.** See the [Paperclip install guide](https://paperclip.gxl.ai/install) to connect it to your agent. These tools are for calling Paperclip from Python programs.
- **Requires authenticated network access.** No literature database is bundled or synchronized locally; every call is an HTTPS request to the Paperclip service.
- **Runs in process on CPU.** There is no environment to build and no worker to start; runtime is dominated by remote retrieval.
