<a href="https://bio-pro.mintlify.app/tools/database-retrieval/arxiv"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# arXiv

> [!NOTE]
> **License:** arXiv retrieves data from arXiv descriptive metadata, distributed under CC0-1.0 (public domain; no attribution required). The client wrapper code is MIT-licensed. Please refer to [the data terms](https://info.arxiv.org/help/api/tou.html) for full terms.

## Overview

[arXiv](https://arxiv.org/) is the open preprint server run by Cornell University for physics, mathematics, computer science, quantitative biology, and related fields. This toolkit exposes one tool, `arxiv-search`, which queries the arXiv API by title, author, abstract, or category and returns each preprint's versioned ID, title, authors, abstract, dates, categories, DOI and journal reference when present, and abstract and PDF links. It runs on CPU in process and needs only network access.

## Background

[arXiv](https://arxiv.org/) has distributed preprints since 1991 and now holds more than two million. Machine learning methods for biology, including many protein structure prediction and design models, often appear there first, in categories such as `q-bio.BM` (biomolecules), `q-bio.QM` (quantitative methods), and `cs.LG` (machine learning). Submissions are moderated but not peer reviewed. Each has an identifier such as `2504.00146`, and each revision adds a version suffix (`v1`, `v2`), so a citation can name the exact text that was read.

### Learning Resources

- [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html) (arXiv) - query syntax, field prefixes, and the Atom response format.
- [arXiv category taxonomy](https://arxiv.org/category_taxonomy) (arXiv) - category codes such as `q-bio.BM` and `cs.LG`.

## Tools

### arXiv Search (`arxiv-search`)

Searches arXiv and returns ranked hits with arXiv ID and version, title, authors, abstract, first and latest submission dates, categories, DOI, journal reference, author comment, and abstract and PDF URLs.

#### Applications

Use search to find machine learning and computational biology preprints, such as a design model's methods paper and its reported benchmarks, that journal databases index late or not at all.

#### Usage Tips

- Queries use arXiv field prefixes: `ti:` (title), `au:` (author), `abs:` (abstract), `cat:` (category), and `all:`. Combine them with `AND`, `OR`, and `ANDNOT`, and quote phrases, as in `ti:"protein binder" AND cat:q-bio.BM`.
- A bare word without a prefix searches all fields, which can match loosely; prefer `ti:` or `abs:` with a phrase.
- Set `sort_by="submittedDate"` for the newest preprints first.
- arXiv asks for at most one request every three seconds over a single connection. The limit applies across all calls in the process, retries included, so concurrent calls queue behind each other and each query waits its turn.
- A query arXiv rejects, such as one with malformed syntax, returns a result with `error` set and no hits, and the other queries still run.
- `doi` and `journal_ref` are filled only when authors add them after publication.
- `paperclip-literature-search` with `sources=["papers"]` also covers arXiv, with full text and line-numbered reading, but needs a `PAPERCLIP_API_KEY`; this tool needs no key and returns arXiv metadata and abstracts only.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **Requires network access.** Every call is an HTTPS request to the arXiv API; no API key is needed.
- **Runs in process on CPU.** There is no environment to build and no worker to start; runtime is dominated by remote retrieval.
