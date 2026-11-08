<a href="https://bio-pro.mintlify.app/tools/database-retrieval/europepmc"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Europe PMC

![Europe PMC](https://proto-bio.github.io/proto-assets/images/tool/europepmc/hero.png)

> [!NOTE]
> **License:** Europe PMC retrieves data from literature records and full text served by Europe PMC, distributed under source-specific terms. Attribution to literature records and full text served by Europe PMC is required when the data is redistributed. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://europepmc.org/Copyright) for full terms.

## Overview

[Europe PMC](https://europepmc.org/) is EMBL-EBI's open literature database, covering PubMed abstracts, PubMed Central full-text articles, and preprints from bioRxiv, medRxiv, and other servers. This toolkit exposes two tools over its REST API: `europepmc-search` finds papers by free text or field query and reports identifiers, citation details, and open-access status, and `europepmc-fulltext` returns an open-access article as plain-text sections that can be searched for an exact passage. Both run on CPU in process and need only network access.

## Background

[Europe PMC](https://europepmc.org/) is run by [EMBL-EBI](https://www.ebi.ac.uk/) as a partner of [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/). It indexes tens of millions of records, mostly PubMed abstracts, along with millions of full-text articles and preprints. Each record has a source code (`MED` for PubMed, `PMC` for PubMed Central, `PPR` for preprints) and an identifier within that source, and carries a PMID, PMCID, or DOI where one exists.

Full text is available for the open-access subset of PubMed Central and Europe PMC, published under licenses such as CC BY that permit text mining. Articles arrive as [JATS](https://jats.nlm.nih.gov/) XML, the standard publishing markup for journal articles, which marks the title, abstracts, nested sections, figures, and tables. Reading an article as text rather than as a PDF makes it possible to find and quote a reported value, such as a binding constant or a list of hotspot residues, exactly as the authors wrote it.

### Learning Resources

- [Europe PMC REST API](https://europepmc.org/RestfulWebService) (EMBL-EBI) - endpoints, parameters, and response formats.
- [Europe PMC search syntax](https://europepmc.org/searchsyntax) (EMBL-EBI) - field codes such as `TITLE:`, `AUTH:`, `PUB_YEAR:`, `SRC:`, and `OPEN_ACCESS:`.

## Tools

### Europe PMC Search (`europepmc-search`)

Searches Europe PMC and returns ranked hits with source, PMID, PMCID, DOI, title, authors, journal, year, open-access and full-text flags, citation count, abstract, and the Europe PMC page (`source_url`).

#### Applications

Use search to find the paper behind a design fact, such as the study that reports a binder's affinity or an alanine scan of an interface, and to tell which hits can be read in full.

#### Usage Tips

- Queries use Europe PMC syntax: combine terms with `AND`, `OR`, and `NOT`, quote phrases, and restrict fields with `TITLE:`, `ABSTRACT:`, `AUTH:`, `PUB_YEAR:2020`, or `PUB_YEAR:[2018 TO 2022]`.
- Add `OPEN_ACCESS:y` to keep only papers whose full text `europepmc-fulltext` can fetch, or `HAS_FT:y` for any paper with full text in Europe PMC.
- Add `SRC:PPR` to search preprints only, including bioRxiv and medRxiv. For a preprint, `published_pmid` gives the PMID of its journal version, when there is one.
- Set `include_abstracts=False` for a compact hit list; each result's `total_count` reports how many records matched.
- A query Europe PMC rejects, such as one with a syntax error, gets an `error` and no hits; the other queries still run.
- `paperclip-literature-*` covers the same PMC full text plus bioRxiv, medRxiv and arXiv, with line-numbered reading for citing evidence, but needs a `PAPERCLIP_API_KEY`. Use these tools when you have no key, and `ncbi-esearch` with `db="pubmed"` for PubMed's own query syntax and MeSH terms.

### Europe PMC Full Text (`europepmc-fulltext`)

Fetches open-access articles by PMCID and returns each as plain-text sections: the title, every abstract, body and back-matter sections under their heading path, figure captions, and tables with cells separated by `|`.

#### Applications

Use full text to find and quote the passage, table row, or figure caption that supports a claim, such as a reported dissociation constant or the residues a mutagenesis study found essential for binding.

#### Usage Tips

- Pass the `pmcid` of a search hit with `is_open_access=True`. A PMCID that is not in Europe PMC or not open access gets an `error` and no sections; the other articles are still read.
- Each section has a `kind` (`title`, `abstract`, `section`, `figure`, `table`) and a `heading` such as `Results > Binding assays` or `Table 2`, so a match can be cited by where it appears.
- `document.text` joins every section into one string for searching, and the output exports to `txt` as well as `json`.
- Reference lists are left out. Inline formulas are kept as plain text, so superscripts and subscripts run together (`10^12` reads as `1012`).
- Check each article's `license` before redistributing its text.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These notes apply to both Europe PMC tools.

- **Requires network access.** Every call is an HTTPS request to the Europe PMC REST API; no API key is needed.
- **Runs in process on CPU.** There is no environment to build and no worker to start; runtime is dominated by remote retrieval.
