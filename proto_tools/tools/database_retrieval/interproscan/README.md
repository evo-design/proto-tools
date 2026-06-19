<a href="https://bio-pro.mintlify.app/tools/database-retrieval/interproscan"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# InterPro

![InterPro](https://proto-bio.github.io/proto-assets/images/tool/interproscan/hero.png)

> [!NOTE]
> **License:** InterPro retrieves data from the InterPro classification, distributed under the EMBL-EBI Terms of Use. The client wrapper code is MIT-licensed. Please refer to [the data terms](https://www.ebi.ac.uk/about/terms-of-use/) for full terms.

## Overview

[InterPro](https://www.ebi.ac.uk/interpro/) integrates protein signatures from member databases such as [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/), [SMART](https://smart.embl.de/), [PROSITE](https://prosite.expasy.org/), [CATH-Gene3D](https://www.cathdb.info/), [Panther](https://www.pantherdb.org/), and [PIRSF](https://proteininformationresource.org/pirsf/) into unified entries describing protein families, domains, and conserved sites. The `interproscan-fetch` tool returns one `InterProDomain` row schema over two paths: a direct lookup of precomputed InterPro annotations for a UniProt accession via the InterPro REST API, or submission of a raw protein sequence to EBI's InterProScan job service, which polls to completion and parses the result. It runs on CPU and requires only network access.

## Background

[InterPro](https://www.ebi.ac.uk/interpro/) ([Blum et al., 2025](https://doi.org/10.1093/nar/gkae1082)) is a freely accessible classification of protein families, domains, conserved sites, and homologous superfamilies, maintained by [EMBL-EBI](https://www.ebi.ac.uk/). A protein family is a set of evolutionarily related proteins that descend from a shared ancestor and share detectable sequence similarity, typically along with a common three-dimensional fold or biological function. A single InterPro entry groups orthogonal member-database signatures, such as a [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam/) HMM and a [CATH-Gene3D](https://www.cathdb.info/) structural model, under one accession. InterProScan is the analysis pipeline that runs the member-database models against a sequence, and EBI exposes it as a public web service.

Internally, the direct path issues `GET https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{accession}`, walking the opaque `next` cursor across paginated responses until the result set is exhausted. The submit path issues `POST https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run/` with a required contact `email` and the sequence, receives a plain-text job ID, polls `/status/{job_id}` every three seconds until the job reaches `FINISHED`, then fetches `/result/{job_id}/json`. Both paths flatten matches into the same row schema, with each member-database match contributing rows carrying 1-indexed inclusive `start` and `end` coordinates to match biological residue selection conventions, a unified `type` label, the parent InterPro accession when integrated, and optional [Gene Ontology](https://geneontology.org/) (GO) and pathway cross-references.

Annotations and their provenance come directly from EMBL-EBI's official [InterPro REST API](https://interpro-documentation.readthedocs.io/) and iprscan5 service. Results reflect the live resource at query time rather than a fixed release snapshot.

### Learning Resources

- [InterPro documentation](https://interpro-documentation.readthedocs.io/) (EMBL-EBI) - official documentation covering InterPro entries, member databases, and the REST API.
- [Job Dispatcher web services documentation](https://www.ebi.ac.uk/jdispatcher/docs/webservices/) (EMBL-EBI) - reference for the iprscan5 submit-and-poll REST service, including fair-use guidance.

## Tools

### InterProScan Fetch (`interproscan-fetch`)

Retrieves InterPro domain annotations for a protein, either by direct REST lookup of a UniProt accession or by submitting a raw sequence to the iprscan5 service, and returns the resolved accession, sequence length, the list of member-database hits, the source URL, the iprscan5 job ID on the sequence path, and the raw API entries.

#### Applications

Use this to attach domain, family, and site annotation to a protein before design or filtering: identify the residues of an `active_site` or `conserved_site` match to lock before a redesign loop, partition a sequence into typed family and domain regions, or collect GO and pathway cross-references for functional grouping. The resolved accession and the parent InterPro identifiers compose with the [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot) and [AlphaFold DB](https://bio-pro.mintlify.app/tools/database-retrieval/alphafold-db) tools for accession resolution and structural context.

#### Usage Tips

- **The sequence-submission path requires a contact email.** When `sequence` is provided, `config.email` must be set. Provide it either via the `email` config attribute or via the `INTERPROSCAN_EMAIL` environment variable; an explicit config value overrides the env var. The tool raises a clear `ValueError` before contacting the server if neither is set. The direct accession path ignores `email`.
- **Provide exactly one of `uniprot_id` or `sequence`.** The input validator rejects a call that supplies both or neither.
- **`score` units are not uniform across rows.** The field carries whichever value the source member database publishes, an e-value for some databases and a bit-score for others, so filter by `member_database` before comparing scores.
- **The direct path returns no pathway cross-references.** InterPro's UniProt-keyed endpoint does not surface pathway data, so `pathways` stays empty on that path regardless of configuration. Pathways are only populated on the sequence-submission path.
- **A direct lookup raises when the accession is not indexed.** Very recent or removed UniProt accessions outside InterPro's coverage return no entries, surfacing as a `ValueError` rather than an empty result.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every InterProScan tool in this toolkit (`interproscan-fetch`).

- **Requires network access.** The tool calls the live InterPro REST API and iprscan5 service. It does not run offline and keeps no local copy of the data.
- **The sequence-submission path requires a contact email for identification.** This email lets EBI contact the submitter about job issues. It does not raise any bandwidth or rate allowance.
- **Sequence submissions are subject to a fair-use concurrency cap.** EBI asks that jobs be submitted in batches of no more than 30 concurrent jobs.
