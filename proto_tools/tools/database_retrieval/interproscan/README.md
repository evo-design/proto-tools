<a href="https://bio-pro.mintlify.app/tools/database-retrieval/interproscan"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# InterPro

## Overview

`interproscan-fetch` retrieves protein domain annotations from [InterPro](https://www.ebi.ac.uk/interpro/) — the EMBL-EBI aggregator that integrates Pfam, SMART, PROSITE, Gene3D / CATH-Gene3D, Panther, PIRSF, and a dozen other member databases under unified InterPro IDs (`IPRxxxxxx`). Two access paths share the same Output schema: a fast direct lookup by UniProt accession, and a submit-and-poll path against EBI's iprscan5 service for novel sequences. Each domain row carries a 1-indexed inclusive `start` / `end`, a unified `type` label (`family` / `domain` / `repeat` / `active_site` / `conserved_site` / `homologous_superfamily` / `binding_site` / `ptm`), the parent InterPro accession when integrated, and optional GO / pathway cross-references.

## Background

**What does this tool measure/predict?**
InterPro is the authoritative classification of protein families, domains, conserved sites, and homologous superfamilies. A single InterPro entry (e.g. `IPR011615` "p53 DNA-binding domain") aggregates orthogonal models from multiple member databases — an HMM in Pfam, a profile in SMART, a structural model in CATH-Gene3D — into one curated grouping. InterProScan is the analysis pipeline that runs every member-DB model against a sequence; iprscan5 exposes that pipeline as a public REST service.

**Why is this important?**
Domain-aware annotation is the load-bearing primitive for most protein-design workflows:

- **"Preserve the active site, redesign the rest":** lock conserved residues identified by an `active_site` or `conserved_site` annotation; let everything else vary.
- **Antibody / scaffold redesign:** keep the framework constant and only redesign within explicitly typed CDR / repeat regions.
- **Enzyme refinement:** identify catalytic / binding-site residues that must be preserved across rounds of directed evolution.
- **Constraint scoring:** weight per-residue mutation penalties by domain importance.

UniProt's per-protein annotations are partial; InterPro is the reference catalog.

**Scientific foundation:**
InterPro 2025 (Blum et al., *Nucleic Acids Res*) integrates 14 member databases into ~46,000 InterPro entries covering ~85% of UniProtKB. For a sequence, iprscan5 runs each member's HMM / profile / regex / structural model in parallel and returns matches with member-DB scores plus the parent InterPro grouping. The direct REST path (`/interpro/api/entry/all/protein/uniprot/{acc}/`) returns the pre-computed integration of those matches keyed by UniProt accession — the same data without the per-job submission cost.

## How It Works

**Two paths, one Output:**

| Path | Input | Endpoint | Wall-clock |
|---|---|---|---|
| Direct lookup | UniProt accession | `interpro/api/entry/all/protein/uniprot/{acc}/?page_size=N` (paginated) | < 1 s typical |
| Submit-and-scan | Raw protein sequence | `Tools/services/rest/iprscan5/run/` → poll `/status/{id}` → fetch `/result/{id}/json` | 2–30 min (queue depth dependent) |

The submit path requires a contact email per EBI policy (`config.email`). Both paths flatten matches into the same `InterProDomain` row schema, so downstream consumers don't need to know which path produced the output.

**Method overview:**
- **Direct path:** single HTTP GET per page; walks the opaque `next` cursor until exhausted. Each result has one `metadata` block (the InterPro / member-DB entry) and one `proteins[0].entry_protein_locations[]` list (the per-protein matches). Each `fragments[]` entry becomes one row.
- **Submit path:** multipart POST with `email` + `sequence` + optional `appl[]` returns a plain-text job ID. We poll `/status/{id}` (vocabulary: `RUNNING` / `QUEUED` / `FINISHED` / `ERROR` / `FAILURE` / `NOT_FOUND`) every 3 s — matching EBI's reference [iprscan5 Python client](https://github.com/ebi-jdispatcher/webservice-clients/blob/master/python/iprscan5.py) — up to a 30 min cap, then GET `/result/{id}/json`. The same flattening logic lifts each `match.locations[]` entry into one row.

**Key assumptions:**
- The provided UniProt accession is in InterPro's coverage (true for ~85% of UniProtKB).
- For the submit path: `config.email` is a valid contact mailbox.
- The canonical UniProt sequence is the relevant isoform.

**Limitations:**
- Direct path returns 404 for accessions InterPro hasn't indexed (very recent UniProt entries, removed accessions).
- iprscan5 is rate-limited by queue depth, not by per-IP throttling — a busy server can mean 30 min wall-clock for a 350-aa sequence.
- Member-DB scores have heterogeneous units (e-value vs bit-score) — the row's `score` field carries whichever the source DB published.

**Computational requirements:**
- **Hardware:** CPU only, network access required. No GPU, no isolated env.
- **Runtime:** Direct path < 1 s. Submit path 2–30 min depending on queue.

## Input Parameters

| Parameter | Type | Description |
|---|---|---|
| `uniprot_id` | `str \| None` | UniProt accession for direct REST lookup (e.g. `"P04637"`). Provide exactly one of `uniprot_id` or `sequence`. |
| `sequence` | `str \| None` | Raw protein sequence for the submit-and-scan path. Requires `config.email`. |

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `email` | `str \| None` | `None` | Contact email — required by EBI when submitting a sequence; ignored on the direct path. |
| `applications` | `list[InterProApp] \| None` | `None` | Submit-only — restrict iprscan5 to a subset of member databases. `None` runs the EBI default set. |
| `include_go_terms` | `bool` | `True` | Include GO term cross-references in each row's `go_terms` list. |
| `include_pathways` | `bool` | `True` | Include pathway IDs (Reactome, MetaCyc) in each row's `pathways` list. |

`InterProApp` enumerates the 24 casings EBI's `parameterdetails` endpoint accepts: `PfamA`, `Panther`, `Gene3d`, `SuperFamily`, `SMART`, `PrositeProfiles`, `PrositePatterns`, `PRINTS`, `PIRSF`, `FunFam`, `HAMAP`, `CDD`, `NCBIfam`, `SFLD`, `Coils`, `MobiDBLite`, `Phobius`, `SignalP`, `SignalP_EUK`, `SignalP_GRAM_POSITIVE`, `SignalP_GRAM_NEGATIVE`, `AntiFam`, `PIRSR`, `TMHMM`. (Note `PfamA` not `Pfam`, `Gene3d` not `Gene3D`.)

## Output Specification

| Field | Type | Description |
|---|---|---|
| `accession` | `str \| None` | Resolved UniProt accession; `None` when the sequence path returns no UniProt cross-reference. |
| `sequence_length` | `int \| None` | Length of the queried protein, when reported. |
| `domains` | `list[InterProDomain]` | All hits across all member databases. |
| `num_domains` | `int` | `len(domains)`. |
| `job_id` | `str` | iprscan5 job ID (sequence path); empty for direct lookup. |
| `source_url` | `str` | Canonical InterPro entry URL (direct) or iprscan5 result URL (submit). |
| `raw_entries` | `list[dict[str, Any]]` | Raw JSON entries / matches for advanced consumers. |

`InterProDomain` rows carry `accession` (the member-DB ID), `name`, `type`, `member_database`, `integrated_ipr` (parent InterPro ID), `start` / `end` (1-indexed inclusive), `score`, `model`, `representative`, `go_terms`, `pathways`.

## Best Practices & Gotchas

**Common mistakes:**
1. **Using `"Pfam"` instead of `"PfamA"`** in `applications`: EBI's iprscan5 expects `PfamA` exactly. The `InterProApp` Literal enforces the correct casing at validation time.
2. **Submitting a sequence without an email:** `config.email` is mandatory for the iprscan5 path; the wrapper raises `ValueError` with a clear message rather than 400-ing on the server.
3. **Treating `score` as a uniform e-value:** the field carries whichever score the member DB published. For Pfam this is an e-value; for some structural-classifier members it's a bit-score. Filter per `member_database` if you need uniform semantics.

**Tips:**
- For programmatic chains, prefer the **direct path** when you have an accession — it's two orders of magnitude faster.
- Filter by `representative=True` to keep only InterPro's chosen representative match per parent IPR entry.
- Disable GO / pathway lookups (`include_go_terms=False`, `include_pathways=False`) when you're only interested in domain coordinates — saves a small amount of payload.

## Quick Start Examples

**Example 1: Direct lookup by UniProt accession (no email required):**
```python
from proto_tools.tools.database_retrieval import (
    InterProScanFetchInput,
    InterProScanFetchConfig,
    run_interproscan_fetch,
)

output = run_interproscan_fetch(InterProScanFetchInput(uniprot_id="P04637"))
assert output.success
print(f"{output.num_domains} hits across {sorted({d.member_database for d in output.domains})}")

dbd = next(d for d in output.domains if d.member_database == "pfam" and d.name.startswith("P53"))
print(f"DBD: {dbd.start}-{dbd.end} ({dbd.type})")  # → 100-288 (domain)
```

**Example 2: chained workflow — UniProt → InterPro for "preserve the active site, redesign the rest":**
```python
from proto_tools.tools.database_retrieval import (
    InterProScanFetchInput, run_interproscan_fetch,
    UniProtFetchInput, run_uniprot_fetch,
)

# Resolve the canonical TP53 sequence and its length.
uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id="P04637"))
length = uniprot.length
assert length is not None

# Pull all InterPro hits for the same accession.
ipr = run_interproscan_fetch(InterProScanFetchInput(uniprot_id="P04637"))

# Build a per-residue lock mask: True at any residue inside an active_site /
# conserved_site / binding_site / ptm match — i.e. positions a redesign loop
# should not touch.
lock_mask = [False] * length
for domain in ipr.domains:
    if domain.type in {"active_site", "conserved_site", "binding_site", "ptm"}:
        for i in range(domain.start - 1, domain.end):
            lock_mask[i] = True

print(f"{sum(lock_mask)} of {length} residues locked")
```

## References

**Primary publication:**
- Blum, M., Andreeva, A., Florentino, L. C., Chuguransky, S. R., Grego, T., Hobbs, E., Pinto, B. L., Orr, A., Paysan-Lafosse, T., Ponamareva, I., Salazar, G. A., Bordin, N., Bork, P., Bridge, A., Colwell, L., Gough, J., Haft, D. H., Letunic, I., Llinares-López, F., Marchler-Bauer, A., Meng-Papaxanthos, L., Mi, H., Natale, D. A., Orengo, C. A., Pandurangan, A. P., Piovesan, D., Rivoire, C., Sigrist, C. J. A., Thanki, N., Thibaud-Nissen, F., Thomas, P. D., Tosatto, S. C. E., Wu, C. H., & Bateman, A. (2025). "InterPro: the protein sequence classification resource in 2025." *Nucleic Acids Research*, 53(D1), D444–D456. [DOI: 10.1093/nar/gkae1082](https://doi.org/10.1093/nar/gkae1082)

**Implementation:**
- InterProScan: [https://github.com/ebi-pf-team/interproscan](https://github.com/ebi-pf-team/interproscan)
- InterPro database: [https://www.ebi.ac.uk/interpro/](https://www.ebi.ac.uk/interpro/)
- iprscan5 REST: [https://www.ebi.ac.uk/Tools/services/rest/iprscan5/](https://www.ebi.ac.uk/Tools/services/rest/iprscan5/)

## Related Tools

- **`uniprot-fetch`**: Resolve a gene symbol / organism pair to a canonical UniProt accession before calling `interproscan-fetch`. Also gives sequence length for `lock_mask`-style outputs.
- **`alphafold-db-fetch`**: Pull AlphaFold's predicted structure + per-residue pLDDT for the same accession. Combined with InterPro domain types, this lets you weight design constraints by both functional importance (InterPro) and structural confidence (pLDDT).
- **`alphamissense-fetch`**: Per-residue pathogenicity. Composes naturally with InterPro-derived locks for "lock the active site, allow benign substitutions elsewhere" workflows.
