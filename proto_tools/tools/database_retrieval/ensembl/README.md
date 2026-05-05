<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ensembl"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Ensembl

## Overview

The `ensembl` toolkit wraps Ensembl's REST API for DNA-design context. Two tools share the toolkit:

- **`ensembl-fetch`** — gene / transcript / exon hierarchy lookup, sequence retrieval, region overlap, and cross-references. One tool with an `endpoint:` switch over five fetch endpoints.
- **`ensembl-vep`** — variant-effect prediction from HGVS notation; returns per-transcript consequences (Sequence Ontology terms, SIFT / PolyPhen, codons, AA changes).

Both call `rest.ensembl.org` (GRCh38) by default; setting `assembly="GRCh37"` routes to `grch37.rest.ensembl.org`. HTTP-only — no isolated env, no GPU.

## Background

NCBI gives sequence; Ensembl gives **annotation**. Without it, DNA design collapses to context-free sequence manipulation — promoter design, splice-site design, regulatory-element work, and variant-aware redesign all suffer. Specifically:

- **Exon boundaries + canonical transcripts** are essential for splice-site-aware oligo design.
- **Regulatory features** (promoters, enhancers, TF binding motifs) anchor cis-element design.
- **Variant consequences** (VEP) tell you whether a designed substitution introduces a stop codon, disrupts splicing, or hits a known clinical variant.
- **Cross-references** (`xrefs`) connect Ensembl IDs to UniProt / NCBI / clinical resources, enabling cross-tool composition with `uniprot-fetch` and the rest of the database-retrieval stack.

The Ensembl REST API is the canonical programmatic interface to the same data the Ensembl genome browser shows; rate-limited at 55,000 req/hour with `Retry-After` honored automatically by `proto-tools`' shared HTTP session.

## How It Works

**`ensembl-fetch`** dispatches on `config.endpoint` to one of five Ensembl REST endpoints:

| `endpoint` | URL | Returns |
|---|---|---|
| `lookup_id` | `GET /lookup/id/{id}?expand=1` | `EnsemblGene` with nested `Transcript[]` / `Exon[]` / `Translation` |
| `lookup_symbol` | `GET /lookup/symbol/{species}/{symbol}?expand=1` | Same shape as `lookup_id` |
| `sequence` | `GET /sequence/id/{id}?type={genomic\|cdna\|cds\|protein}` | `EnsemblSequence` |
| `overlap` | `GET /overlap/id/{id}?feature={gene\|exon\|regulatory\|...}` | `list[EnsemblOverlapFeatureRecord]` |
| `xrefs` | `GET /xrefs/id/{id}` or `/xrefs/symbol/{species}/{symbol}` | `list[EnsemblXref]` |

The result is a discriminated union driven by `output.endpoint`. Pascal-case keys (`Transcript`, `Exon`, `Translation`) are preserved verbatim from the API to round-trip cleanly through `model_dump()`.

**`ensembl-vep`** wraps `GET /vep/{species}/hgvs/{hgvs}`. The HGVS form accepts genomic (`9:g.22125504G>C`), coding (`ENST00000357654:c.5074G>A`), and protein (`ENSP00000418960:p.Tyr124Cys`) notations.

### Rate limiting + transient errors

Ensembl's 429 responses include `Retry-After`; urllib3's `Retry()` honors it automatically (default `respect_retry_after_header=True`). 5xx + connection resets are retried up to the configured budget. 4xx (bad ID, malformed HGVS) propagates immediately.

## Input Parameters

### `ensembl-fetch`

| Field | Type | Description |
|---|---|---|
| `ensembl_id` | `str \| None` | Ensembl ID (`ENSG...`, `ENST...`, `ENSP...`). Required for `lookup_id`, `sequence`, `overlap`. |
| `symbol` | `str \| None` | Gene symbol (e.g. `BRCA1`). Required for `lookup_symbol`. Either field works for `xrefs`. |

### `ensembl-vep`

| Field | Type | Description |
|---|---|---|
| `hgvs` | `str` | HGVS notation. Genomic / coding / protein forms all accepted. |

## Configuration

### `ensembl-fetch`

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `Literal["lookup_id","lookup_symbol","sequence","overlap","xrefs"]` | `"lookup_id"` | Which Ensembl REST endpoint to call. |
| `species` | `Literal[5 species]` | `"homo_sapiens"` | Used by `lookup_symbol` and symbol-form `xrefs`. |
| `assembly` | `Literal["GRCh38","GRCh37"]` | `"GRCh38"` | GRCh37 routes to `grch37.rest.ensembl.org`. |
| `sequence_type` | `Literal["genomic","cdna","cds","protein"]` | `"cdna"` | Sequence flavor (`sequence` endpoint only). |
| `overlap_feature` | `Literal[...]` | `"gene"` | Feature class (`overlap` endpoint only). |
| `expand` | `bool` | `True` | Include transcripts/exons (`lookup_*` only). |

### `ensembl-vep`

| Field | Type | Default | Description |
|---|---|---|---|
| `species` | `Literal[5 species]` | `"homo_sapiens"` | Species slug for VEP. |
| `assembly` | `Literal["GRCh38","GRCh37"]` | `"GRCh38"` | Genome assembly. |

## Output Specification

### `ensembl-fetch`

| Field | Type | Description |
|---|---|---|
| `endpoint` | `EnsemblEndpoint` | Echoes the endpoint that produced this result. |
| `result` | `EnsemblGene \| EnsemblSequence \| list[EnsemblXref] \| list[EnsemblOverlapFeatureRecord]` | Parsed payload (concrete type depends on `endpoint`). |
| `source_url` | `str` | Final URL hit. |
| `raw_payload` | `dict \| list[dict]` | Raw API JSON. |

### `ensembl-vep`

| Field | Type | Description |
|---|---|---|
| `consequences` | `list[EnsemblVEPConsequence]` | One record per VEP input. |
| `num_consequences` | `int` | `len(consequences)`. |
| `source_url` | `str` | Final URL hit. |
| `raw_payload` | `list[dict]` | Raw API JSON. |

## Quick Start Examples

**Example 1: lookup BRCA1 by symbol (returns gene + 47 transcripts)**

```python
from proto_tools.tools.database_retrieval import (
    EnsemblFetchInput,
    EnsemblFetchConfig,
    run_ensembl_fetch,
)

out = run_ensembl_fetch(
    EnsemblFetchInput(symbol="BRCA1"),
    EnsemblFetchConfig(endpoint="lookup_symbol"),
)
assert out.success
gene = out.result
print(f"{gene.display_name} ({gene.id}): {len(gene.Transcript)} transcripts; canonical={gene.canonical_transcript}")
# BRCA1 (ENSG00000012048): 47 transcripts; canonical=ENST00000357654.9
```

**Example 2: protein sequence of the canonical transcript**

```python
out = run_ensembl_fetch(
    EnsemblFetchInput(ensembl_id="ENST00000357654"),
    EnsemblFetchConfig(endpoint="sequence", sequence_type="protein"),
)
print(f"{out.result.id}: {len(out.result.seq)} aa")
```

**Example 3: variant effect prediction**

```python
from proto_tools.tools.database_retrieval import EnsemblVEPInput, run_ensembl_vep

out = run_ensembl_vep(EnsemblVEPInput(hgvs="9:g.22125504G>C"))
top = out.consequences[0]
print(f"{top.allele_string}: {top.most_severe_consequence}, {len(top.transcript_consequences)} transcripts affected")
```

**Example 4: chained workflow — Ensembl → UniProt cross-reference**

```python
from proto_tools.tools.database_retrieval import (
    EnsemblFetchInput, EnsemblFetchConfig, run_ensembl_fetch,
    UniProtFetchInput, run_uniprot_fetch,
)

xrefs = run_ensembl_fetch(
    EnsemblFetchInput(ensembl_id="ENSG00000012048"),
    EnsemblFetchConfig(endpoint="xrefs"),
)
uniprot_id = next(x.primary_id for x in xrefs.result if x.dbname == "Uniprot_gn")
uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id=uniprot_id))
print(f"BRCA1 UniProt: {uniprot.accession}, {uniprot.length} aa")
```

## Best Practices & Gotchas

**Common mistakes:**

1. **Forgetting `expand=True` on `lookup_*`:** without it, the result has no nested `Transcript[]`. The default is `True` for this reason; only flip to `False` if you genuinely just want the gene-level fields.
2. **Mixing assemblies:** GRCh38 and GRCh37 coordinates differ. Pin `config.assembly` per workflow; don't assume one default fits both.
3. **Treating `overlap` records as fully typed:** `feature_type` differs per query (`gene` vs `regulatory` vs `variation`) and the typed fields are the intersection. Use `record.raw` for feature-specific keys.
4. **HGVS protein form for VEP:** `ENSP00000418960:p.Tyr124Cys` works, but the corresponding cdNA/genomic notation is often more reliable when the protein ID has multiple matching transcripts.

**Tips:**

- For gene-symbol → genomic-context flows, chain `lookup_symbol` → `sequence` (with `sequence_type="genomic"`) on the canonical transcript ID.
- `overlap_feature="regulatory"` returns Ensembl Regulatory Build features (CTCF binding sites, open chromatin, etc.) — useful for cis-element design.

## References

**Primary publication:**

- Dyer, S. C., Austine-Orimoloye, O., Azov, A. G., Barba, M., Barnes, I., Barrera-Enriquez, V. P., et al. (2025). "Ensembl 2025." *Nucleic Acids Research*, 53(D1), D948–D957. [DOI: 10.1093/nar/gkae1071](https://doi.org/10.1093/nar/gkae1071)

**Implementation:**

- Ensembl REST source: [https://github.com/Ensembl/ensembl-rest](https://github.com/Ensembl/ensembl-rest)
- Ensembl REST live docs: [https://rest.ensembl.org/](https://rest.ensembl.org/)
- Ensembl genome browser: [https://www.ensembl.org/](https://www.ensembl.org/)

## Related Tools

- **`uniprot-fetch`**: Resolve Ensembl `xrefs` rows with `dbname="Uniprot_gn"` to a UniProt accession; chain into `interproscan-fetch` / `alphamissense-fetch` for protein-level annotations.
- **`ncbi-efetch`**: Pull GenBank / RefSeq records when you need NCBI-side identifiers; Ensembl's `xrefs` endpoint also exposes `EntrezGene` mappings.
- **`alphafold-db-fetch`**: Once an Ensembl protein is mapped to UniProt, fetch its predicted structure for downstream design.
