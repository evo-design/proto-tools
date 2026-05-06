<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ensembl"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Ensembl

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

The `ensembl` toolkit wraps Ensembl's REST API for DNA-design context. Five tools share the toolkit, one per endpoint family — each with a concretely typed result so JSON Schema / MCP consumers see the real shape:

- **`ensembl-lookup`** — gene lookup by Ensembl gene ID **or** gene symbol → `EnsemblGene`.
- **`ensembl-sequence`** — DNA / cDNA / CDS / protein sequence retrieval by Ensembl ID → `EnsemblSequence`.
- **`ensembl-overlap`** — features overlapping a region (genes, exons, regulatory elements, motifs, variants) → `list[EnsemblOverlapFeatureRecord]`.
- **`ensembl-xrefs`** — cross-references from an Ensembl ID to external databases (UniProt, EntrezGene, RefSeq) → `list[EnsemblXref]`.
- **`ensembl-vep`** — variant-effect prediction from HGVS notation; returns per-transcript consequences (Sequence Ontology terms, SIFT / PolyPhen, codons, AA changes).

All five call `rest.ensembl.org` (GRCh38) by default; setting `assembly="GRCh37"` routes to `grch37.rest.ensembl.org`. HTTP-only — no isolated env, no GPU.

## Background

NCBI gives sequence; Ensembl gives **annotation**. Without it, DNA design collapses to context-free sequence manipulation — promoter design, splice-site design, regulatory-element work, and variant-aware redesign all suffer. Specifically:

- **Exon boundaries + canonical transcripts** are essential for splice-site-aware oligo design.
- **Regulatory features** (promoters, enhancers, TF binding motifs) anchor cis-element design.
- **Variant consequences** (VEP) tell you whether a designed substitution introduces a stop codon, disrupts splicing, or hits a known clinical variant.
- **Cross-references** (`xrefs`) connect Ensembl IDs to UniProt / NCBI / clinical resources, enabling cross-tool composition with `uniprot-fetch` and the rest of the database-retrieval stack.

The Ensembl REST API is the canonical programmatic interface to the same data the Ensembl genome browser shows; rate-limited at 55,000 req/hour with `Retry-After` honored automatically by `proto-tools`' shared HTTP session.

## Tools

### Ensembl Lookup (`ensembl-lookup`)

Fetch a gene record via Ensembl REST.

### Ensembl Overlap (`ensembl-overlap`)

Fetch overlapping features for a region from Ensembl REST.

### Ensembl Sequence (`ensembl-sequence`)

Fetch a sequence record from Ensembl REST.

### Ensembl VEP (`ensembl-vep`)

Submit an HGVS notation to Ensembl VEP and parse the consequence list.

### Ensembl Xrefs (`ensembl-xrefs`)

Fetch cross-references from Ensembl REST.

## How It Works

Each tool wraps one Ensembl REST endpoint family:

| Tool | URL | Returns |
|---|---|---|
| `ensembl-lookup` | `GET /lookup/id/{id}?object_type=gene` or `/lookup/symbol/{species}/{symbol}`; add `expand=1` for nested transcripts/exons | `EnsemblGene` |
| `ensembl-sequence` | `GET /sequence/id/{id}?type={genomic\|cdna\|cds\|protein}` | `EnsemblSequence` |
| `ensembl-overlap` | `GET /overlap/id/{id}?feature={gene\|exon\|regulatory\|...}` | `list[EnsemblOverlapFeatureRecord]` |
| `ensembl-xrefs` | `GET /xrefs/id/{id}` | `list[EnsemblXref]` |
| `ensembl-vep` | `GET /vep/{species}/hgvs/{hgvs}` | `list[EnsemblVEPConsequence]` |

Pascal-case keys (`Transcript`, `Exon`, `Translation`) are preserved verbatim from the API to round-trip cleanly through `model_dump()`.

`ensembl-vep`'s HGVS form accepts genomic (`9:g.22125504G>C`), coding (`ENST00000357654:c.5074G>A`), and protein (`ENSP00000418960:p.Tyr124Cys`) notations.

### Rate limiting + transient errors

Ensembl's 429 responses include `Retry-After`; urllib3's `Retry()` honors it automatically (default `respect_retry_after_header=True`). 5xx + connection resets are retried up to the configured budget. 4xx (bad ID, malformed HGVS) propagates immediately.

## Input Parameters

### `ensembl-lookup`

| Field | Type | Description |
|---|---|---|
| `ensembl_id` | `str \| None` | Ensembl gene ID (for example, `ENSG00000012048`). |
| `symbol` | `str \| None` | Gene symbol (e.g. `BRCA1`). Requires `config.species`. |

Exactly one of the two must be provided.

### `ensembl-sequence` / `ensembl-overlap` / `ensembl-xrefs`

| Field | Type | Description |
|---|---|---|
| `ensembl_id` | `str` | Ensembl ID (`ENSG...`, `ENST...`, `ENSP...`). |

### `ensembl-vep`

| Field | Type | Description |
|---|---|---|
| `hgvs` | `str` | HGVS notation. Genomic / coding / protein forms all accepted. |

## Configuration

All five tools share `assembly: Literal["GRCh38","GRCh37"]` (default `"GRCh38"`; `GRCh37` routes to `grch37.rest.ensembl.org`).

| Tool | Field | Type | Default | Description |
|---|---|---|---|---|
| `ensembl-lookup` | `species` | `Literal[5 species]` | `"homo_sapiens"` | Used when `symbol` is provided. |
| `ensembl-lookup` | `expand` | `bool` | `False` | Advanced. Include transcripts / exons in the response. |
| `ensembl-sequence` | `sequence_type` | `Literal["genomic","cdna","cds","protein"]` | `"genomic"` | Sequence flavor. |
| `ensembl-overlap` | `overlap_feature` | `Literal["gene","transcript","exon",...]` | `"gene"` | Feature class to retrieve. |
| `ensembl-vep` | `species` | `Literal[5 species]` | `"homo_sapiens"` | Species slug for VEP. |

## Output Specification

Every output carries `source_url: str` and `raw_payload: dict | list[dict]` alongside the typed payload.

| Tool | Output payload field |
|---|---|
| `ensembl-lookup` | `result: EnsemblGene` |
| `ensembl-sequence` | `results: list[EnsemblSequence]` (length 1 unless `multiple_sequences=True`) |
| `ensembl-overlap` | `result: list[EnsemblOverlapFeatureRecord]` |
| `ensembl-xrefs` | `result: list[EnsemblXref]` |
| `ensembl-vep` | `consequences: list[EnsemblVEPConsequence]`, `num_consequences: int` |

## Quick Start Examples

**Example 1: lookup BRCA1 by symbol (returns gene + 47 transcripts)**

```python
from proto_tools.tools.database_retrieval import EnsemblLookupConfig, EnsemblLookupInput, run_ensembl_lookup

out = run_ensembl_lookup(EnsemblLookupInput(symbol="BRCA1"), EnsemblLookupConfig(expand=True))
gene = out.result
print(f"{gene.display_name} ({gene.id}): {len(gene.Transcript)} transcripts; canonical={gene.canonical_transcript}")
# BRCA1 (ENSG00000012048): 47 transcripts; canonical=ENST00000357654.9
```

**Example 2: protein sequence of the canonical transcript**

```python
from proto_tools.tools.database_retrieval import EnsemblSequenceInput, EnsemblSequenceConfig, run_ensembl_sequence

out = run_ensembl_sequence(
    EnsemblSequenceInput(ensembl_id="ENST00000357654"),
    EnsemblSequenceConfig(sequence_type="protein"),
)
print(f"{out.results[0].id}: {len(out.results[0].seq)} aa")
```

**Example 3: regulatory features overlapping BRCA1**

```python
from proto_tools.tools.database_retrieval import EnsemblOverlapInput, EnsemblOverlapConfig, run_ensembl_overlap

out = run_ensembl_overlap(
    EnsemblOverlapInput(ensembl_id="ENSG00000012048"),
    EnsemblOverlapConfig(overlap_feature="regulatory"),
)
for record in out.result[:5]:
    print(record.feature_type, record.raw.get("feature_name"))
```

**Example 4: variant effect prediction**

```python
from proto_tools.tools.database_retrieval import EnsemblVEPInput, run_ensembl_vep

out = run_ensembl_vep(EnsemblVEPInput(hgvs="9:g.22125504G>C"))
top = out.consequences[0]
print(f"{top.allele_string}: {top.most_severe_consequence}, {len(top.transcript_consequences)} transcripts affected")
```

**Example 5: chained workflow — Ensembl xrefs → UniProt fetch**

```python
from proto_tools.tools.database_retrieval import (
    EnsemblXrefsInput, run_ensembl_xrefs,
    UniProtFetchInput, run_uniprot_fetch,
)

xrefs = run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id="ENSG00000012048"))
uniprot_id = next(x.primary_id for x in xrefs.result if x.dbname == "Uniprot_gn")
uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id=uniprot_id))
print(f"BRCA1 UniProt: {uniprot.accession}, {uniprot.length} aa")
```

## Best Practices & Gotchas

**Common mistakes:**

1. **Forgetting `expand=True` on `ensembl-lookup`:** without it, the result has no nested `Transcript[]`. The default follows Ensembl REST (`False`), so request expansion explicitly when you need transcript/exon detail.
2. **Mixing assemblies:** GRCh38 and GRCh37 coordinates differ. Pin `config.assembly` per workflow; don't assume one default fits both.
3. **Treating `ensembl-overlap` records as fully typed:** `feature_type` differs per query (`gene` vs `regulatory` vs `variation`) and the typed fields are the intersection. Use `record.raw` for feature-specific keys.
4. **HGVS protein form for VEP:** `ENSP00000418960:p.Tyr124Cys` works, but the corresponding cDNA/genomic notation is often more reliable when the protein ID has multiple matching transcripts.

**Tips:**

- For gene-symbol → sequence flows, chain `ensembl-lookup` (by symbol) → `ensembl-sequence` on the canonical transcript ID. Pick `sequence_type="cds"` for the coding sequence, `"cdna"` for the spliced mRNA, or `"genomic"` for the intron-included pre-mRNA span.
- `overlap_feature="regulatory"` returns Ensembl Regulatory Build features (CTCF binding sites, open chromatin, etc.) — useful for cis-element design.

## References

**Primary publication:**

- Dyer, S. C., Austine-Orimoloye, O., Azov, A. G., Barba, M., Barnes, I., Barrera-Enriquez, V. P., et al. (2025). "Ensembl 2025." *Nucleic Acids Research*, 53(D1), D948–D957. [DOI: 10.1093/nar/gkae1071](https://doi.org/10.1093/nar/gkae1071)

**Implementation:**

- Ensembl REST source: [https://github.com/Ensembl/ensembl-rest](https://github.com/Ensembl/ensembl-rest)
- Ensembl REST live docs: [https://rest.ensembl.org/](https://rest.ensembl.org/)
- Ensembl genome browser: [https://www.ensembl.org/](https://www.ensembl.org/)

## Related Tools

- **`uniprot-fetch`**: Resolve `ensembl-xrefs` rows with `dbname="Uniprot_gn"` to a UniProt accession; chain into `interproscan-fetch` / `alphamissense-fetch` for protein-level annotations.
- **`ncbi-efetch`**: Pull GenBank / RefSeq records when you need NCBI-side identifiers; `ensembl-xrefs` also exposes `EntrezGene` mappings.
- **`alphafold-db-fetch`**: Once an Ensembl protein is mapped to UniProt, fetch its predicted structure for downstream design.
