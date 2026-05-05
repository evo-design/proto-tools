"""Smoke test the 5 ensembl-* tools via their public registry API.

Hits the live Ensembl REST API once per tool, asserts the typed result is
populated, and prints a one-line summary. Exits non-zero on any failure.

Run:
    python scripts/smoke_ensembl.py
"""

from __future__ import annotations

import sys

from proto_tools.tools.database_retrieval import (
    EnsemblLookupConfig,
    EnsemblLookupInput,
    EnsemblOverlapConfig,
    EnsemblOverlapInput,
    EnsemblSequenceConfig,
    EnsemblSequenceInput,
    EnsemblVEPInput,
    EnsemblXrefsInput,
    run_ensembl_lookup,
    run_ensembl_overlap,
    run_ensembl_sequence,
    run_ensembl_vep,
    run_ensembl_xrefs,
)

BRCA1_GENE = "ENSG00000012048"
BRCA1_CANONICAL_TRANSCRIPT = "ENST00000357654"


def main() -> int:
    """Run each ensembl-* tool once on a BRCA1 anchor and assert typed results."""
    failures: list[str] = []

    lookup = run_ensembl_lookup(EnsemblLookupInput(symbol="BRCA1"), EnsemblLookupConfig(expand=True))
    if not lookup.success or lookup.result.id != BRCA1_GENE:
        failures.append(f"lookup: success={lookup.success} id={lookup.result.id if lookup.success else 'n/a'}")
    print(f"[lookup]   BRCA1 → {lookup.result.id} ({len(lookup.result.Transcript)} transcripts)")

    seq = run_ensembl_sequence(
        EnsemblSequenceInput(ensembl_id=BRCA1_CANONICAL_TRANSCRIPT),
        EnsemblSequenceConfig(sequence_type="protein"),
    )
    if not seq.success or len(seq.result.seq) != 1863:
        failures.append(f"sequence: success={seq.success} len={len(seq.result.seq) if seq.success else 'n/a'}")
    print(f"[sequence] {BRCA1_CANONICAL_TRANSCRIPT} protein → {len(seq.result.seq)} aa")

    overlap = run_ensembl_overlap(
        EnsemblOverlapInput(ensembl_id=BRCA1_GENE), EnsemblOverlapConfig(overlap_feature="exon")
    )
    if not overlap.success or len(overlap.result) < 10:
        failures.append(f"overlap: success={overlap.success} n={len(overlap.result) if overlap.success else 'n/a'}")
    print(f"[overlap]  {BRCA1_GENE} exons → {len(overlap.result)} records")

    xrefs = run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id=BRCA1_GENE))
    uniprot_ids = [x.primary_id for x in xrefs.result if x.dbname == "Uniprot_gn"] if xrefs.success else []
    if not xrefs.success or not uniprot_ids:
        failures.append(f"xrefs: success={xrefs.success} uniprot_gn={uniprot_ids}")
    print(f"[xrefs]    {BRCA1_GENE} → {len(xrefs.result)} xrefs (UniProt: {uniprot_ids[:1]})")

    vep = run_ensembl_vep(EnsemblVEPInput(hgvs=f"{BRCA1_CANONICAL_TRANSCRIPT}:c.181T>G"))
    severe = vep.consequences[0].most_severe_consequence if vep.success and vep.consequences else None
    if not vep.success or severe != "missense_variant":
        failures.append(f"vep: success={vep.success} most_severe={severe}")
    print(f"[vep]      BRCA1 c.181T>G → {severe}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll 5 tools OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
