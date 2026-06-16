"""tests/database_retrieval_tests/test_ensembl_workflow.py.

Cross-tool stress / real-user workflow integration tests for the ensembl-*
toolkit. Each test emulates a realistic chained query (BRCA1, TP53, EGFR
panel; non-coding RNA; non-human species; concurrent calls) and verifies
the wrappers behave correctly under live conditions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from proto_tools.tools.database_retrieval import (
    EnsemblLookupConfig,
    EnsemblLookupInput,
    EnsemblOverlapConfig,
    EnsemblOverlapInput,
    EnsemblSequenceConfig,
    EnsemblSequenceInput,
    EnsemblVEPInput,
    EnsemblXrefsInput,
    UniProtFetchInput,
    run_ensembl_lookup,
    run_ensembl_overlap,
    run_ensembl_sequence,
    run_ensembl_vep,
    run_ensembl_xrefs,
    run_uniprot_fetch,
)


@pytest.mark.integration
def test_panel_full_chain_brca1():
    """Full chain on a single gene (BRCA1).

    lookup_symbol → ensembl_id → expand transcripts → take canonical →
    fetch protein sequence → list overlapping exons → resolve UniProt
    via xrefs → cross-confirm gene_name on UniProt side.
    """
    lookup = run_ensembl_lookup(EnsemblLookupInput(symbol="BRCA1"), EnsemblLookupConfig(expand=True))
    assert lookup.success, lookup.errors
    assert lookup.result.id == "ENSG00000012048"
    canonical = lookup.result.canonical_transcript
    assert canonical is not None
    transcript_id = canonical.split(".")[0]

    seq = run_ensembl_sequence(
        EnsemblSequenceInput(ensembl_id=transcript_id),
        EnsemblSequenceConfig(sequence_type="protein"),
    )
    assert seq.success, seq.errors
    assert len(seq.results[0].seq) == 1863  # BRCA1 canonical isoform

    exons = run_ensembl_overlap(
        EnsemblOverlapInput(ensembl_id=lookup.result.id),
        EnsemblOverlapConfig(overlap_feature="exon"),
    )
    assert exons.success, exons.errors
    assert len(exons.result) >= 10

    xrefs = run_ensembl_xrefs(EnsemblXrefsInput(ensembl_id=lookup.result.id))
    assert xrefs.success, xrefs.errors
    # Resolve to the canonical reviewed entry; TrEMBL accessions carry no gene names.
    uniprot_ids = {x.primary_id for x in xrefs.result if x.dbname == "Uniprot_gn"}
    assert "P38398" in uniprot_ids
    uniprot = run_uniprot_fetch(UniProtFetchInput(uniprot_id="P38398"))
    assert uniprot.success
    assert "brca1" in {n.lower() for n in uniprot.gene_names}


@pytest.mark.integration
@pytest.mark.parametrize(
    "symbol, expected_ensembl_id_prefix",
    [
        ("BRCA1", "ENSG00000012048"),
        ("TP53", "ENSG00000141510"),
        ("EGFR", "ENSG00000146648"),
    ],
    ids=["BRCA1", "TP53", "EGFR"],
)
def test_lookup_panel_well_known_oncogenes(symbol: str, expected_ensembl_id_prefix: str):
    """Lookup the three most-studied human oncogenes; each resolves to its known stable ID."""
    out = run_ensembl_lookup(EnsemblLookupInput(symbol=symbol))
    assert out.success, out.errors
    assert out.result.id == expected_ensembl_id_prefix
    assert out.result.display_name == symbol


@pytest.mark.integration
def test_non_coding_rna_lookup_returns_no_translation():
    """MALAT1 is a long non-coding RNA: lookup with expand should return transcripts but no Translation block."""
    out = run_ensembl_lookup(EnsemblLookupInput(symbol="MALAT1"), EnsemblLookupConfig(expand=True))
    assert out.success, out.errors
    assert out.result.biotype == "lncRNA"
    assert len(out.result.Transcript) >= 1
    assert all(t.Translation is None for t in out.result.Transcript)


@pytest.mark.integration
def test_non_human_species_mouse_lookup():
    """Mouse Trp53 lookup with species=mus_musculus returns a mouse ENSMUSG ID."""
    out = run_ensembl_lookup(
        EnsemblLookupInput(symbol="Trp53"),
        EnsemblLookupConfig(species="mus_musculus"),
    )
    assert out.success, out.errors
    assert out.result.id.startswith("ENSMUSG")
    assert out.result.species == "mus_musculus"


@pytest.mark.integration
def test_vep_clinvar_pathogenic_variant():
    """A well-studied BRCA1 missense (c.181T>G, p.Cys61Gly) is reported as missense_variant."""
    vep = run_ensembl_vep(EnsemblVEPInput(hgvs="ENST00000357654:c.181T>G"))
    assert vep.success, vep.errors
    assert vep.num_consequences == 1
    cons = vep.consequences[0]
    assert cons.most_severe_consequence == "missense_variant"
    assert any(
        tc.get("transcript_id") == "ENST00000357654" and "missense_variant" in tc.get("consequence_terms", [])
        for tc in cons.transcript_consequences
    )


@pytest.mark.integration
def test_concurrent_lookup_panel_dispatch():
    """8 concurrent lookups across distinct symbols all return correct gene records.

    Exercises per-thread session lifecycle and result independence; under
    typical load this won't trip Ensembl's rate limiter, so retry behavior
    is not asserted here.
    """
    symbols = ["BRCA1", "TP53", "EGFR", "MYC", "PTEN", "KRAS", "ALK", "BRAF"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(lambda s: run_ensembl_lookup(EnsemblLookupInput(symbol=s)), symbols))
    assert all(o.result.species == "homo_sapiens" for o in outs)
    assert {o.result.display_name for o in outs} == set(symbols)
