"""Tests for the MSA conditioning report and the per-chain records writers build it from."""

import pytest

from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import MSAConditioning, Structure
from proto_tools.tools.structure_prediction.msa_plan import (
    SentMSA,
    conditions_on_msa,
    protein_chain_msas,
    short_chain_reason,
    unpaired_depth_rows,
)
from proto_tools.tools.structure_prediction.msa_reporting import build_msa_conditioning
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    normalize_output_chain_ids,
)

_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C  \nEND\n"


def _msa(*rows, ids=None):
    return MSA(aligned_sequences=list(rows), sequence_ids=ids or [f"s{i}" for i in range(len(rows))])


# ── Report from sent records ─────────────────────────────────────────────────


def test_statuses_follow_what_was_sent():
    """Sent rows mean consumed; otherwise not requested, no hits (our search), or ignored."""
    chains = [Chain(sequence="AAA", entity_type="protein", id=chain_id) for chain_id in "ABCDE"]
    msas = ComplexMSAs(
        per_chain={2: _msa("AAA", "AAG"), 3: _msa("AAA", "AAT"), 4: _msa("AAA", "AAC")},
        source="colabfold_server",
        requested_chain_indices=[1, 2, 3, 4],
    )
    sent = {
        1: SentMSA(rows=1),  # query-only file after a no-hit search
        2: SentMSA(rows=2),
        3: SentMSA(ignored_reason="backend dropped it"),
        4: SentMSA(rows=2, ignored_reason="backend skips short chains"),
    }
    report = build_msa_conditioning(chains, msas, sent)
    assert [chain.status for chain in report.chains] == ["not_requested", "no_hits", "consumed", "ignored", "ignored"]
    assert [chain.requested_source for chain in report.chains] == ["none"] + ["colabfold_server"] * 4
    assert [chain.ignored_reason for chain in report.chains][3:] == ["backend dropped it", "backend skips short chains"]


def test_report_counts_sent_rows_and_round_trips_through_json():
    """Rows are what the backend received; summaries are derived and survive serialization."""
    chains = [Chain(sequence="AAA", entity_type="protein"), Chain(sequence="GGG", entity_type="protein")]
    msas = ComplexMSAs(per_chain={0: _msa("AAA", "AAG", "AAT")})
    report = build_msa_conditioning(
        chains, msas, {0: SentMSA(rows=3, unpaired_rows=2, paired=True), 1: SentMSA(rows=1)}
    )
    assert [(c.rows, c.unpaired_rows, c.mode) for c in report.chains] == [(3, 2, "paired"), (1, 0, "none")]
    assert (report.coverage, report.pairing_mode) == ("partial", "paired")
    assert MSAConditioning.model_validate_json(report.model_dump_json()) == report


def test_ignored_alignment_does_not_condition_the_structure():
    chains = [Chain(sequence="AAA", entity_type="protein")]
    report = build_msa_conditioning(chains, ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}), {})
    assert (report.chains[0].status, report.chains[0].rows, report.coverage) == ("ignored", 0, "none")
    assert report.chains[0].ignored_reason == "no alignment rows reached the backend"


def test_unused_homolog_rows_warn_once_without_failing(caplog, monkeypatch):
    """A sweep that hands every predictor the same MSAs is warned once per reason, not stopped."""
    import logging

    from proto_tools.tools.structure_prediction import msa_reporting

    monkeypatch.setattr(msa_reporting, "_WARNED_UNUSED", set())
    chains = [Chain(sequence="AAA", entity_type="protein", id=chain_id) for chain_id in "ABC"]
    msas = ComplexMSAs(per_chain={0: _msa("AAA", "AAG"), 1: _msa("AAA", "AAT")}, requested_chain_indices=[0, 1, 2])
    sent = {0: SentMSA(ignored_reason="single-sequence model"), 1: SentMSA(rows=2)}
    with caplog.at_level(logging.DEBUG, logger="proto_tools"):
        report = build_msa_conditioning(chains, msas, sent, tool="tool-prediction")
        build_msa_conditioning(chains, msas, sent, tool="tool-prediction")
    assert [chain.status for chain in report.chains] == ["ignored", "consumed", "ignored"]
    # Chain C was requested but carried no rows, so nothing went unused; the repeat drops to DEBUG.
    records = [r for r in caplog.records if "were not used" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG]
    assert records[0].getMessage() == "tool-prediction: MSA rows for chain(s) A were not used: single-sequence model"


def test_nonprotein_complex_has_not_applicable_msa_conditioning():
    report = build_msa_conditioning([Chain(sequence="ACGT", entity_type="dna")], None, {})
    assert (report.coverage, report.chains, report.subsampling.mode) == ("not_applicable", [], "none")


def test_normalized_output_attaches_report_and_logs_it_at_debug(caplog):
    import logging

    structure = Structure(structure=_PDB, structure_format="pdb", source="tool-prediction")
    chains = [Chain(sequence="AAA", entity_type="protein", id="T")]
    msas = ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}, source="colabfold_server")
    with caplog.at_level(logging.DEBUG, logger="proto_tools"):
        result = normalize_output_chain_ids(
            structure, chains, complex_msas=msas, report_msa=True, msa_sent={0: SentMSA(rows=2)}
        )
    restored = Structure.model_validate_json(result.model_dump_json())
    assert [(c.chain_id, c.status, c.rows) for c in restored.msa_conditioning.chains] == [("T", "consumed", 2)]
    [record] = [r for r in caplog.records if "MSA sent" in r.getMessage()]
    assert record.levelno == logging.DEBUG
    assert record.getMessage() == "tool-prediction MSA sent: T=2 rows (colabfold_server, unpaired); selection unknown"


# ── Shared writer helpers ────────────────────────────────────────────────────


def test_protein_chain_msas_skips_non_protein_chains():
    chains = [
        Chain(sequence="AAA", entity_type="protein"),
        Chain(sequence="ACGU", entity_type="rna"),
        Chain(sequence="GGG", entity_type="protein"),
    ]
    msas = ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}, unpaired_per_chain={2: _msa("GGG", "GGA")})
    found = [
        (c.index, c.chain_id, c.primary is not None, c.unpaired is not None) for c in protein_chain_msas(chains, msas)
    ]
    assert found == [(0, "A", True, False), (2, "C", False, True)]


def test_unpaired_depth_rows_skip_the_query_and_duplicates():
    assert unpaired_depth_rows(["AAA", "AAG"], _msa("AAA", "AAG", "AAT", "AAT")) == [("AAT", "s2")]
    assert unpaired_depth_rows(["AAA"], None) == []


def test_short_chain_reason_and_conditioning_check():
    assert (
        short_chain_reason(Chain(sequence="MKTL", entity_type="protein"), "AF3")
        == "AF3 skips the MSA of chains with 4 or fewer residues"
    )
    assert short_chain_reason(Chain(sequence="MKTLA", entity_type="protein"), "AF3") is None
    assert not conditions_on_msa({0: SentMSA(rows=1), 1: SentMSA(rows=5, ignored_reason="skipped")})
    assert conditions_on_msa({0: SentMSA(rows=1, unpaired_rows=3)})


# ── Per-writer records ───────────────────────────────────────────────────────


def test_alphafold2_tiles_chain_zero_across_homo_oligomer_copies():
    from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import _alphafold2_sent_msas

    assert _alphafold2_sent_msas(ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}), 2, None, None) == {
        0: SentMSA(rows=2),
        1: SentMSA(rows=2),
    }
    sent = _alphafold2_sent_msas(ComplexMSAs(per_chain={1: _msa("AAA", "AAT")}), 2, None, None)
    assert set(sent) == {1} and sent[1].ignored_reason is not None


def test_alphafold2_heteromer_stacks_unpaired_rows_only_beneath_paired_alignments():
    from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import _alphafold2_sent_msas

    deep = {0: _msa("AAA", "AAT", "AAC"), 1: _msa("CCC", "CCG")}
    paired = ComplexMSAs(per_chain={0: _msa("AAA", "AAG"), 1: _msa("CCC", "CCA")}, paired=True, unpaired_per_chain=deep)
    sent = _alphafold2_sent_msas(paired, 2, ["a3m", "a3m"], ["a3m", "a3m"])
    # Appended unpaired depth excludes each chain's query row.
    assert sent == {0: SentMSA(rows=2, unpaired_rows=2, paired=True), 1: SentMSA(rows=2, unpaired_rows=1, paired=True)}

    # A paired bundle whose chain B has only unpaired rows: B gets depth but no paired rows.
    partial = ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}, paired=True, unpaired_per_chain=deep)
    sent = _alphafold2_sent_msas(partial, 2, ["a3m", None], ["a3m", "a3m"])
    assert sent[1] == SentMSA(rows=0, unpaired_rows=1, paired=False)
    chains = [Chain(sequence="AAA", entity_type="protein"), Chain(sequence="CCC", entity_type="protein")]
    report = build_msa_conditioning(chains, partial, sent)
    assert ([c.mode for c in report.chains], report.pairing_mode) == (["paired", "unpaired"], "mixed")

    unpaired = ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}, unpaired_per_chain=deep)
    sent = _alphafold2_sent_msas(unpaired, 2, ["a3m", None], ["a3m", "a3m"])
    assert sent[0] == SentMSA(rows=2) and sent[1].ignored_reason is not None


def test_rf3_pairs_only_distinct_chains_with_taxonomy_headers(tmp_path):
    """RF3 gets the deep unpaired a3m and pairs only distinct chains whose headers carry TaxID=."""
    from proto_tools.tools.structure_prediction.rf3.helpers import build_chain_a3m_paths

    def chains(*seqs):
        return Complex(chains=[Chain(sequence=seq, entity_type="protein") for seq in seqs])

    deep = {0: _msa("AAA", "AAG", "AAT"), 1: _msa("CCC", "CCG", "CCT")}
    paired = {0: _msa("AAA", "AAG"), 1: _msa("CCC", "CCG")}
    _, sent = build_chain_a3m_paths(
        chains("AAA", "CCC"), ComplexMSAs(per_chain=paired, paired=True, unpaired_per_chain=deep), str(tmp_path / "a")
    )
    assert sent == {0: SentMSA(rows=3), 1: SentMSA(rows=3)}

    tagged = {
        0: _msa("AAA", "AAG", "AAT", ids=["q", "UniRef100_X TaxID=9606", "UniRef100_Y TaxID=10090"]),
        1: _msa("CCC", "CCG", ids=["q", "UniRef100_Z TaxID=9606"]),
    }
    _, sent = build_chain_a3m_paths(chains("AAA", "CCC"), ComplexMSAs(per_chain=tagged), str(tmp_path / "b"))
    assert all(record.paired for record in sent.values())
    _, sent = build_chain_a3m_paths(
        chains("AAA", "AAA"), ComplexMSAs(per_chain={0: tagged[0], 1: tagged[0]}), str(tmp_path / "c")
    )
    assert not any(record.paired for record in sent.values())


def test_boltz2_records_appended_depth_and_identical_chains_share_a_file(tmp_path):
    from proto_tools.tools.structure_prediction.boltz2.helpers import build_chain_msa_paths

    cx = Complex(chains=[Chain(sequence=seq, entity_type="protein") for seq in ("AAA", "AAA", "CCC")])
    msas = ComplexMSAs(
        per_chain={0: _msa("AAA", "AAG"), 1: _msa("AAA", "AAW")},
        unpaired_per_chain={0: _msa("AAA", "AAG", "AAT")},
        paired=True,
    )
    paths, sent = build_chain_msa_paths(cx, msas, str(tmp_path))
    assert paths["A"] == paths["B"]
    assert sent == {0: SentMSA(rows=2, unpaired_rows=1, paired=True), 1: SentMSA(rows=2, unpaired_rows=1, paired=True)}


def test_structure_transforms_carry_the_conditioning_report_for_their_chains():
    """Renames relabel the report's chains and selections drop the chains they remove."""
    from tests._structure_fixtures import synthetic_cif

    structure = normalize_output_chain_ids(
        Structure(structure=synthetic_cif(["A", "B"]), structure_format="cif"),
        [Chain(sequence="AAA", entity_type="protein"), Chain(sequence="GGG", entity_type="protein")],
        complex_msas=ComplexMSAs(per_chain={0: _msa("AAA", "AAG"), 1: _msa("GGG", "GGA")}),
        report_msa=True,
        msa_sent={0: SentMSA(rows=2), 1: SentMSA(rows=2)},
    )
    assert Structure(structure=structure).msa_conditioning == structure.msa_conditioning
    renamed = structure.with_renamed_chains({"A": "X"}).msa_conditioning
    assert [chain.chain_id for chain in renamed.chains] == ["X", "B"]
    selected = structure.select_chains("B").msa_conditioning
    assert [(chain.chain_index, chain.chain_id) for chain in selected.chains] == [(1, "B")]
    assert selected.coverage == "full"


@pytest.mark.parametrize(
    ("max_msa_seqs", "subsample_msa", "mode", "max_rows"),
    [
        (8192, True, "stochastic", 1024),
        (64, True, "stochastic", 64),
        (8192, False, "deterministic", 8192),
        (20000, False, "deterministic", 16384),
    ],
)
def test_boltz2_subsampling_reports_boltz_row_caps(max_msa_seqs, subsample_msa, mode, max_rows):
    """The random draw is boltz's num_subsampled_msa, never more than the featurized depth."""
    from proto_tools.tools.structure_prediction.boltz2.helpers import boltz2_msa_subsampling

    report = boltz2_msa_subsampling(max_msa_seqs=max_msa_seqs, subsample_msa=subsample_msa)
    assert (report.mode, report.max_rows) == (mode, max_rows)


def test_chai1_keeps_the_unpaired_only_reason_without_any_primary_alignment(monkeypatch):
    from proto_tools.tools.structure_prediction.chai1.chai1 import Chai1Config, run_chai1_on_complex
    from tests._structure_fixtures import synthetic_cif

    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.chai1.chai1.ToolInstance.dispatch",
        lambda *a, **k: {
            "cif_output": synthetic_cif(["A"]),
            "metrics": {
                "avg_plddt": 0.8,
                "ptm": 0.7,
                "iptm": 0.0,
                "avg_pae": 1.0,
                "pae": None,
                "confidence_score": 0.7,
            },
        },
    )
    msas = ComplexMSAs(per_chain={}, unpaired_per_chain={0: _msa("MKTAYIAKQR", "MKTAYIAKQA")})
    structure = run_chai1_on_complex(
        Complex(chains=[Chain(sequence="MKTAYIAKQR", entity_type="protein")]), Chai1Config(), complex_msas=msas
    )
    assert structure.msa_conditioning.chains[0].ignored_reason == (
        "only unpaired rows were supplied; Chai-1 is sent a chain's primary alignment"
    )
