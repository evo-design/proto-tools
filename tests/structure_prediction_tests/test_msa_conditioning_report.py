"""Regression tests for mixed per-chain MSA reporting and retry interpretation."""

import warnings

import pytest

from proto_tools.entities.msa import MSA
from proto_tools.entities.structures import MSAConditioning, MSASubsampling, Structure
from proto_tools.tools.structure_prediction.boltz2.helpers import build_chain_msa_paths
from proto_tools.tools.structure_prediction.msa_reporting import build_msa_conditioning
from proto_tools.tools.structure_prediction.shared_data_models import (
    Chain,
    Complex,
    ComplexMSAs,
    normalize_output_chain_ids,
)

_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C  \nEND\n"


def test_report_summaries_are_derived_and_round_trip_through_json():
    """Coverage and pairing_mode follow the chains and survive a serialize/validate cycle."""
    chains = [Chain(sequence="AAA", entity_type="protein"), Chain(sequence="GGG", entity_type="protein")]
    msas = ComplexMSAs(per_chain={0: MSA(aligned_sequences=["AAA", "AAG"])}, source="colabfold_server")
    report = build_msa_conditioning(chains, msas, subsampling=MSASubsampling(mode="stochastic", max_rows=4))
    assert (report.coverage, report.pairing_mode) == ("partial", "unpaired")

    restored = MSAConditioning.model_validate_json(report.model_dump_json())
    assert restored == report
    assert (restored.coverage, restored.pairing_mode) == ("partial", "unpaired")


def test_requested_effective_statuses_are_distinct():
    chains = [Chain(sequence="AAA", entity_type="protein", id=chain_id) for chain_id in "ABCD"]
    msas = ComplexMSAs(
        per_chain={
            2: MSA(aligned_sequences=["AAA", "AAG"]),
            3: MSA(aligned_sequences=["AAA", "AAT"]),
        },
        source="colabfold_server",
        requested_chain_indices=[1, 2, 3],
    )
    report = build_msa_conditioning(chains, msas, consumed={1, 2})
    assert [chain.status for chain in report.chains] == [
        "not_requested",
        "no_hits",
        "consumed",
        "ignored",
    ]
    assert [chain.requested_source for chain in report.chains] == [
        "none",
        "colabfold_server",
        "colabfold_server",
        "colabfold_server",
    ]


def test_partial_supplied_msa_reports_requested_chain_only():
    chains = [Chain(sequence="AAA", entity_type="protein"), Chain(sequence="GGG", entity_type="protein")]
    msas = ComplexMSAs(per_chain={1: MSA(aligned_sequences=["GGG", "GGA"])})
    report = build_msa_conditioning(chains, msas)
    assert report.coverage == "partial"
    assert [(chain.chain_id, chain.status) for chain in report.chains] == [
        ("A", "not_requested"),
        ("B", "consumed"),
    ]


def test_paired_rows_and_subsampling_are_separate():
    chain = Chain(sequence="AAA", entity_type="protein")
    msas = ComplexMSAs(
        per_chain={0: MSA(aligned_sequences=["AAA", "AAG"])},
        paired=True,
        unpaired_per_chain={0: MSA(aligned_sequences=["AAA", "AAG", "AAT"])},
    )
    report = build_msa_conditioning(
        [chain],
        msas,
        subsampling=MSASubsampling(mode="stochastic", max_rows=4, max_extra_rows=8),
    )
    assert report.pairing_mode == "paired"
    assert report.subsampling == MSASubsampling(mode="stochastic", max_rows=4, max_extra_rows=8)
    assert report.chains[0].model_dump() == {
        "chain_index": 0,
        "chain_id": "A",
        "status": "consumed",
        "rows": 2,
        "unpaired_rows": 3,
        "requested_source": "supplied",
        "mode": "paired",
        "ignored_reason": None,
    }


def test_nonprotein_complex_has_not_applicable_msa_conditioning():
    report = build_msa_conditioning([Chain(sequence="ACGT", entity_type="dna")], None)
    assert report.coverage == "not_applicable"
    assert report.chains == []
    assert report.subsampling.mode == "none"


def test_partial_target_msa_survives_serialization_without_false_failure(tmp_path):
    chains = [Chain(sequence="AAA", entity_type="protein"), Chain(sequence="GGG", entity_type="protein")]
    msas = ComplexMSAs(per_chain={1: MSA(aligned_sequences=["GGG", "GGA", "GGT"])})
    structure = Structure(structure=_PDB, structure_format="pdb")
    with warnings.catch_warnings(record=True) as caught:
        paths = build_chain_msa_paths(Complex(chains=chains), msas, str(tmp_path))
    assert list(paths) == ["B"]
    assert not caught
    result = normalize_output_chain_ids(structure, chains, complex_msas=msas, report_msa=True)
    restored = Structure.model_validate_json(result.model_dump_json())
    report = restored.msa_conditioning
    assert report.coverage == "partial"
    assert [(c.chain_id, c.rows, c.status) for c in report.chains] == [
        ("A", 0, "not_requested"),
        ("B", 3, "consumed"),
    ]


@pytest.mark.parametrize("rows,expected", [(0, "not_requested"), (2, "consumed")])
def test_reports_actual_rows_not_requested_settings(rows, expected):
    chain = Chain(sequence="AAA", entity_type="protein", id="T")
    cm = ComplexMSAs(per_chain={0: MSA(aligned_sequences=["AAA"] * rows)}) if rows else None
    s = normalize_output_chain_ids(
        Structure(structure=_PDB, structure_format="pdb"), [chain], complex_msas=cm, report_msa=True
    )
    assert s.msa_conditioning.chains[0].status == expected
    assert s.msa_conditioning.chains[0].chain_id == "T"
    assert s.get_chain_ids() == ["T"]
    assert s.msa_conditioning.coverage == ("full" if rows > 1 else "none")


def test_normalized_output_preserves_generated_no_hit_status():
    chain = Chain(sequence="AAA", entity_type="protein", id="T")
    complex_msas = ComplexMSAs(per_chain={}, source="colabfold_server", requested_chain_indices=[0])

    result = normalize_output_chain_ids(
        Structure(structure=_PDB, structure_format="pdb"),
        [chain],
        complex_msas=complex_msas,
        report_msa=True,
    )

    assert result.msa_conditioning.chains[0].status == "no_hits"
    assert result.msa_conditioning.chains[0].requested_source == "colabfold_server"


def test_paired_and_unpaired_depth_are_not_added_or_lost():
    msa = MSA(aligned_sequences=["AAA", "AAG"])
    depth = MSA(aligned_sequences=["AAA", "AAG", "AAT"])
    cm = ComplexMSAs(per_chain={0: msa}, paired=True, unpaired_per_chain={0: depth})
    s = normalize_output_chain_ids(
        Structure(structure=_PDB, structure_format="pdb"),
        [Chain(sequence="AAA", entity_type="protein")],
        complex_msas=cm,
        report_msa=True,
    )
    chain = s.msa_conditioning.chains[0]
    assert (chain.rows, chain.unpaired_rows, chain.mode) == (2, 3, "paired")
    assert chain.status == "consumed"
    assert s.msa_conditioning.coverage == "full"


def _msa(*rows, ids=None):
    return MSA(aligned_sequences=list(rows), sequence_ids=ids or [f"s{i}" for i in range(len(rows))])


def test_alphafold2_homo_oligomer_reports_only_tiled_chain_zero_rows_as_consumed():
    """ColabDesign tiles chain 0's alignment, so a copy with different rows of its own is ignored."""
    from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import _alphafold2_msa_consumption

    shared, other = _msa("AAA", "AAG"), _msa("AAA", "AAT")
    msas = ComplexMSAs(per_chain={0: shared, 1: shared, 2: other})
    consumed, reasons = _alphafold2_msa_consumption(msas, 3, None, None)
    assert consumed == {0, 1}
    assert set(reasons) == {2}

    consumed, reasons = _alphafold2_msa_consumption(ComplexMSAs(per_chain={1: other}), 2, None, None)
    assert consumed == set()
    assert set(reasons) == {1}


def test_alphafold2_heteromer_uses_unpaired_rows_only_beneath_paired_alignments():
    """The standalone stacks unpaired depth under paired alignments; unpaired heteromers drop it."""
    from proto_tools.tools.structure_prediction.alphafold2.alphafold2 import _alphafold2_msa_consumption

    deep = {0: _msa("AAA", "AAT"), 1: _msa("CCC", "CCG")}

    paired = ComplexMSAs(per_chain={0: _msa("AAA", "AAG"), 1: _msa("CCC", "CCA")}, paired=True, unpaired_per_chain=deep)
    consumed, reasons = _alphafold2_msa_consumption(paired, 2, ["a3m", "a3m"], ["a3m", "a3m"])
    assert (consumed, reasons) == ({0, 1}, {})

    # Unpaired: chain 1 has only unpaired rows, which the standalone never appends.
    unpaired = ComplexMSAs(per_chain={0: _msa("AAA", "AAG")}, unpaired_per_chain=deep)
    consumed, reasons = _alphafold2_msa_consumption(unpaired, 2, ["a3m", None], ["a3m", "a3m"])
    assert consumed == {0}
    assert set(reasons) == {1}


def test_consumption_requires_a_primary_alignment_and_skips_short_chains():
    """Chains with only unpaired rows are never written; short chains are dropped when asked."""
    from proto_tools.tools.structure_prediction.msa_reporting import msa_consumption

    chains = [Chain(sequence=seq, entity_type="protein") for seq in ("MKTAY", "MKT", "MKTAQ")]
    msas = ComplexMSAs(
        per_chain={0: _msa("MKTAY", "MKTAA"), 1: _msa("MKT", "MKA")},
        unpaired_per_chain={2: _msa("MKTAQ", "MKTAA")},
    )
    consumed, reasons = msa_consumption(chains, msas, backend="Boltz-2")
    assert consumed == {0, 1}
    assert set(reasons) == {2}

    consumed, reasons = msa_consumption(chains, msas, backend="Protenix", max_skipped_length=4)
    assert consumed == {0}
    assert reasons[1] == "Protenix skips the MSA of chains with 4 or fewer residues"


def test_summaries_count_only_consumed_chains():
    """An ignored alignment does not make the structure MSA-conditioned."""
    chains = [Chain(sequence="AAA", entity_type="protein")]
    msas = ComplexMSAs(per_chain={0: MSA(aligned_sequences=["AAA", "AAG"])})
    report = build_msa_conditioning(chains, msas, consumed=False)
    assert (report.chains[0].status, report.coverage, report.pairing_mode) == ("ignored", "none", "none")


def test_structure_transforms_keep_the_conditioning_report():
    structure = normalize_output_chain_ids(
        Structure(structure=_PDB, structure_format="pdb"),
        [Chain(sequence="AAA", entity_type="protein")],
        complex_msas=ComplexMSAs(per_chain={0: MSA(aligned_sequences=["AAA", "AAG"])}),
        report_msa=True,
    )
    for derived in (
        structure.select_chains("A"),
        structure.with_renamed_chains({"A": "B"}),
        Structure(structure=structure),
    ):
        assert derived.msa_conditioning == structure.msa_conditioning


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

    report = boltz2_msa_subsampling(max_msa_seqs=max_msa_seqs, subsample_msa=subsample_msa, has_msa=True)
    assert (report.mode, report.max_rows) == (mode, max_rows)


def test_rf3_reports_written_alignment_and_pairs_only_on_taxonomy_headers():
    """RF3 gets the deep unpaired a3m and pairs only distinct chains whose headers carry TaxID=."""
    from proto_tools.tools.structure_prediction.rf3.helpers import rf3_written_msas

    heteromer = Complex(
        chains=[Chain(sequence="AAA", entity_type="protein"), Chain(sequence="CCC", entity_type="protein")]
    )
    paired = {0: _msa("AAA", "AAG"), 1: _msa("CCC", "CCG")}
    deep = {0: _msa("AAA", "AAG", "AAT"), 1: _msa("CCC", "CCG", "CCT")}
    written, is_paired = rf3_written_msas(
        heteromer, ComplexMSAs(per_chain=paired, paired=True, unpaired_per_chain=deep)
    )
    assert not is_paired
    assert {i: len(m) for i, m in written.per_chain.items()} == {0: 3, 1: 3}

    # Tagged alignments of different depths are paired by RF3, and must not fail validation.
    tagged = {
        0: _msa("AAA", "AAG", "AAT", ids=["q", "UniRef100_X TaxID=9606", "UniRef100_Y TaxID=10090"]),
        1: _msa("CCC", "CCG", ids=["q", "UniRef100_Z TaxID=9606"]),
    }
    _, is_paired = rf3_written_msas(heteromer, ComplexMSAs(per_chain=tagged))
    assert is_paired

    homodimer = Complex(
        chains=[Chain(sequence="AAA", entity_type="protein"), Chain(sequence="AAA", entity_type="protein")]
    )
    _, is_paired = rf3_written_msas(homodimer, ComplexMSAs(per_chain={0: tagged[0], 1: tagged[0]}))
    assert not is_paired


def test_msa_summary_is_logged_at_debug(caplog):
    """The per-structure MSA line stays out of default (INFO) output."""
    import logging

    structure = Structure(structure=_PDB, structure_format="pdb", source="tool-prediction")
    chains = [Chain(sequence="AAA", entity_type="protein")]
    msas = ComplexMSAs(per_chain={0: MSA(aligned_sequences=["AAA", "AAG"])}, source="colabfold_server")
    with caplog.at_level(logging.DEBUG, logger="proto_tools"):
        normalize_output_chain_ids(structure, chains, complex_msas=msas, report_msa=True)
    [record] = [r for r in caplog.records if "MSA sent" in r.getMessage()]
    assert record.levelno == logging.DEBUG
    assert record.getMessage() == "tool-prediction MSA sent: A=2 rows (colabfold_server, unpaired); selection unknown"
