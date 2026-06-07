"""tests/gene_annotation_tests/test_miranda_scan.py.

Tests for the miRanda microRNA target-site prediction tool.
"""

import csv
import json

import pytest

from proto_tools.tools import (
    MirandaConfig,
    MirandaInput,
    MirandaOutput,
    MirandaSequenceResult,
    MirandaTargetSite,
    run_miranda_scan,
)
from proto_tools.tools.gene_annotation.miranda.miranda_scan import (
    _EXAMPLES_DIR,
    _parse_miranda_output,
    _read_fasta,
)

# One captured hit: the 3-line alignment block, the tab-delimited self-identifying
# `>` line, and the `>>` summary line (which must be ignored).
SAMPLE_STDOUT = (
    "   Forward:\tScore: 165.000000  Q:1 to 16  R:100 to 117 Align Len (16) (75.00%) (87.50%)\n"
    "   Query:    3' gucgaaaguuuuac uagagug 5'\n"
    "                |||||||  |||||| \n"
    "   Ref:      5' uaguuuucacAAUGAUCUCGG 3'\n"
    "   Energy:  -20.470000 kCal/Mol\n"
    "Scores for this hit:\n"
    ">q0\tt0\t165.00\t-20.47\t1 16\t100 117\t16\t75.00%\t87.50%\n"
    ">>q0\tt0\t165.00\t-20.47\t165.00\t-20.47\t1\t21\t2282\t100\n"
)


# -- Parser unit tests (no binary required) ---------------------------------


def test_parse_extracts_all_fields_and_ignores_summary_line():
    """Every `>` field maps to the right attribute; the `>>` summary line is not a hit."""
    sites = _parse_miranda_output(SAMPLE_STDOUT, ["miR-bantam"])[0]

    assert len(sites) == 1  # the `>>` line is excluded
    s = sites[0]
    assert s.mirna_id == "miR-bantam"
    assert (s.score, s.energy) == (165.0, -20.47)
    assert (s.mirna_start, s.mirna_end) == (1, 16)
    assert (s.target_start, s.target_end) == (100, 117)
    assert s.alignment_length == 16
    assert (s.identity, s.similarity) == (75.0, 87.5)
    assert s.mirna_alignment and s.target_alignment and "|" in s.pairing


def test_parse_groups_hits_by_target_index():
    """`q{i}`/`t{j}` ids map back to positional indices, bucketing hits per target."""
    stdout = (
        ">q0\tt0\t150.00\t-18.00\t1 16\t100 117\t16\t70.00%\t80.00%\n"
        ">q0\tt1\t140.00\t-15.00\t2 17\t200 216\t15\t65.00%\t78.00%\n"
    )
    sites = _parse_miranda_output(stdout, ["miR-bantam"])

    assert set(sites) == {0, 1}
    assert sites[0][0].target_start == 100
    assert sites[1][0].target_start == 200


def test_parse_no_hits_returns_empty():
    """Output with no `>` hit lines parses to an empty mapping (not an error)."""
    assert _parse_miranda_output("miRanda v3.3a\nNo Hits Found above Threshold\n", ["miR-bantam"]) == {}


def test_parse_malformed_hit_line_raises():
    """A malformed single-`>` hit line raises rather than silently yielding no hits."""
    with pytest.raises(RuntimeError):
        # Space-delimited instead of tab-delimited: a format regression must surface.
        _parse_miranda_output(">q0 t0 165.00 -20.47 0.00 1 16 100 117 16\n", ["miR-bantam"])


def test_parse_unrecognized_internal_id_raises():
    """A well-formed `>` line whose ids aren't the wrapper's q{i}/t{j} surfaces as an error."""
    with pytest.raises(RuntimeError):
        # 9 tab fields (passes the field-count guard) but ids lack the q/t prefix.
        _parse_miranda_output(">x0\ty0\t165.00\t-20.47\t1 16\t100 117\t16\t75.00%\t87.50%\n", ["miR-bantam"])


# -- FASTA reader (example loading) -----------------------------------------


def test_read_fasta_keeps_ids_and_seqs_aligned_for_empty_records(tmp_path):
    """Trailing/consecutive headers with no sequence lines still yield equal-length lists."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">seq1\nACGT\n>seq2\n>seq3\nUU\n")
    ids, seqs = _read_fasta(fasta)
    assert ids == ["seq1", "seq2", "seq3"]
    assert seqs == ["ACGT", "", "UU"]  # empty record keeps its slot instead of shifting


# -- Input validation (custom validator) ------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_sequences": [], "mirna_queries": ["UGAGAU"]},  # empty targets
        {"target_sequences": ["ACGT"], "mirna_queries": []},  # empty queries
        {"target_sequences": ["ACGT"], "mirna_queries": ["UGAGAU"], "mirna_ids": ["a", "b"]},  # id length mismatch
    ],
)
def test_input_validation_rejects_bad_input(kwargs):
    """Empty sequence lists and mismatched mirna_ids are rejected at construction."""
    with pytest.raises(ValueError):
        MirandaInput(**kwargs)


# -- Output export (in-memory; no binary) -----------------------------------


def test_export_flattens_sites_to_csv_and_json(tmp_path):
    """CSV and JSON are both one flat row per site; targets with no sites contribute none."""
    output = MirandaOutput(
        results=[
            MirandaSequenceResult(
                target_id="hid",
                target_sequence="ACGU",
                target_sites=[
                    MirandaTargetSite(
                        mirna_id="miR-bantam",
                        score=161.0,
                        energy=-24.54,
                        target_start=1720,
                        target_end=1740,
                        mirna_start=2,
                        mirna_end=20,
                        alignment_length=18,
                        identity=88.89,
                        similarity=94.44,
                    )
                ],
            ),
            MirandaSequenceResult(target_id="no_hits", target_sequence="UUUU", target_sites=[]),
        ]
    )

    output.export(name="out", export_path=str(tmp_path), file_format="csv")
    output.export(name="out", export_path=str(tmp_path), file_format="json")

    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == 1  # only the one site; the empty target adds no row
    assert rows[0]["target_id"] == "hid"
    assert rows[0]["mirna_id"] == "miR-bantam"
    assert float(rows[0]["energy"]) == -24.54
    assert int(rows[0]["target_start"]) == 1720

    data = json.loads((tmp_path / "out.json").read_text())
    assert len(data) == 1  # flat per-site schema, identical to the CSV
    assert data[0]["target_id"] == "hid"
    assert data[0]["score"] == 161.0
    assert "target_sequence" not in data[0]  # raw target sequences are not embedded


# -- Integration (runs the compiled binary) ---------------------------------


@pytest.mark.integration
def test_miranda_scan_finds_bantam_site():
    """End-to-end: miR-bantam scanned against the Drosophila hid 3'UTR finds a real site."""
    _, mirna_seqs = _read_fasta(_EXAMPLES_DIR / "bantam_mirna.fasta")
    _, target_seqs = _read_fasta(_EXAMPLES_DIR / "hid_utr.fasta")

    result = run_miranda_scan(
        MirandaInput(target_sequences=target_seqs, mirna_queries=mirna_seqs, mirna_ids=["miR-bantam"]),
        MirandaConfig(),
    )

    assert result.success is True
    assert result.tool_id == "miranda-scan"
    assert len(result) == 1  # one result per input target
    assert result.total_sites >= 1

    top = result[0].target_sites[0]
    assert top.mirna_id == "miR-bantam"  # query id surfaced end-to-end
    assert top.energy < 0  # a favorable duplex
    assert 1 <= top.target_start <= top.target_end <= len(target_seqs[0])  # 1-indexed, in-bounds
