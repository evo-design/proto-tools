"""tests/gene_annotation_tests/test_meme_fimo_scan.py.

Tests for the MEME Suite FIMO motif scanning tool.
"""

from pathlib import Path

import pytest

from proto_tools.tools import (
    MEMEFimoScanConfig,
    MEMEFimoScanInput,
    run_meme_fimo_scan,
)
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_MEME_DIR = Path(__file__).parent.parent.parent / "proto_tools" / "tools" / "gene_annotation" / "meme"
EXAMPLE_MEME_FILE = _MEME_DIR / "examples" / "example.meme"
_LOCUS_FASTA = _MEME_DIR / "examples" / "spyogenes_crispr_locus.fasta"


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


# Every DNA fixture below is a verbatim slice of the bundled S. pyogenes SF370 CRISPR1
# locus, the same sequence example.meme's direct-repeat motif was derived from.
LOCUS = "".join(line for line in _LOCUS_FASTA.read_text().splitlines() if not line.startswith(">"))

# The 36 bp direct repeat, and the array's first two copies of it (separated by a 30 bp spacer).
_R1 = LOCUS.index("GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC")
_R2 = LOCUS.index("GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC", _R1 + 1)
CRISPR_REPEAT = LOCUS[_R1 : _R1 + 36]

# One repeat-spacer-repeat unit: the motif occurs twice on the forward strand.
SAMPLE_SEQUENCE = LOCUS[_R1 : _R2 + len(CRISPR_REPEAT)]

# 30 bp of the array's leader sequence ending flush with the first repeat's final base.
LEADER_THEN_REPEAT = LOCUS[_R1 - 30 : _R1 + len(CRISPR_REPEAT)]

# The minus strand of the first repeat-plus-spacer unit, so the repeat's reverse
# complement sits flush against the 3' end.
REPEAT_SPACER_MINUS = _reverse_complement(LOCUS[_R1:_R2])

# A 60 bp stretch well upstream of the array, carrying no occurrence of the repeat.
NO_MATCH_SEQUENCE = LOCUS[600:660]

# Human histone H3.1 N-terminal tail (UniProt P68431), minus the initiator Met that is
# removed in vivo — so positions match the standard residue numbering. The 'ARKS' motif
# occurs twice, at A7-S10 and A25-S28: the H3K9 and H3K27 methylation sites.
H3_TAIL = "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPH"
H3_MOTIF_CONSENSUS = "ARKS"


# ── Input validation ─────────────────────────────────────────────────────


def test_input_single_sequence_normalized_to_list():
    """A single sequence string is normalized to a one-element list."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    assert inputs.sequences == [SAMPLE_SEQUENCE]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fimo_scan_basic_execution():
    """FIMO finds the bundled motif in the sample DNA sequence."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.success is True
    assert result.tool_id == "meme-fimo-scan"
    assert len(result.results) == 1  # one bundle per input sequence
    assert result.num_matches >= 1

    match = result.results[0].matches[0]
    assert match.motif_id == "SPYO_CRISPR1"
    assert match.strand in {"+", "-"}
    assert 0 < match.pvalue <= 1e-4
    assert match.start <= match.stop


@pytest.mark.integration
def test_fimo_scan_results_align_to_inputs():
    """Output is 1:1 with inputs; a sequence with no occurrences yields an empty bundle."""
    inputs = MEMEFimoScanInput(sequences=[SAMPLE_SEQUENCE, NO_MATCH_SEQUENCE], motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.success is True
    assert len(result.results) == 2  # positionally aligned to the two inputs
    assert len(result.results[0].matches) >= 1  # motif present in sequence 0
    assert result.results[1].matches == []  # no motif in sequence 1 -> empty bundle


@pytest.mark.integration
def test_fimo_scan_single_strand_finds_forward_matches():
    """Disabling reverse-strand scanning still recovers the forward occurrences."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig(both_strands=False))

    assert result.success is True
    assert result.tool_id == "meme-fimo-scan"
    assert result.num_matches >= 1
    assert all(m.strand == "+" for r in result.results for m in r.matches)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("sequence", "strand"),
    [(LEADER_THEN_REPEAT, "+"), (REPEAT_SPACER_MINUS, "-")],
    ids=["forward", "reverse"],
)
def test_fimo_scan_finds_motif_flush_against_sequence_end(sequence, strand):
    """An occurrence ending at the final base is reported, on either strand.

    pymemesuite's sliding window stops one position short of the sequence end, so
    without a pad the last window is never scored and the hit is silently lost.
    """
    inputs = MEMEFimoScanInput(sequences=sequence, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.num_matches == 1
    match = result.results[0].matches[0]
    assert (match.start, match.stop) == (31, len(sequence))  # flush against the 3' end
    assert match.strand == strand


@pytest.mark.integration
def test_fimo_scan_sequence_exactly_motif_width_is_scanned():
    """A sequence exactly as long as the motif yields its single window's match."""
    inputs = MEMEFimoScanInput(sequences=CRISPR_REPEAT, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.num_matches == 1
    match = result.results[0].matches[0]
    assert (match.start, match.stop) == (1, len(CRISPR_REPEAT))


def _write_protein_motif(path: Path) -> None:
    """Write the histone H3 'ARKS' methylation-site motif in MEME format."""
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rows = "\n".join(" ".join("0.81" if c == dom else "0.01" for c in aa) for dom in H3_MOTIF_CONSENSUS)
    path.write_text(
        f"MEME version 4\n\nALPHABET= {aa}\n\n"
        f"MOTIF H3_ARKS meth_site\n"
        f"letter-probability matrix: alength= 20 w= {len(H3_MOTIF_CONSENSUS)} nsites= 20 E= 0\n{rows}\n"
    )


@pytest.mark.integration
def test_fimo_scan_protein_motif_is_forward_only(tmp_path):
    """both_strands=True is ignored for a protein (non-complementable) motif.

    Without the alphabet guard, pymemesuite emits spurious reverse-strand hits on
    protein; the tool must match the FIMO CLI and scan the given strand only.
    """
    motif = tmp_path / "prot.meme"
    _write_protein_motif(motif)
    inputs = MEMEFimoScanInput(sequences=H3_TAIL, motifs=str(motif))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig(both_strands=True, threshold=1e-3))

    assert result.success is True
    # The H3K9 and H3K27 sites, both on the given strand.
    assert [(m.start, m.stop) for m in result.results[0].matches] == [(7, 10), (25, 28)]
    assert all(m.strand == "+" for r in result.results for m in r.matches)


@pytest.mark.integration
def test_fimo_scan_export_csv(tmp_path):
    """Results export to a CSV file on disk."""
    inputs = MEMEFimoScanInput(sequences=SAMPLE_SEQUENCE, motifs=str(EXAMPLE_MEME_FILE))
    result = run_meme_fimo_scan(inputs, MEMEFimoScanConfig())

    assert result.success is True
    assert result.num_matches >= 1

    result.export(name="fimo_matches", export_path=tmp_path, file_format="csv")

    written = tmp_path / "fimo_matches"
    assert written.exists()
    assert written.stat().st_size > 0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark("meme-fimo-scan")
@pytest.mark.slow
def test_meme_fimo_scan_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark meme-fimo-scan: scan 15000 random 1 kb DNA sequences against the bundled motif (cold + warm)."""
    sequences = random_dna_sequences(n=15000, length=1000, seed=0)
    inputs = MEMEFimoScanInput(sequences=sequences, motifs=str(EXAMPLE_MEME_FILE))
    config = MEMEFimoScanConfig()

    result = benchmark_twice(request, "meme", lambda: run_meme_fimo_scan(inputs, config))
    validate_output(result)

    assert result.tool_id == "meme-fimo-scan"
    assert result.success is True
    assert len(result.results) == 15000  # one bundle per input sequence
    assert result.num_matches > 0
